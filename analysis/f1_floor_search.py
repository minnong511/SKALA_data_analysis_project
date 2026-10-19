"""Maximize toxic recall subject to non-toxic F1 >= floor, with explicit infeasibility."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution
from optuna.trial import TrialState, create_trial

from analysis.optuna_recall import save_probability, suggest_spec, write_json
from analysis.search_recall import fit_predict
from analysis.train_safe import ROOT, SEED, checked_splits, make_features, metrics


def select_threshold(y, probability, floor=.82):
    y, probability = np.asarray(y), np.asarray(probability)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(probability).all():
        raise ValueError('Both classes and finite probabilities required')
    if not 0 < floor <= 1:
        raise ValueError('F1 floor must be in (0,1]')
    order = np.argsort(probability, kind='stable')
    p, labels = probability[order], y[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(p))+1, len(y)]
    fn = np.r_[0, np.cumsum(labels == 1)][cuts]
    tn = cuts - fn
    tp, fp = (y == 1).sum()-fn, (y == 0).sum()-tn
    f1 = 2*tp/(2*tp+fp+fn)
    recall = tn/(y == 0).sum()
    thresholds = np.r_[p[cuts[:-1]], np.nextafter(p[-1], np.inf)]
    feasible = np.flatnonzero(f1 >= floor)
    if len(feasible):
        best = min(feasible, key=lambda i: (-recall[i], -f1[i], abs(thresholds[i]-.5), thresholds[i]))
    else:
        # A diagnostic fallback, not a candidate that meets the floor.
        best = min(range(len(cuts)), key=lambda i: (-f1[i], -recall[i], abs(thresholds[i]-.5), thresholds[i]))
    threshold = float(thresholds[best])
    chosen = dict(threshold=threshold, **metrics(y, probability, threshold),
                  f1_floor_met=bool(len(feasible)), max_attainable_f1=float(f1.max()))
    chosen['objective'] = (chosen['recall_toxic'] if chosen['f1_floor_met'] else
                           chosen['max_attainable_f1']-floor)
    return chosen


class FocusedTrial:
    """Keep historical parameter names, narrowing only the family distribution."""
    def __init__(self, trial):
        self.trial = trial
        self.number = trial.number

    def suggest_categorical(self, name, choices):
        if name == 'family':
            choices = ['lightgbm', 'xgboost']
        return self.trial.suggest_categorical(name, choices)

    def __getattr__(self, name):
        return getattr(self.trial, name)


def focused_spec(trial):
    return suggest_spec(FocusedTrial(trial))


def study_for(out, scope, seed):
    state = out/f'{scope}_sampler.joblib'
    sampler = joblib.load(state) if state.exists() else optuna.samplers.TPESampler(seed=seed, n_startup_trials=8)
    return optuna.create_study(study_name=scope, storage=f'sqlite:///{(out/"studies.sqlite3").resolve()}',
                               direction='maximize', sampler=sampler, load_if_exists=True,
                               pruner=optuna.pruners.NopPruner())


def seed_cached(study, source, scope, y, ids, floor):
    """Re-score only this outer fold's completed inner OOF, no outer labels used."""
    if study.user_attrs.get('cache_import_complete'):
        return
    imported = {t.user_attrs.get('source_trial') for t in study.trials if t.user_attrs.get('cached')}
    old = optuna.load_study(study_name=scope, storage=f'sqlite:///{(source/"studies.sqlite3").resolve()}')
    for previous in old.get_trials(states=(TrialState.COMPLETE,)):
        if previous.number in imported:
            continue
        spec = previous.user_attrs['spec']
        if spec['family'] not in {'lightgbm', 'xgboost'}:
            continue
        path = source/f'{scope}_trial_{previous.number:04d}.npz'
        with np.load(path) as saved:
            probability = saved['probability']
        assert np.isfinite(probability[ids]).all()
        assert np.isnan(probability[np.setdiff1d(np.arange(len(y)),ids)]).all()
        chosen = select_threshold(y[ids],probability[ids],floor)
        distributions = copy.deepcopy(previous.distributions)
        distributions['family'] = CategoricalDistribution(['lightgbm','xgboost'])
        study.add_trial(create_trial(params=previous.params,distributions=distributions,
                                    value=chosen['objective'],
                                    user_attrs=dict(spec=spec,inner_metrics=chosen,cached=True,
                                                    probability_path=str(path.resolve()),source_trial=previous.number)))
    study.set_user_attr('cache_import_complete',True)


def tune(X,y,groups,ids,source,out,scope,seed,audit,floor,extra_trials,jobs):
    splits=list(checked_splits(ids,y,groups,seed,scope,audit))
    study=study_for(out,scope,seed)
    seed_cached(study,source,scope,y,ids,floor)
    for trial in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(trial.number,state=TrialState.FAIL);study.enqueue_trial(trial.params)

    def objective(trial):
        spec=focused_spec(trial);trial.set_user_attr('spec',spec);trial.set_user_attr('cached',False)
        probability=np.full(len(y),np.nan)
        for train,valid in splits:
            _,probability[valid]=fit_predict(X,y,train,valid,spec,jobs)
        score=select_threshold(y[ids],probability[ids],floor)
        trial.set_user_attr('inner_metrics',score)
        path=out/f'{scope}_trial_{trial.number:04d}.npz';save_probability(path,probability)
        trial.set_user_attr('probability_path',str(path.resolve()))
        return score['objective']

    def checkpoint(study,trial):
        study.trials_dataframe().to_csv(out/f'{scope}_trials.csv',index=False)
        state=out/f'{scope}_sampler.joblib';temp=state.with_suffix('.pending.joblib')
        joblib.dump(study.sampler,temp);temp.replace(state)
        if trial.state==TrialState.COMPLETE:
            m=trial.user_attrs['inner_metrics']
            print(f'{scope} trial {trial.number}: F1={m["f1_non_toxic"]:.4f}, '
                  f'Recall={m["recall_toxic"]:.4f}, floor_met={m["f1_floor_met"]}',flush=True)
    completed_new=sum(t.state==TrialState.COMPLETE and not t.user_attrs.get('cached',False) for t in study.trials)
    remaining=max(0,extra_trials-completed_new)
    if remaining:
        study.optimize(objective,n_trials=remaining,n_jobs=1,callbacks=[checkpoint])
    study.trials_dataframe().to_csv(out/f'{scope}_trials.csv',index=False)
    trials=study.get_trials(states=(TrialState.COMPLETE,))
    ranked=sorted(trials,key=lambda t:(-t.value,t.number))
    best=ranked[0]
    chosen=dict(candidate=f'trial_{best.number:04d}',specs=[best.user_attrs['spec']],**best.user_attrs['inner_metrics'])
    diverse=[]
    for family in ['lightgbm','xgboost']:
        match=next((t for t in ranked if t.user_attrs['spec']['family']==family),None)
        if match is not None:diverse.append(match)
    if len(diverse)==2:
        vectors=[]
        for trial in diverse:
            with np.load(trial.user_attrs['probability_path']) as saved:vectors.append(saved['probability'])
        blend=np.mean(vectors,axis=0)
        candidate=dict(candidate='diverse_blend',specs=[t.user_attrs['spec'] for t in diverse],
                       **select_threshold(y[ids],blend[ids],floor))
        write_json(out/f'{scope}_blend.json',candidate)
        if candidate['objective']>chosen['objective']:chosen=candidate
    write_json(out/f'{scope}_selected.json',chosen)
    return chosen


def run(X,y,groups,source,out,floor=.82,extra_trials=12,jobs=4,previous_audit=None):
    ids=np.arange(len(y));audit=[];rows=[];predictions=[]
    for fold,(train,valid) in enumerate(checked_splits(ids,y,groups,SEED,'outer',audit)):
        scope=f'outer_{fold}_inner'
        print(f'Outer {fold+1}/5: F1 floor={floor}, additional trials={extra_trials}',flush=True)
        # Verify fold identity BEFORE importing any old prediction vectors.
        planned=list(checked_splits(train,y,groups,SEED+fold+1,scope,[]))
        if previous_audit is not None:
            original=next(e for e in previous_audit if e['scope']=='outer' and e['fold']==fold)
            assert original['train_ids']==train.tolist() and original['valid_ids']==valid.tolist()
            for k,(a,b) in enumerate(planned):
                original=next(e for e in previous_audit if e['scope']==scope and e['fold']==k)
                assert original['train_ids']==a.tolist() and original['valid_ids']==b.tolist()
        chosen=tune(X,y,groups,train,source,out,scope,SEED+fold+1,audit,floor,extra_trials,jobs)
        assert all(not set(e['train_ids']+e['valid_ids'])&set(valid) for e in audit if e['scope']==scope)
        write_json(out/f'outer_{fold}_locked.json',chosen)
        probability=np.mean([fit_predict(X,y,train,valid,spec,jobs)[1] for spec in chosen['specs']],axis=0)
        score=metrics(y[valid],probability,chosen['threshold'])
        rows.append(dict(fold=fold,candidate=chosen['candidate'],threshold=chosen['threshold'],
                         inner_floor_met=chosen['f1_floor_met'],outer_floor_met=score['f1_non_toxic']>=floor,**score))
        predictions.append(pd.DataFrame(dict(source_row=valid,fold=fold,label=y[valid],
                                            probability_non_toxic=probability,prediction=(probability>=chosen['threshold']).astype(int))))
        pd.DataFrame(rows).to_csv(out/'outer_metrics.csv',index=False)
        pd.concat(predictions).sort_values('source_row').to_csv(out/'outer_predictions.csv',index=False)
        write_json(out/'split_audit.json',audit)
        print(f'Outer result: {rows[-1]}',flush=True)
    frame=pd.DataFrame(rows)
    assert pd.concat(predictions).source_row.nunique()==len(y)
    result=dict(f1_floor=floor,f1_aspiration=.83,additional_trials_per_outer_fold=extra_trials,
                mean_f1_non_toxic=float(frame.f1_non_toxic.mean()),mean_recall_toxic=float(frame.recall_toxic.mean()),
                std_f1_non_toxic=float(frame.f1_non_toxic.std(ddof=0)),min_f1_non_toxic=float(frame.f1_non_toxic.min()),
                min_recall_toxic=float(frame.recall_toxic.min()),inner_folds_meeting_floor=int(frame.inner_floor_met.sum()),
                outer_folds_meeting_floor=int(frame.outer_floor_met.sum()),mean_f1_floor_met=bool(frame.f1_non_toxic.mean()>=floor),
                mean_f1_aspiration_met=bool(frame.f1_non_toxic.mean()>=.83),
                limitations=['internal development evaluation after prior adaptive exploration',
                             'infeasible inner candidates use a diagnostic max-F1 fallback, explicitly flagged',
                             'outer F1 floor not guaranteed; no full-data deployment refit'])
    write_json(out/'evaluation.json',result);return result


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--f1-floor',type=float,default=.82)
    parser.add_argument('--trials',type=int,default=12,help='Additional completed trials per outer fold; cached trials excluded')
    parser.add_argument('--source',type=Path,default=ROOT/'Data/training/optuna_recall87_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'Data/training/f1floor82_recall_v1')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not 0<args.f1_floor<=1 or args.trials<0 or args.jobs<1:parser.error('Invalid floor, budget or jobs')
    source,out=args.source.resolve(),args.output.resolve()
    prior=json.loads((source/'run_config.json').read_text())
    digest=hashlib.sha256((ROOT/'Data/input_data/train.csv').read_bytes()).hexdigest()
    if prior['source_sha256']!=digest or prior['seed']!=SEED or prior['inner_folds']!=5 or prior['outer_folds']!=5:
        raise ValueError('Source dataset or split configuration changed')
    # Cached probabilities must come from the same model/preprocessing implementation.
    original=b''.join((ROOT/f'analysis/{n}.py').read_bytes() for n in ['optuna_recall','search_recall','train_safe'])
    if hashlib.sha256(original).hexdigest()!=prior['code_sha256']:raise ValueError('Source prediction code changed')
    code=Path(__file__).read_bytes()+original
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','lightgbm','xgboost','rdkit','optuna']}
    if versions!=prior['versions']:raise ValueError('Source package versions changed')
    config=dict(f1_floor=args.f1_floor,f1_aspiration=.83,objective='maximize toxic recall subject to non-toxic F1 floor',
                fallback='maximize attainable F1 when infeasible; never count fallback as success',source_study=str(source),
                source_db_sha256=hashlib.sha256((source/'studies.sqlite3').read_bytes()).hexdigest(),
                source_sha256=digest,code_sha256=hashlib.sha256(code).hexdigest(),seed=SEED,inner_folds=5,outer_folds=5,
                jobs=args.jobs,versions=versions,pruning='disabled; evaluate full inner OOF for F1 feasibility')
    if args.dry_run:print(json.dumps(dict(config,additional_trials_per_outer_fold=args.trials,max_new_fits=25*args.trials+10),indent=2));return
    if out.exists() and any(out.iterdir()):
        if not args.resume:raise FileExistsError('Use --resume or a new output directory')
        if json.loads((out/'run_config.json').read_text())!=config:raise ValueError('Resume configuration changed')
        if (out/'evaluation.json').exists():
            old=json.loads((out/'evaluation.json').read_text())
            if old['additional_trials_per_outer_fold']>=args.trials:print(json.dumps(old,indent=2));return
            (out/'evaluation.json').replace(out/'previous_evaluation.json')
    out.mkdir(parents=True,exist_ok=True);write_json(out/'run_config.json',config)
    write_json(out/'requested_budget.json',dict(additional_trials_per_outer_fold=args.trials,max_new_fits=25*args.trials+10))
    raw=pd.read_csv(ROOT/'Data/input_data/train.csv')
    if not raw.label.isin([0,1]).all():raise ValueError('Invalid labels')
    X,groups,canonical=make_features(raw);y=raw.label.to_numpy(dtype=int)
    if pd.DataFrame(dict(canonical=canonical,label=y)).groupby('canonical').label.nunique().gt(1).any():raise ValueError('Conflicting labels')
    result=run(X,y,groups,source,out,args.f1_floor,args.trials,args.jobs,json.loads((source/'split_audit.json').read_text()))
    assert hashlib.sha256((ROOT/'Data/input_data/train.csv').read_bytes()).hexdigest()==digest
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
