"""Controlled feature-selection comparison around fold-local promising models."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.feature_selection import f_classif

from analysis.optuna_recall import optimal_threshold, save_probability, write_json
from analysis.search_recall import build_model, feature_columns
from analysis.train_safe import ROOT, SEED, TrainOnlyCleaner, checked_splits, make_features, metrics

FAMILIES = ('lightgbm', 'xgboost')
COUNTS = (256, 512, 768, 1024, 1536)


def load_templates(source):
    """Select from each outer fold's own inner results, never outer scores."""
    templates = {}
    storage = f'sqlite:///{(source / "studies.sqlite3").resolve()}'
    for fold in range(5):
        study = optuna.load_study(study_name=f'outer_{fold}_inner', storage=storage)
        ranked = sorted(study.get_trials(states=(TrialState.COMPLETE,)), key=lambda t: (-t.value, t.number))
        templates[str(fold)] = []
        for family in FAMILIES:
            trial = next(t for t in ranked if t.user_attrs['spec']['family'] == family)
            spec = copy.deepcopy(trial.user_attrs['spec'])
            spec.update(top_k=0, source_trial=trial.number)
            templates[str(fold)].append(spec)
    return templates


def variant_specs(template, counts=COUNTS):
    return [dict(method='none', top_k=0)] + [dict(method=method, top_k=k)
                for method in ('anova', 'gain') for k in counts]


def estimator(template, jobs):
    model = build_model(dict(template, top_k=0), jobs)['model']
    model.set_params(importance_type='gain')
    return model


def prepare(X, y, train, valid, template, jobs):
    """Cleaner and both supervised rankings see training rows only."""
    columns = feature_columns(X, template)
    clean = TrainOnlyCleaner().fit(X.iloc[train][columns])
    assert set(clean.fit_rows_) == set(train)
    assert not set(train) & set(valid)
    a, b = clean.transform(X.iloc[train][columns]), clean.transform(X.iloc[valid][columns])
    weights = np.where(y[train] == 0, template['toxic_weight'], 1.)
    base = estimator(template, jobs).fit(a, y[train], sample_weight=weights)
    statistics, _ = f_classif(a, y[train])
    # Stable ties follow original input column order; +inf remains highest.
    statistics = np.nan_to_num(statistics, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    ranks = dict(anova=np.argsort(-statistics, kind='stable'),
                 gain=np.argsort(-np.asarray(base.feature_importances_), kind='stable'))
    return a, b, weights, base, ranks, clean.columns_


def selected_indices(ranks, method, k, width):
    if method == 'none':
        return np.arange(width)
    return np.sort(ranks[method][:min(k, width)])


def fit_variants(X, y, train, valid, template, variants, jobs, destination=None):
    a, b, weights, base, ranks, columns = prepare(X, y, train, valid, template, jobs)
    results, feature_lists = {}, {}
    for variant in variants:
        name = f'{variant["method"]}_{variant["top_k"]}'
        selected = selected_indices(ranks, variant['method'], variant['top_k'], a.shape[1])
        if variant['method'] == 'none':
            model = base
        else:
            model = estimator(template, jobs).fit(a[:, selected], y[train], sample_weight=weights)
        probability = model.predict_proba(b[:, selected])[:, 1]
        assert np.isfinite(probability).all()
        results[name] = probability
        feature_lists[name] = [columns[i] for i in selected]
    if destination:
        write_json(destination, dict(train_ids=train.tolist(), valid_ids=valid.tolist(),
                                     selected_features=feature_lists))
    return results


def choose_candidate(rows):
    return min(rows, key=lambda r: (-r['f1_non_toxic'], r['candidate']))


def run_focused(X, y, groups, templates, out, target=.87, jobs=4, counts=COUNTS, previous_audit=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ids = np.arange(len(y)); audit = []; rows = []; control_rows = []; prediction_rows = []
    for fold, (train, valid) in enumerate(checked_splits(ids, y, groups, SEED, 'outer', audit)):
        print(f'Outer {fold+1}/5: focused feature selection', flush=True)
        scope = f'outer_{fold}_inner'
        splits = list(checked_splits(train, y, groups, SEED+fold+1, scope, audit))
        if previous_audit is not None:
            for entry in [e for e in audit if e['scope'] in ('outer', scope)]:
                original = next(e for e in previous_audit if e['scope'] == entry['scope'] and e['fold'] == entry['fold'])
                assert original['train_ids'] == entry['train_ids'] and original['valid_ids'] == entry['valid_ids']
        assert all(not set(a.tolist()+b.tolist()) & set(valid) for a,b in splits)
        candidates, vectors = [], {}
        bases = {s['family']: s for s in templates[str(fold)]}
        for family, template in bases.items():
            variants = variant_specs(template, counts)
            names = [f'{v["method"]}_{v["top_k"]}' for v in variants]
            probability = {name: np.full(len(y), np.nan) for name in names}
            for inner_fold, (a,b) in enumerate(splits):
                cache = out/f'{scope}_{family}_{inner_fold}.npz'
                if cache.exists():
                    with np.load(cache) as saved:
                        batch = {n: saved[n] for n in names}
                else:
                    print(f'  {family}, inner Fold {inner_fold+1}/5: {len(variants)} variants', flush=True)
                    batch = fit_variants(X,y,a,b,template,variants,jobs,
                                         out/f'{scope}_{family}_{inner_fold}_features.json')
                    temp = cache.with_suffix('.pending.npz')
                    np.savez_compressed(temp, **batch); temp.replace(cache)
                for name in names:
                    assert batch[name].shape == (len(b),)
                    probability[name][b] = batch[name]
            for variant in variants:
                name = f'{variant["method"]}_{variant["top_k"]}'
                candidate = f'{family}_{name}'
                p = probability[name]
                assert np.isfinite(p[train]).all() and np.isnan(p[valid]).all()
                vectors[candidate] = p
                score = optimal_threshold(y[train], p[train], target)
                candidates.append(dict(candidate=candidate, members=[dict(family=family, **variant)], **score))
                print(f'  {candidate}: inner F1={score["f1_non_toxic"]:.4f}, Recall={score["recall_toxic"]:.4f}', flush=True)
        control_candidates = [r for r in candidates if r['members'][0]['method'] == 'none']
        # Evaluate two family-diverse blends: best selectors, and no-selection control.
        for kind, pool in [('selected_blend', candidates.copy()), ('control_blend', control_candidates.copy())]:
            best = [choose_candidate([r for r in pool if r['members'][0]['family'] == f]) for f in bases]
            blended = np.mean([vectors[r['candidate']] for r in best], axis=0)
            score = optimal_threshold(y[train], blended[train], target)
            row = dict(candidate=kind, members=[r['members'][0] for r in best], **score)
            candidates.append(row)
            if kind == 'control_blend':
                control_candidates.append(row)
        chosen, control = choose_candidate(candidates), choose_candidate(control_candidates)
        assert chosen['recall_toxic'] >= target and control['recall_toxic'] >= target
        write_json(out/f'outer_{fold}_locked.json', dict(selected=chosen, control=control, templates=bases))
        pd.DataFrame(candidates).sort_values('f1_non_toxic', ascending=False).to_csv(out/f'{scope}_leaderboard.csv',index=False)
        # Fit selected members and control once each; share identical fits.
        needed = {(m['family'],m['method'],m['top_k']) for r in (chosen,control) for m in r['members']}
        outer_p = {}
        for family in bases:
            variants = [dict(method=method,top_k=k) for f,method,k in sorted(needed) if f==family]
            batch = fit_variants(X,y,train,valid,bases[family],variants,jobs,
                                 out/f'outer_{fold}_{family}_features.json')
            for variant in variants:
                key = (family,variant['method'],variant['top_k'])
                outer_p[key] = batch[f'{variant["method"]}_{variant["top_k"]}']
        predictions = {}
        for name, choice, sink in [('selected',chosen,rows), ('control',control,control_rows)]:
            p = np.mean([outer_p[(m['family'],m['method'],m['top_k'])] for m in choice['members']],axis=0)
            predictions[name] = (p >= choice['threshold']).astype(int)
            sink.append(dict(fold=fold,candidate=choice['candidate'],threshold=choice['threshold'],
                             **metrics(y[valid],p,choice['threshold'])))
        prediction_rows.append(pd.DataFrame(dict(source_row=valid,fold=fold,label=y[valid],
                                                 selected_prediction=predictions['selected'],
                                                 control_prediction=predictions['control'])))
        pd.DataFrame(rows).to_csv(out/'outer_metrics.csv',index=False)
        pd.DataFrame(control_rows).to_csv(out/'control_outer_metrics.csv',index=False)
        pd.concat(prediction_rows).sort_values('source_row').to_csv(out/'outer_predictions.csv',index=False)
        write_json(out/'split_audit.json',audit)
        print(f'Outer {fold+1}: selected={rows[-1]}, control={control_rows[-1]}',flush=True)
    result = dict(recall_target=target,f1_target=.85,selection_counts=list(counts),
                  limitations=['internal development evaluation after prior adaptive exploration',
                               'templates selected only from each matching outer fold inner trials',
                               'outer recall is not guaranteed; no full-data deployment refit'])
    for name, values in [('selected',rows),('control',control_rows)]:
        d = pd.DataFrame(values)
        result[name] = dict(mean_f1_non_toxic=float(d.f1_non_toxic.mean()),
                            std_f1_non_toxic=float(d.f1_non_toxic.std(ddof=0)),
                            mean_recall_toxic=float(d.recall_toxic.mean()),
                            min_recall_toxic=float(d.recall_toxic.min()),
                            folds_recall_target=int((d.recall_toxic>=target).sum()),
                            mean_targets_met=bool(d.f1_non_toxic.mean()>=.85 and d.recall_toxic.mean()>=target))
    result['f1_change_vs_control'] = result['selected']['mean_f1_non_toxic']-result['control']['mean_f1_non_toxic']
    result['recall_change_vs_control'] = result['selected']['mean_recall_toxic']-result['control']['mean_recall_toxic']
    assert pd.concat(prediction_rows).source_row.nunique()==len(y)
    write_json(out/'evaluation.json',result)
    return result


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'Data/training/optuna_recall87_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'Data/training/focused_features_recall87_v1')
    parser.add_argument('--recall-target',type=float,default=.87)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not 0<args.recall_target<=1 or args.jobs<1: parser.error('Invalid target or jobs')
    source=args.source.resolve();out=args.output.resolve()
    prior=json.loads((source/'run_config.json').read_text())
    digest=hashlib.sha256((ROOT/'Data/input_data/train.csv').read_bytes()).hexdigest()
    if prior['source_sha256']!=digest: raise ValueError('Source dataset changed')
    if prior['seed']!=SEED or prior['outer_folds']!=5 or prior['inner_folds']!=5:
        raise ValueError('Source split configuration differs')
    templates=load_templates(source)
    code=b''.join((ROOT/f'analysis/{n}.py').read_bytes() for n in
                  ['focused_features','search_recall','optuna_recall','train_safe'])
    config=dict(source_sha256=digest,source_study=str(source),templates=templates,seed=SEED,
                code_sha256=hashlib.sha256(code).hexdigest(),recall_target=args.recall_target,
                f1_target=.85,outer_folds=5,inner_folds=5,counts=list(COUNTS),jobs=args.jobs,
                importance='gain for both model families; training-fold ranking only',
                versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','lightgbm','xgboost','optuna','rdkit']})
    if args.dry_run:
        print(json.dumps(config,indent=2));print('22 single-model variants + 2 blends per outer fold; <=572 model fits');return
    if out.exists() and any(out.iterdir()):
        if not args.resume: raise FileExistsError('Use --resume or a new output path')
        if json.loads((out/'run_config.json').read_text())!=config: raise ValueError('Resume configuration changed')
        if (out/'evaluation.json').exists(): print((out/'evaluation.json').read_text());return
    out.mkdir(parents=True,exist_ok=True);write_json(out/'run_config.json',config)
    previous_audit=json.loads((source/'split_audit.json').read_text())
    raw=pd.read_csv(ROOT/'Data/input_data/train.csv')
    if not raw.label.isin([0,1]).all(): raise ValueError('Expected binary labels')
    X,groups,canonical=make_features(raw);y=raw.label.to_numpy(dtype=int)
    if pd.DataFrame(dict(canonical=canonical,label=y)).groupby('canonical').label.nunique().gt(1).any():
        raise ValueError('Conflicting labels')
    result=run_focused(X,y,groups,templates,out,args.recall_target,args.jobs,previous_audit=previous_audit)
    assert hashlib.sha256((ROOT/'Data/input_data/train.csv').read_bytes()).hexdigest()==digest
    comparisons=[]
    for label,path in [('previous_optuna',source/'outer_metrics.csv'),('focused_control',out/'control_outer_metrics.csv'),('focused_selection',out/'outer_metrics.csv')]:
        frame=pd.read_csv(path)
        comparisons.append(dict(experiment=label,mean_f1_non_toxic=frame.f1_non_toxic.mean(),
                                mean_recall_toxic=frame.recall_toxic.mean(),min_recall_toxic=frame.recall_toxic.min()))
    pd.DataFrame(comparisons).to_csv(out/'comparison.csv',index=False)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
