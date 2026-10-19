"""Optuna TPE search: maximize non-toxic F1 subject to toxic recall >= target."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.metrics import f1_score, recall_score

from analysis.search_recall import fit_predict
from analysis.train_safe import ROOT, SEED, checked_splits, make_features, metrics


def write_json(path, value):
    temp = path.with_suffix('.pending.json')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def optimal_threshold(y, probability, target):
    """Exact observed-score boundaries, with ties kept together, O(n log n)."""
    y, probability = np.asarray(y), np.asarray(probability)
    if not np.isfinite(probability).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('Both classes and finite probabilities are required')
    order = np.argsort(probability, kind='stable')
    scores, labels = probability[order], y[order]
    # Cut index = number predicted toxic; ties cannot be split by a threshold.
    cuts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1, len(y)]
    non_toxic_below = np.r_[0, np.cumsum(labels == 1)][cuts]
    toxic_below = cuts - non_toxic_below
    toxic_total, non_toxic_total = int((y == 0).sum()), int((y == 1).sum())
    tp = non_toxic_total - non_toxic_below
    fp = toxic_total - toxic_below
    fn = non_toxic_below
    f1 = 2 * tp / (2 * tp + fp + fn)
    recall = toxic_below / toxic_total
    thresholds = np.r_[scores[cuts[:-1]], np.nextafter(scores[-1], np.inf)]
    eligible = np.flatnonzero(recall >= target)
    best = min(eligible, key=lambda i: (-f1[i], abs(thresholds[i]-.5), thresholds[i]))
    threshold = float(thresholds[best])
    return dict(threshold=threshold, **metrics(y, probability, threshold))


def suggest_spec(trial):
    family = trial.suggest_categorical('family', ['lightgbm', 'xgboost', 'extra_trees', 'logistic'])
    spec = dict(id=f'trial_{trial.number:04d}', family=family,
                features=trial.suggest_categorical('features', ['all', 'ecfp', 'fcfp', 'ptfp', 'ecfp_fcfp']),
                top_k=trial.suggest_categorical('top_k', [0, 256, 512, 1024]),
                toxic_weight=trial.suggest_float('toxic_weight', .75, 5., log=True))
    if family == 'lightgbm':
        params = dict(n_estimators=trial.suggest_int('lgb_trees', 200, 800, step=100),
                      learning_rate=trial.suggest_float('lgb_lr', .02, .15, log=True),
                      num_leaves=trial.suggest_int('lgb_leaves', 15, 63),
                      min_child_samples=trial.suggest_int('lgb_min_rows', 15, 100),
                      colsample_bytree=trial.suggest_float('lgb_columns', .5, 1.),
                      reg_alpha=trial.suggest_float('lgb_l1', 1e-4, 10., log=True),
                      reg_lambda=trial.suggest_float('lgb_l2', 1e-3, 30., log=True))
    elif family == 'xgboost':
        params = dict(n_estimators=trial.suggest_int('xgb_trees', 200, 800, step=100),
                      learning_rate=trial.suggest_float('xgb_lr', .02, .15, log=True),
                      max_depth=trial.suggest_int('xgb_depth', 3, 8),
                      min_child_weight=trial.suggest_float('xgb_min_weight', 1., 20., log=True),
                      colsample_bytree=trial.suggest_float('xgb_columns', .5, 1.),
                      subsample=trial.suggest_float('xgb_rows', .65, 1.),
                      reg_lambda=trial.suggest_float('xgb_l2', .1, 30., log=True))
    elif family == 'extra_trees':
        params = dict(n_estimators=trial.suggest_int('et_trees', 200, 600, step=100),
                      min_samples_leaf=trial.suggest_int('et_min_rows', 1, 10),
                      max_features=trial.suggest_categorical('et_columns', ['sqrt', 'log2', .2]),
                      max_depth=trial.suggest_categorical('et_depth', [None, 12, 24, 40]))
    else:
        params = dict(C=trial.suggest_float('logistic_c', .001, 30., log=True), max_iter=2000)
    spec['params'] = params
    return spec


def save_probability(path, probability):
    temp = path.with_suffix('.pending.npz')
    np.savez_compressed(temp, probability=probability)
    temp.replace(path)


def tune_inner(X, y, groups, ids, target, budget, out, scope, seed, audit, jobs, spec_factory=suggest_spec):
    splits = list(checked_splits(ids, y, groups, seed, scope, audit))
    storage = f'sqlite:///{(out / "studies.sqlite3").resolve()}'
    state_path = out/f'{scope}_sampler.joblib'
    sampler = (joblib.load(state_path) if state_path.exists() else
               optuna.samplers.TPESampler(seed=seed, n_startup_trials=8))
    study = optuna.create_study(study_name=scope, storage=storage, direction='maximize',
                               sampler=sampler, load_if_exists=True,
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2))
    if not study.trials:
        for family in ['lightgbm', 'xgboost', 'extra_trees', 'logistic']:
            study.enqueue_trial(dict(family=family, features='all', top_k=0, toxic_weight=1.))
    # A killed process leaves RUNNING trials. They are marked failed and retried.
    for frozen in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(frozen.number, state=TrialState.FAIL)
        study.enqueue_trial(frozen.params)

    def objective(trial):
        spec = spec_factory(trial)
        trial.set_user_attr('spec', spec)
        probability = np.full(len(y), np.nan)
        used = []
        for step, (train, valid) in enumerate(splits):
            _, probability[valid] = fit_predict(X, y, train, valid, spec, jobs)
            used.extend(valid.tolist())
            partial = optimal_threshold(y[used], probability[used], target)
            trial.report(partial['f1_non_toxic'], step)
            if step < len(splits)-1 and trial.should_prune():
                raise optuna.TrialPruned()
        assert np.isfinite(probability[ids]).all()
        assert np.isnan(probability[np.setdiff1d(np.arange(len(y)), ids)]).all()
        chosen = optimal_threshold(y[ids], probability[ids], target)
        assert chosen['recall_toxic'] >= target
        trial.set_user_attr('inner_metrics', chosen)
        save_probability(out/f'{scope}_trial_{trial.number:04d}.npz', probability)
        return chosen['f1_non_toxic']

    def checkpoint(study, trial):
        study.trials_dataframe().to_csv(out/f'{scope}_trials.csv', index=False)
        temp = state_path.with_suffix('.pending.joblib')
        joblib.dump(study.sampler, temp)
        temp.replace(state_path)
        if trial.state == TrialState.COMPLETE:
            m = trial.user_attrs['inner_metrics']
            print(f'{scope} trial {trial.number}: F1={trial.value:.4f}, '
                  f'Recall={m["recall_toxic"]:.4f}, model={trial.user_attrs["spec"]["family"]}', flush=True)

    attempted = len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED)))
    remaining = max(0, budget-attempted)
    if remaining:
        study.optimize(objective, n_trials=remaining, callbacks=[checkpoint], n_jobs=1)
    complete = study.get_trials(states=(TrialState.COMPLETE,))
    if not complete:
        raise RuntimeError('No completed trials; increase budget or inspect errors')
    ranked = sorted(complete, key=lambda t: (-t.value, t.number))
    best = ranked[0]
    chosen = dict(candidate=f'trial_{best.number:04d}', specs=[best.user_attrs['spec']],
                  **best.user_attrs['inner_metrics'])
    # Compare a family-diverse blend using only complete inner OOF predictions.
    diverse, families = [], set()
    for trial in ranked:
        family = trial.user_attrs['spec']['family']
        if family not in families:
            diverse.append(trial); families.add(family)
        if len(diverse) == 3:
            break
    if len(diverse) >= 2:
        vectors = []
        for trial in diverse:
            with np.load(out/f'{scope}_trial_{trial.number:04d}.npz') as saved:
                vectors.append(saved['probability'])
        blended = np.mean(vectors, axis=0)
        blend_metrics = optimal_threshold(y[ids], blended[ids], target)
        blend = dict(candidate='diverse_blend', specs=[t.user_attrs['spec'] for t in diverse], **blend_metrics)
        write_json(out/f'{scope}_blend.json', blend)
        if blend['f1_non_toxic'] > chosen['f1_non_toxic']:
            chosen = blend
    write_json(out/f'{scope}_selected.json', chosen)
    return chosen


def run_nested(X, y, groups, out, target=.87, budget=20, jobs=4, spec_factory=suggest_spec):
    ids = np.arange(len(y)); audit = []; rows = []; predictions = []
    for fold, (train, valid) in enumerate(checked_splits(ids, y, groups, SEED, 'outer', audit)):
        scope = f'outer_{fold}_inner'
        print(f'Outer {fold+1}/5, inner budget={budget}', flush=True)
        chosen = tune_inner(X, y, groups, train, target, budget, out, scope,
                            SEED+fold+1, audit, jobs, spec_factory)
        assert all(not set(e['train_ids'] + e['valid_ids']) & set(valid)
                   for e in audit if e['scope'] == scope)
        write_json(out/f'outer_{fold}_locked.json', chosen)
        # Threshold and model locked without looking at this outer fold's labels.
        probability = np.mean([fit_predict(X, y, train, valid, spec, jobs)[1]
                               for spec in chosen['specs']], axis=0)
        row = dict(fold=fold, candidate=chosen['candidate'], threshold=chosen['threshold'],
                   **metrics(y[valid], probability, chosen['threshold']))
        rows.append(row)
        predictions.append(pd.DataFrame(dict(source_row=valid, fold=fold, label=y[valid],
                            probability_non_toxic=probability,
                            prediction=(probability >= chosen['threshold']).astype(int))))
        pd.DataFrame(rows).to_csv(out/'outer_metrics.csv', index=False)
        pd.concat(predictions).sort_values('source_row').to_csv(out/'outer_predictions.csv', index=False)
        write_json(out/'split_audit.json', audit)
        print(f'Outer {fold+1} result: {row}', flush=True)
    frame = pd.DataFrame(rows)
    oof = pd.concat(predictions).sort_values('source_row')
    assert oof.source_row.tolist() == ids.tolist()
    summary = dict(recall_target=target, f1_target=.85, trials_per_outer_fold=budget,
                   mean_f1_non_toxic=float(frame.f1_non_toxic.mean()),
                   std_f1_non_toxic=float(frame.f1_non_toxic.std(ddof=0)),
                   mean_recall_toxic=float(frame.recall_toxic.mean()),
                   min_recall_toxic=float(frame.recall_toxic.min()),
                   pooled_f1_non_toxic=float(f1_score(y, oof.prediction)),
                   pooled_recall_toxic=float(recall_score(y, oof.prediction, pos_label=0)),
                   mean_targets_met=bool(frame.f1_non_toxic.mean() >= .85 and frame.recall_toxic.mean() >= target),
                   folds_meeting_both_targets=int(((frame.f1_non_toxic >= .85) & (frame.recall_toxic >= target)).sum()),
                   limitations=['internal development evaluation; prior full-data exploration remains',
                                'outer recall constraint is not guaranteed',
                                'evaluation of search procedure; folds may select different models',
                                'no full-data deployment model refit or independent test score'])
    write_json(out/'evaluation.json', summary)
    return summary


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=20, help='Total completed/pruned trials PER outer fold')
    parser.add_argument('--recall-target', type=float, default=.87)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT/'Data/training/optuna_recall87_v1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.trials < 4 or args.jobs < 1 or not 0 < args.recall_target <= 1:
        parser.error('Require trials >= 4, jobs >= 1, 0 < recall-target <= 1')
    source = ROOT/'Data/input_data/train.csv'
    code = b''.join((ROOT/f'analysis/{name}.py').read_bytes() for name in
                    ['optuna_recall', 'search_recall', 'train_safe'])
    config = dict(recall_target=args.recall_target, f1_target=.85, seed=SEED, inner_folds=5, outer_folds=5,
                  jobs=args.jobs, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(code).hexdigest(), threshold_selection='exact observed score boundaries',
                  sampler='TPE, 8 startup trials, 4 seeded model families',
                  pruning='MedianPruner after 8 startup trials, 3 inner folds minimum',
                  versions={k: importlib.metadata.version(k) for k in
                            ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'rdkit', 'optuna']})
    if args.dry_run:
        print(json.dumps(dict(config, trials_per_outer_fold=args.trials,
                              max_model_fits=25*args.trials+15), indent=2)); return
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise FileExistsError('Use a new output directory or --resume')
        if json.loads((out/'run_config.json').read_text()) != config:
            raise ValueError('Source, code, environment or evaluation configuration changed')
        if (out/'evaluation.json').exists():
            previous = json.loads((out/'evaluation.json').read_text())
            if previous['trials_per_outer_fold'] >= args.trials:
                print(json.dumps(previous, indent=2)); return
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'run_config.json', config)
    write_json(out/'requested_budget.json', dict(trials_per_outer_fold=args.trials, max_model_fits=25*args.trials+15))
    # Extending a completed budget invalidates its final summary until all folds finish.
    if (out/'evaluation.json').exists():
        (out/'evaluation.json').replace(out/'previous_evaluation.json')
    raw = pd.read_csv(source)
    if not raw.label.isin([0, 1]).all():
        raise ValueError('Expected 0=toxic, 1=non-toxic labels')
    X, groups, canonical = make_features(raw)
    y = raw.label.to_numpy(dtype=int)
    if pd.DataFrame(dict(canonical=canonical, label=y)).groupby('canonical').label.nunique().gt(1).any():
        raise ValueError('Conflicting structure labels')
    result = run_nested(X, y, groups, out, args.recall_target, args.trials, args.jobs)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config['source_sha256']
    comparisons = []
    for name, directory in [('f1_baseline', ROOT/'Data/training/safe_ensemble_v1_5fold'),
                            ('recall90_baseline', ROOT/'Data/training/recall90_baseline_5fold'),
                            ('optuna_recall87', out)]:
        if not (directory/'outer_metrics.csv').exists():
            continue
        old_config = json.loads((directory/'run_config.json').read_text())
        if old_config['source_sha256'] != config['source_sha256']:
            continue
        frame = pd.read_csv(directory/'outer_metrics.csv')
        comparisons.append(dict(experiment=name, mean_f1_non_toxic=frame.f1_non_toxic.mean(),
                                mean_recall_toxic=frame.recall_toxic.mean(), min_recall_toxic=frame.recall_toxic.min()))
    pd.DataFrame(comparisons).to_csv(out/'baseline_comparison.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
