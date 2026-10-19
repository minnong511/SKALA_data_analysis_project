"""Budgeted, nested scaffold model search. Run --help for options."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from analysis.train_safe import (ROOT, SEED, TrainOnlyCleaner, checked_splits,
                                 choose_threshold, make_features, metrics)


class CappedSelectKBest(SelectKBest):
    """Keep search valid when training-only cleaning leaves fewer than k columns."""
    def fit(self, X, y=None):
        requested = self.k
        self.k = min(requested, X.shape[1])
        try:
            return super().fit(X, y)
        finally:
            # get_support uses k, so retain the effective training-fold value.
            self.requested_k_ = requested


def candidates(count, seed=SEED):
    """Predeclare reproducible candidates; outer scores never influence search."""
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        family = ['lightgbm', 'xgboost', 'extra_trees', 'logistic'][i % 4]
        weight = float(rng.choice([1., 1.5, 2., 3., 5.]))
        common = dict(id=f'trial_{i:03d}', family=family, toxic_weight=weight,
                      features=str(rng.choice(['all', 'ecfp', 'fcfp', 'ptfp', 'ecfp_fcfp'])),
                      top_k=int(rng.choice([0, 256, 512, 1024])))
        if family == 'lightgbm':
            params = dict(n_estimators=int(rng.choice([300, 600, 900])),
                          learning_rate=float(rng.choice([.02, .05, .1])),
                          num_leaves=int(rng.choice([15, 31, 63])),
                          min_child_samples=int(rng.choice([20, 50, 100])),
                          colsample_bytree=float(rng.choice([.6, .8, 1.])),
                          reg_lambda=float(rng.choice([0., 1., 10.])))
        elif family == 'xgboost':
            params = dict(n_estimators=int(rng.choice([300, 600, 900])),
                          learning_rate=float(rng.choice([.02, .05, .1])),
                          max_depth=int(rng.choice([3, 5, 7])),
                          min_child_weight=float(rng.choice([1., 5., 15.])),
                          subsample=float(rng.choice([.7, 1.])),
                          colsample_bytree=float(rng.choice([.6, .8, 1.])),
                          reg_lambda=float(rng.choice([1., 10., 30.])))
        elif family == 'extra_trees':
            params = dict(n_estimators=400, min_samples_leaf=int(rng.choice([1, 3, 8])),
                          max_features=rng.choice(['sqrt', 'log2', '.5']).item())
            if params['max_features'] == '.5':
                params['max_features'] = .5
        else:
            params = dict(C=float(rng.choice([.01, .1, 1., 10.])), max_iter=2000)
        common['params'] = params
        result.append(common)
    # A fixed single-model anchor, included independent of the random sample.
    if result:
        result[0] = dict(id='trial_000', family='lightgbm', toxic_weight=1.,
                         features='all', top_k=0,
                         params=dict(n_estimators=300, learning_rate=.05,
                                     num_leaves=31, min_child_samples=20))
    return result


def feature_columns(X, spec):
    families = spec['features'].split('_')
    return [c for c in X if spec['features'] == 'all'
            or not c.startswith(('ecfp_', 'fcfp_', 'ptfp_'))
            or any(c.startswith(f + '_') for f in families)]


def build_model(spec, jobs=4):
    kw = dict(spec['params'], random_state=SEED)
    family = spec['family']
    if family == 'lightgbm':
        model = LGBMClassifier(**kw, n_jobs=jobs, verbosity=-1,
                               deterministic=True, force_col_wise=True)
    elif family == 'xgboost':
        model = XGBClassifier(**kw, n_jobs=jobs, tree_method='hist', eval_metric='logloss')
    elif family == 'extra_trees':
        model = ExtraTreesClassifier(**kw, n_jobs=jobs)
    else:
        model = LogisticRegression(**kw, solver='lbfgs')
    steps = [('clean', TrainOnlyCleaner())]
    if spec['top_k']:
        steps.append(('select', CappedSelectKBest(f_classif, k=spec['top_k'])))
    if family == 'logistic':
        steps.append(('scale', StandardScaler()))
    return Pipeline(steps + [('model', model)])


def fit_predict(X, y, train, valid, spec, jobs):
    columns = feature_columns(X, spec)
    model = build_model(spec, jobs)
    sample_weight = np.where(y[train] == 0, spec['toxic_weight'], 1.)
    model.fit(X.iloc[train][columns], y[train], model__sample_weight=sample_weight)
    assert set(model['clean'].fit_rows_) == set(train)
    assert not set(train) & set(valid)
    return model, model.predict_proba(X.iloc[valid][columns])[:, 1]


def inner_search(X, y, groups, ids, specs, target, seed, scope, out, audit, jobs):
    splits = list(checked_splits(ids, y, groups, seed, scope, audit))
    scores, predictions = [], {}
    for spec in specs:
        cache = out / f'{scope}_{spec["id"]}.npz'
        if cache.exists():
            with np.load(cache) as saved:
                probability = saved['probability']
        else:
            probability = np.full(len(y), np.nan)
            for train, valid in splits:
                _, probability[valid] = fit_predict(X, y, train, valid, spec, jobs)
            temporary = cache.with_suffix('.pending.npz')
            np.savez_compressed(temporary, probability=probability)
            temporary.replace(cache)
        assert np.isfinite(probability[ids]).all()
        chosen, curve = choose_threshold(y[ids], probability[ids], target)
        pd.DataFrame(curve).to_csv(out / f'{scope}_{spec["id"]}_thresholds.csv', index=False)
        predictions[spec['id']] = probability
        scores.append(dict(candidate=spec['id'], members=[spec['id']], **chosen))
        print(f'{scope} {spec["id"]} {spec["family"]}: F1={chosen["f1_non_toxic"]:.4f}, '
              f'Recall={chosen["recall_toxic"]:.4f}', flush=True)
    # Blend diverse families selected using only inner OOF results.
    ranked = sorted(scores, key=lambda r: (-r['f1_non_toxic'], r['candidate']))
    members, families = [], set()
    by_id = {s['id']: s for s in specs}
    for row in ranked:
        family = by_id[row['candidate']]['family']
        if family not in families:
            members.append(row['candidate']); families.add(family)
        if len(members) == 3:
            break
    if len(members) > 1:
        blend = np.mean([predictions[m] for m in members], axis=0)
        chosen, curve = choose_threshold(y[ids], blend[ids], target)
        scores.append(dict(candidate='diverse_blend', members=members, **chosen))
        pd.DataFrame(curve).to_csv(out / f'{scope}_blend_thresholds.csv', index=False)
    frame = pd.DataFrame(scores).sort_values(['f1_non_toxic', 'candidate'], ascending=[False, True])
    frame.to_csv(out / f'{scope}_leaderboard.csv', index=False)
    # Model and threshold are locked before any outer predictions are evaluated.
    return sorted(scores, key=lambda r: (-r['f1_non_toxic'], r['candidate']))[0]


def run_search(X, y, groups, specs, target, out, jobs=4):
    out = Path(out)
    ids = np.arange(len(y)); audit = []; rows = []; predictions = []
    by_id = {s['id']: s for s in specs}
    for fold, (train, valid) in enumerate(checked_splits(ids, y, groups, SEED, 'outer', audit)):
        scope = f'outer_{fold}_inner'
        print(f'Outer {fold+1}/5', flush=True)
        chosen = inner_search(X, y, groups, train, specs, target, SEED+fold+1,
                              scope, out, audit, jobs)
        assert all(not set(e['train_ids'] + e['valid_ids']) & set(valid)
                   for e in audit if e['scope'] == scope)
        (out/f'outer_{fold}_locked.json').write_text(json.dumps(chosen, indent=2))
        probability = np.mean([fit_predict(X, y, train, valid, by_id[m], jobs)[1]
                               for m in chosen['members']], axis=0)
        row = dict(fold=fold, candidate=chosen['candidate'], threshold=chosen['threshold'],
                   **metrics(y[valid], probability, chosen['threshold']))
        rows.append(row)
        predictions.append(pd.DataFrame(dict(source_row=valid, fold=fold, label=y[valid],
                            probability_non_toxic=probability,
                            prediction=(probability >= chosen['threshold']).astype(int))))
        pd.DataFrame(rows).to_csv(out/'outer_metrics.csv', index=False)
        print(f'Outer result: {row}', flush=True)
    frame = pd.DataFrame(rows)
    oof = pd.concat(predictions).sort_values('source_row')
    assert oof.source_row.tolist() == ids.tolist()
    oof.to_csv(out/'outer_predictions.csv', index=False)
    summary = dict(mean_f1_non_toxic=float(frame.f1_non_toxic.mean()),
                   std_f1_non_toxic=float(frame.f1_non_toxic.std(ddof=0)),
                   mean_recall_toxic=float(frame.recall_toxic.mean()),
                   min_recall_toxic=float(frame.recall_toxic.min()),
                   folds_meeting_both_targets=int(((frame.f1_non_toxic >= .85) &
                                                   (frame.recall_toxic >= target)).sum()),
                   mean_targets_met=bool(frame.f1_non_toxic.mean() >= .85 and
                                         frame.recall_toxic.mean() >= target),
                   interpretation='Internal development evaluation of the search procedure; '
                                  'outer folds may select different models; no independent test score.')
    (out/'split_audit.json').write_text(json.dumps(audit))
    (out/'evaluation.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--recall-target', type=float, default=.90)
    parser.add_argument('--output', type=Path, default=ROOT/'Data/training/recall_search_v1')
    parser.add_argument('--candidates', type=Path, help='Optional JSON list of candidate specifications')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.trials < 4 or not 0 < args.recall_target <= 1 or args.jobs < 1:
        parser.error('Require trials >= 4, 0 < recall-target <= 1, jobs >= 1')
    specs = json.loads(args.candidates.read_text()) if args.candidates else candidates(args.trials)
    if not specs or len({s['id'] for s in specs}) != len(specs):
        raise ValueError('Candidate IDs must be unique and list nonempty')
    for spec in specs:
        if (spec['family'] not in {'lightgbm', 'xgboost', 'extra_trees', 'logistic'}
                or spec['features'] not in {'all', 'ecfp', 'fcfp', 'ptfp', 'ecfp_fcfp'}
                or spec['top_k'] < 0 or spec['toxic_weight'] <= 0
                or not spec['id'].replace('_', '').isalnum()):
            raise ValueError(f'Invalid candidate: {spec}')
        build_model(spec, args.jobs)
    fit_budget = len(specs) * 25 + 15
    config = dict(seed=SEED, outer_folds=5, inner_folds=5, recall_target=args.recall_target,
                  f1_target=.85, candidates=specs, jobs=args.jobs,
                  source_sha256=hashlib.sha256((ROOT/'Data/input_data/train.csv').read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes() +
                                            (ROOT/'analysis/train_safe.py').read_bytes()).hexdigest(),
                  versions={k: importlib.metadata.version(k) for k in
                            ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'rdkit']})
    if args.dry_run:
        print(json.dumps(config, indent=2))
        print(f'Max fits: {fit_budget}; no full-data deployment refit included.')
        return
    out = args.output
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise FileExistsError('Use a new output directory or --resume')
        if json.loads((out/'run_config.json').read_text()) != config:
            raise ValueError('Resume configuration, source, code or environment differs')
        if (out/'evaluation.json').exists():
            print((out/'evaluation.json').read_text()); return
    out.mkdir(parents=True, exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(config, indent=2))
    raw = pd.read_csv(ROOT/'Data/input_data/train.csv')
    if not raw.label.isin([0, 1]).all():
        raise ValueError('Expected 0=toxic, 1=non-toxic labels')
    X, groups, canonical = make_features(raw)
    y = raw.label.to_numpy(dtype=int)
    if pd.DataFrame(dict(canonical=canonical, label=y)).groupby('canonical').label.nunique().gt(1).any():
        raise ValueError('Conflicting structure labels')
    summary = run_search(X, y, groups, specs, args.recall_target, out, args.jobs)
    assert hashlib.sha256((ROOT/'Data/input_data/train.csv').read_bytes()).hexdigest() == config['source_sha256']
    comparison = []
    for name, directory in [('f1_baseline', ROOT/'Data/training/safe_ensemble_v1_5fold'),
                            ('recall_baseline', ROOT/'Data/training/recall90_baseline_5fold'),
                            ('model_search', out)]:
        path = directory/'outer_metrics.csv'
        if path.exists():
            if name != 'model_search':
                previous = json.loads((directory/'run_config.json').read_text())
                if previous['source_sha256'] != config['source_sha256']:
                    continue
            frame = pd.read_csv(path)
            comparison.append(dict(experiment=name, mean_f1_non_toxic=frame.f1_non_toxic.mean(),
                                   mean_recall_toxic=frame.recall_toxic.mean(),
                                   min_recall_toxic=frame.recall_toxic.min()))
    pd.DataFrame(comparison).to_csv(out/'baseline_comparison.csv', index=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
