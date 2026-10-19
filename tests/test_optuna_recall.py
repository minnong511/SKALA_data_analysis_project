"""Exact-threshold, nested leakage-boundary and SQLite-resume checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.datasets import make_classification

from analysis.optuna_recall import optimal_threshold, run_nested, suggest_spec
from analysis.train_safe import metrics


def tiny_spec(trial):
    spec = suggest_spec(trial)
    spec['top_k'] = 0
    if 'n_estimators' in spec['params']:
        spec['params']['n_estimators'] = 3
    if spec['family'] == 'logistic':
        spec['params']['C'] = .1
    return spec


class OptunaTests(unittest.TestCase):
    def test_exact_constraint_and_ties(self):
        rng = np.random.default_rng(43)
        y = np.array([0, 1] * 50)
        for probability in [rng.random(100), np.round(rng.random(100), 1), np.ones(100), np.zeros(100)]:
            for target in [.87, 1.]:
                chosen = optimal_threshold(y, probability, target)
                boundaries = np.r_[np.unique(probability), np.nextafter(probability.max(), np.inf)]
                eligible = [metrics(y, probability, t) for t in boundaries]
                best = max(r['f1_non_toxic'] for r in eligible if r['recall_toxic'] >= target)
                self.assertGreaterEqual(chosen['recall_toxic'], target)
                self.assertAlmostEqual(chosen['f1_non_toxic'], best)

    def test_sqlite_resume_budget_extension_and_outer_isolation(self):
        data, y = make_classification(n_samples=150, n_features=12, n_informative=6, random_state=43)
        X = pd.DataFrame(data, columns=[f'ecfp_{i}' for i in range(6)] +
                         [f'fcfp_{i}' for i in range(5)] + ['MolWt'])
        groups = np.repeat(np.arange(50), 3)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            first = run_nested(X, y, groups, out, .87, 4, 1, tiny_spec)
            again = run_nested(X, y, groups, out, .87, 4, 1, tiny_spec)
            self.assertEqual(first, again)
            extended = run_nested(X, y, groups, out, .87, 5, 1, tiny_spec)
            self.assertEqual(extended['trials_per_outer_fold'], 5)
            audit = json.loads((out/'split_audit.json').read_text())
            self.assertEqual(len(audit), 30)
            for fold in range(5):
                outer = next(e for e in audit if e['scope'] == 'outer' and e['fold'] == fold)
                for entry in audit:
                    if entry['scope'] == f'outer_{fold}_inner':
                        self.assertFalse(set(entry['train_ids'] + entry['valid_ids']) & set(outer['valid_ids']))
                        self.assertFalse(set(groups[entry['train_ids']]) & set(groups[entry['valid_ids']]))
                chosen = json.loads((out/f'outer_{fold}_locked.json').read_text())
                self.assertGreaterEqual(chosen['recall_toxic'], .87)
                study = optuna.load_study(study_name=f'outer_{fold}_inner', storage=f'sqlite:///{out/"studies.sqlite3"}')
                self.assertEqual(len(study.trials), 5)
                self.assertEqual({t.user_attrs['spec']['family'] for t in study.trials[:4]},
                                 {'lightgbm', 'xgboost', 'extra_trees', 'logistic'})
            self.assertEqual(len(pd.read_csv(out/'outer_predictions.csv')), len(y))


if __name__ == '__main__':
    unittest.main()
