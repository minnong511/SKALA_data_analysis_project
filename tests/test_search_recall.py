"""Synthetic end-to-end checks; does not claim performance on molecular data."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from analysis.search_recall import candidates, feature_columns, run_search


class SearchTests(unittest.TestCase):
    def test_nested_search_and_cached_resume(self):
        data, y = make_classification(n_samples=200, n_features=24, n_informative=8,
                                      random_state=43)
        X = pd.DataFrame(data, columns=[f'ecfp_{i}' for i in range(8)] +
                         [f'fcfp_{i}' for i in range(8)] +
                         [f'ptfp_{i}' for i in range(7)] + ['MolWt'])
        X['duplicate'] = X.ecfp_0
        X['constant'] = 1.
        X.loc[0, 'MolWt'] = np.nan
        groups = np.repeat(np.arange(50), 4)
        specs = candidates(4)
        self.assertEqual(specs, candidates(4))
        for spec in specs:
            spec['top_k'] = 512  # deliberately exceeds available cleaned features
            spec['features'] = 'ecfp'
            if 'n_estimators' in spec['params']:
                spec['params']['n_estimators'] = 5
        self.assertNotIn('fcfp_0', feature_columns(X, specs[0]))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            summary = run_search(X, y, groups, specs, .9, out, jobs=1)
            self.assertTrue((out/'evaluation.json').is_file())
            self.assertEqual(len(pd.read_csv(out/'outer_predictions.csv')), len(y))
            audit = json.loads((out/'split_audit.json').read_text())
            self.assertEqual(len(audit), 30)
            for entry in audit:
                self.assertFalse(set(groups[entry['train_ids']]) & set(groups[entry['valid_ids']]))
            for fold in range(5):
                chosen = json.loads((out/f'outer_{fold}_locked.json').read_text())
                self.assertGreaterEqual(chosen['recall_toxic'], .9)
                board = pd.read_csv(out/f'outer_{fold}_inner_leaderboard.csv')
                self.assertAlmostEqual(chosen['f1_non_toxic'], board.f1_non_toxic.max())
            cached = run_search(X, y, groups, specs, .9, out, jobs=1)
            self.assertEqual(summary, cached)


if __name__ == '__main__':
    unittest.main()
