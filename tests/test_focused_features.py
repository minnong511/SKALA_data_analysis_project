"""Validate feature-ranking isolation, controlled comparison and cached resume."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from analysis.focused_features import prepare, run_focused


class FocusedTests(unittest.TestCase):
    def data(self):
        a,y=make_classification(n_samples=150,n_features=12,n_informative=6,random_state=43)
        X=pd.DataFrame(a,columns=[f'ecfp_{i}' for i in range(6)]+[f'fcfp_{i}' for i in range(5)]+['MolWt'])
        X['constant']=1.;X['duplicate']=X.ecfp_0
        return X,y,np.repeat(np.arange(50),3)

    def templates(self):
        specs=[dict(id='lgb',family='lightgbm',features='all',top_k=0,toxic_weight=1.5,
                    params=dict(n_estimators=3,num_leaves=7,min_child_samples=5,learning_rate=.1)),
               dict(id='xgb',family='xgboost',features='all',top_k=0,toxic_weight=1.5,
                    params=dict(n_estimators=3,max_depth=2,learning_rate=.1))]
        return {str(f):copy.deepcopy(specs) for f in range(5)}

    def test_rankings_ignore_validation_labels_and_values(self):
        X,y,_=self.data();train=np.arange(120);valid=np.arange(120,150)
        spec=self.templates()['0'][0]
        first=prepare(X,y,train,valid,spec,1)
        other=X.copy();other.iloc[valid]=1000
        labels=y.copy();labels[valid]=1-labels[valid]
        second=prepare(other,labels,train,valid,spec,1)
        self.assertEqual(first[5],second[5])
        self.assertNotIn('constant',first[5]);self.assertNotIn('duplicate',first[5])
        for method in ('gain','anova'): np.testing.assert_array_equal(first[4][method],second[4][method])

    def test_nested_control_and_resume(self):
        X,y,groups=self.data()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            first=run_focused(X,y,groups,self.templates(),out,.87,1,counts=(3,6))
            second=run_focused(X,y,groups,self.templates(),out,.87,1,counts=(3,6))
            self.assertEqual(first,second)
            self.assertEqual(len(pd.read_csv(out/'outer_predictions.csv')),len(y))
            audit=json.loads((out/'split_audit.json').read_text());self.assertEqual(len(audit),30)
            for f in range(5):
                chosen=json.loads((out/f'outer_{f}_locked.json').read_text())
                self.assertGreaterEqual(chosen['selected']['recall_toxic'],.87)
                self.assertGreaterEqual(chosen['selected']['f1_non_toxic'],chosen['control']['f1_non_toxic'])
                self.assertTrue(all(m['method']=='none' for m in chosen['control']['members']))
                outer=next(e for e in audit if e['scope']=='outer' and e['fold']==f)
                for e in audit:
                    if e['scope']==f'outer_{f}_inner':
                        self.assertFalse(set(e['train_ids']+e['valid_ids']) & set(outer['valid_ids']))
                        self.assertFalse(set(groups[e['train_ids']]) & set(groups[e['valid_ids']]))
            for path in out.glob('*_features.json'):
                selected=json.loads(path.read_text())['selected_features']
                for name,columns in selected.items():
                    if not name.startswith('none'):self.assertLessEqual(len(columns),int(name.split('_')[-1]))


if __name__=='__main__': unittest.main()
