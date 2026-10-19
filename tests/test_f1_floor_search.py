import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution
from optuna.trial import create_trial
from sklearn.datasets import make_classification

from analysis.f1_floor_search import run, select_threshold
from analysis.optuna_recall import save_probability
from analysis.search_recall import fit_predict
from analysis.train_safe import SEED, checked_splits, metrics


class FloorTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_search(self):
        rng=np.random.default_rng(43);y=np.array([0,1]*50)
        for p in [rng.random(100),np.round(rng.random(100),1),np.ones(100),.2+.6*y]:
            for floor in [.65,.82,1.]:
                candidates=[metrics(y,p,t) for t in np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]]
                eligible=[r for r in candidates if r['f1_non_toxic']>=floor]
                chosen=select_threshold(y,p,floor)
                self.assertEqual(chosen['f1_floor_met'],bool(eligible))
                if eligible:
                    self.assertAlmostEqual(chosen['recall_toxic'],max(r['recall_toxic'] for r in eligible))
                    self.assertGreaterEqual(chosen['objective'],0)
                else:
                    self.assertAlmostEqual(chosen['f1_non_toxic'],max(r['f1_non_toxic'] for r in candidates))
                    self.assertLess(chosen['objective'],0)

    def test_cached_nested_rescore_and_resume(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        data,y=make_classification(n_samples=150,n_features=12,n_informative=6,random_state=43)
        X=pd.DataFrame(data,columns=[f'ecfp_{i}' for i in range(12)])
        groups=np.repeat(np.arange(50),3);audit=[];ids=np.arange(len(y))
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'old';out=Path(folder)/'new';source.mkdir();out.mkdir()
            for f,(train,valid) in enumerate(checked_splits(ids,y,groups,SEED,'outer',audit)):
                scope=f'outer_{f}_inner';study=optuna.create_study(study_name=scope,storage=f'sqlite:///{source/"studies.sqlite3"}',direction='maximize')
                splits=list(checked_splits(train,y,groups,SEED+f+1,scope,audit))
                for k,family in enumerate(['lightgbm','xgboost']):
                    spec=dict(id=f'trial_{k:04d}',family=family,features='all',top_k=0,toxic_weight=1.,params=dict(n_estimators=3,max_depth=2))
                    p=np.full(len(y),np.nan)
                    for a,b in splits:_,p[b]=fit_predict(X,y,a,b,spec,1)
                    save_probability(source/f'{scope}_trial_{k:04d}.npz',p)
                    study.add_trial(create_trial(params={'family':family},distributions={'family':CategoricalDistribution(['lightgbm','xgboost','extra_trees','logistic'])},value=.5,user_attrs={'spec':spec}))
            first=run(X,y,groups,source,out,.82,0,1,audit)
            again=run(X,y,groups,source,out,.82,0,1,audit)
            self.assertEqual(first,again)
            for f in range(5):
                study=optuna.load_study(study_name=f'outer_{f}_inner',storage=f'sqlite:///{out/"studies.sqlite3"}')
                self.assertEqual(len(study.trials),2)
                chosen=json.loads((out/f'outer_{f}_locked.json').read_text())
                self.assertEqual(chosen['f1_floor_met'],chosen['f1_non_toxic']>=.82)
            self.assertEqual(len(pd.read_csv(out/'outer_predictions.csv')),len(y))


if __name__=='__main__':unittest.main()
