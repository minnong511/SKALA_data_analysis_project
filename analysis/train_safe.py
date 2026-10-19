"""Nested scaffold evaluation of the historical winning ensemble family.
Run from the project root: python -m analysis.train_safe
No external labels, historical predictions, or historical fitted models are loaded.
"""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors, ChemicalFeatures
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SEED = 43
N_SPLITS = 5
MEMBERS = ['lightgbm', 'lightgbm_balanced', 'xgboost']
DESCRIPTORS = ['MolMR', 'LabuteASA', 'TPSA', 'NumHDonors', 'NumHAcceptors',
               'NumAromaticRings', 'RingCount', 'NumAromaticHeterocycles',
               'NumAliphaticHeterocycles', 'NumRotatableBonds', 'FractionCSP3']
THRESHOLDS = np.round(np.arange(.20, .801, .01), 2)


class TrainOnlyCleaner(TransformerMixin, BaseEstimator):
    """Learn constant/duplicate exclusions and medians only from fit rows."""
    def fit(self, X, y=None):
        frame = X.replace([np.inf, -np.inf], np.nan)
        eligible = frame.loc[:, frame.nunique(dropna=True).gt(1)]
        seen, self.columns_ = set(), []
        for column in eligible:
            values = eligible[column].to_numpy(dtype=np.float64, copy=True)
            values[values == 0] = 0.
            values[np.isnan(values)] = np.nan
            key = hashlib.sha256(values.tobytes()).digest()
            if key not in seen:
                seen.add(key)
                self.columns_.append(column)
        if not self.columns_:
            raise ValueError('No usable training features')
        self.medians_ = frame[self.columns_].median()
        self.fit_rows_ = X.index.to_numpy().copy()
        return self

    def transform(self, X):
        return (X[self.columns_].replace([np.inf, -np.inf], np.nan)
                .fillna(self.medians_).to_numpy(dtype=np.float32))


def make_features(raw):
    """Deterministic per-molecule features; no fitted dataset statistics or labels."""
    fps = [f'{family}_{i}' for family in ['ecfp', 'fcfp', 'ptfp'] for i in range(1024)]
    required = ['SMILES', 'clogp', 'MolWt'] + fps
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)[:10]}')
    if not raw[fps].isin([0, 1]).all().all():
        raise ValueError('Fingerprint must be binary and nonmissing')
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'))
    records, groups, canonical = [], [], []
    for index, value in raw.SMILES.items():
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
        if mol is None:
            raise ValueError(f'Invalid SMILES at source row {index}')
        canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        # All acyclic molecules share the empty scaffold: conservative grouping.
        groups.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        record = {'rdkit_'+name: float(getattr(Descriptors, name)(mol)) for name in DESCRIPTORS}
        charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
        families = [f.GetFamily() for f in factory.GetFeaturesForMol(mol)]
        record.update({'herg_FormalCharge': sum(charges),
                       'herg_AbsFormalCharge': sum(map(abs, charges)),
                       'herg_PositiveAtomCount': sum(c > 0 for c in charges),
                       'herg_NegativeAtomCount': sum(c < 0 for c in charges),
                       'herg_PosIonizablePatternCount': families.count('PosIonizable'),
                       'herg_NegIonizablePatternCount': families.count('NegIonizable')})
        records.append(record)
    # Historical winning input excluded sa_score and qed. This is a replacement
    # configuration, not a test of the pure addition effect of domain features.
    X = pd.concat([raw[fps + ['clogp', 'MolWt']], pd.DataFrame(records, index=raw.index)], axis=1)
    assert X.columns.is_unique and X.shape[1] == 3091
    assert not {'label', 'toxic', 'SMILES'} & set(X.columns)
    return X, np.asarray(groups), np.asarray(canonical)


def new_model(name):
    if name == 'xgboost':
        estimator = XGBClassifier(n_estimators=300, learning_rate=.05, max_depth=6,
                                  tree_method='hist', eval_metric='logloss', random_state=SEED, n_jobs=4)
    else:
        estimator = LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31,
            min_child_samples=20, random_state=SEED, n_jobs=4, verbosity=-1,
            deterministic=True, force_col_wise=True,
            class_weight='balanced' if name == 'lightgbm_balanced' else None)
    return Pipeline([('clean', TrainOnlyCleaner()), ('model', estimator)])


def metrics(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    recall = recall_score(y, pred, pos_label=0, zero_division=0)
    return dict(f1_non_toxic=float(f1_score(y, pred, zero_division=0)),
                precision_non_toxic=float(precision_score(y, pred, zero_division=0)),
                recall_toxic=float(recall), miss_rate_toxic=float(1-recall),
                dummy_f1=float(f1_score(y, np.ones(len(y)), zero_division=0)))


def checked_splits(ids, y, groups, seed, scope, audit):
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (a, b) in enumerate(splitter.split(ids, y[ids], groups[ids])):
        train, valid = ids[a], ids[b]
        assert not set(train) & set(valid)
        assert not set(groups[train]) & set(groups[valid])
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise ValueError('Each fold must contain both classes')
        audit.append(dict(scope=scope, fold=fold, train_ids=train.tolist(), valid_ids=valid.tolist(),
                          row_overlap=0, scaffold_overlap=0))
        yield train, valid


def fit_members(X, y, train, valid=None):
    models, predictions = [], []
    for name in MEMBERS:
        model = new_model(name).fit(X.iloc[train], y[train])
        assert set(model.named_steps['clean'].fit_rows_) == set(train)
        if valid is not None:
            assert not set(model.named_steps['clean'].fit_rows_) & set(valid)
            predictions.append(model.predict_proba(X.iloc[valid])[:, 1])
        models.append(model)
    return models, np.column_stack(predictions) if valid is not None else None


def choose_threshold(y, probability, recall_target=None):
    # Include an all-toxic endpoint so the recall constraint is always feasible.
    thresholds = (np.r_[np.round(np.arange(0., 1.001, .01), 2), np.nextafter(1., 2.)]
                  if recall_target is not None else THRESHOLDS)
    candidates = [dict(threshold=float(t), **metrics(y, probability, t)) for t in thresholds]
    eligible = [r for r in candidates if recall_target is None or r['recall_toxic'] >= recall_target]
    chosen = min(eligible, key=lambda r: (-r['f1_non_toxic'], abs(r['threshold']-.5), r['threshold']))
    return chosen, candidates


def select_inner(X, y, groups, ids, seed, scope, audit, recall_target=None):
    inner = np.full((len(X), len(MEMBERS)), np.nan)
    coverage = np.zeros(len(X), dtype=int)
    for train, valid in checked_splits(ids, y, groups, seed, scope, audit):
        _, inner[valid] = fit_members(X, y, train, valid)
        coverage[valid] += 1
    assert (coverage[ids] == 1).all()
    assert np.isfinite(inner[ids]).all()
    # Historical weighting rule, estimated anew using inner OOF only.
    weights = np.array([f1_score(y[ids], inner[ids, j] >= .5, zero_division=0)
                        for j in range(len(MEMBERS))])
    weights = weights / weights.sum() if weights.sum() else np.ones(len(MEMBERS))/len(MEMBERS)
    prob = np.sum(inner[ids] * weights, axis=1)
    chosen, candidates = choose_threshold(y[ids], prob, recall_target)
    return weights, chosen['threshold'], candidates


def predict(raw, artifact):
    """Predict original label convention with the saved training transformations."""
    X, _, _ = make_features(raw)
    X = X[artifact['feature_columns']]
    probability = sum(weight * model.predict_proba(X)[:, 1]
                      for weight, model in zip(artifact['weights'], artifact['models']))
    return pd.DataFrame({'probability_non_toxic': probability,
                         'label': (probability >= artifact['threshold']).astype(int)}, index=raw.index)


def run(output=None, recall_target=None):
    if recall_target is not None and not 0 <= recall_target <= 1:
        raise ValueError('recall_target must be between 0 and 1')
    source = ROOT / 'Data/input_data/train.csv'
    default_name = 'recall90_baseline_5fold' if recall_target is not None else 'safe_ensemble_v1_5fold'
    out = Path(output) if output else ROOT / 'Data/training' / default_name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new output directory to preserve the locked run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = dict(source_sha256=digest, seed=SEED, members=MEMBERS, outer_folds=N_SPLITS, inner_folds=N_SPLITS,
        grouping='nonchiral Murcko scaffold; all acyclic molecules grouped together',
        input='historical fingerprint + 19 domain features (3091); no supervised feature selection',
        trees=300, learning_rate=.05,
        threshold_grid=(np.r_[np.round(np.arange(0., 1.001, .01), 2), np.nextafter(1., 2.)].tolist()
                        if recall_target is not None else THRESHOLDS.tolist()),
        recall_target=recall_target, f1_target=.85 if recall_target is not None else None,
        selection='maximize inner OOF non-toxic F1 subject to toxic recall target'
                  if recall_target is not None else 'maximize inner OOF non-toxic F1',
        weighting='member inner OOF F1 at 0.5, normalized; fixed member family',
        interpretation='internal nested evaluation; prior whole-data exploration and model-family selection remain',
        versions={k: importlib.metadata.version(k) for k in ['numpy','pandas','rdkit','scikit-learn','lightgbm','xgboost']},
        python=platform.python_version())
    (out/'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    raw = pd.read_csv(source)
    if not raw.label.isin([0, 1]).all():
        raise ValueError('Labels must be 0=toxic, 1=non-toxic')
    y = raw.label.to_numpy(dtype=int)
    X, groups, canonical = make_features(raw)
    conflicts = pd.DataFrame({'canonical':canonical, 'label':y}).groupby('canonical').label.nunique()
    if conflicts.gt(1).any():
        raise ValueError('Conflicting labels for identical isomeric structures require review')
    ids = np.arange(len(raw)); audit = []; rows = []; outer_oof = np.full(len(raw),np.nan)
    outer_pred = np.full(len(raw),-1); coverage = np.zeros(len(raw),dtype=int); fold_ids = np.full(len(raw),-1)
    for fold,(train,valid) in enumerate(checked_splits(ids,y,groups,SEED,'outer',audit)):
        print(f'Outer {fold+1}/{N_SPLITS}: inner selection on {len(train)} rows', flush=True)
        weights, threshold, search = select_inner(X,y,groups,train,SEED+fold+1,f'outer_{fold}_inner',audit,recall_target)
        for entry in audit:
            if entry['scope']==f'outer_{fold}_inner':
                assert not set(entry['train_ids']+entry['valid_ids']) & set(valid)
        (out/f'outer_{fold}_locked.json').write_text(json.dumps(dict(weights=weights.tolist(),threshold=threshold),indent=2))
        _, prob = fit_members(X,y,train,valid)
        outer_oof[valid] = np.sum(prob * weights, axis=1); outer_pred[valid] = (outer_oof[valid]>=threshold).astype(int)
        coverage[valid] += 1; fold_ids[valid] = fold
        rows.append(dict(fold=fold,rows=len(valid),threshold=threshold,**metrics(y[valid],outer_oof[valid],threshold)))
        pd.DataFrame(search).to_csv(out/f'outer_{fold}_inner_selection.csv',index=False)
    assert (coverage==1).all() and np.isfinite(outer_oof).all()
    # These predictions use only their own outer fold's inner-selected threshold.
    pooled = dict(f1_non_toxic=float(f1_score(y,outer_pred)),
                  recall_toxic=float(recall_score(y,outer_pred,pos_label=0)),
                  confusion_matrix=confusion_matrix(y,outer_pred,labels=[0,1]).tolist())
    pd.DataFrame(rows).to_csv(out/'outer_metrics.csv',index=False)
    pd.DataFrame(dict(source_row=ids,scaffold=groups,fold=fold_ids,label=y,
                     probability_non_toxic=outer_oof,prediction=outer_pred)).to_csv(out/'outer_predictions.csv',index=False)
    # Deployment selection is rerun on all data, not copied from the best outer fold.
    print('Final inner selection and full-data deployment refit',flush=True)
    weights,threshold,search = select_inner(X,y,groups,ids,SEED+100,'final_inner',audit,recall_target)
    pd.DataFrame(search).to_csv(out/'final_inner_selection.csv',index=False)
    models,_ = fit_members(X,y,ids)
    artifact = dict(models=models,weights=weights,threshold=threshold,feature_columns=X.columns.tolist(),
                    source_sha256=digest,label_definition={0:'toxic',1:'non-toxic'},config=config)
    joblib.dump(artifact,out/'model.joblib')
    loaded=joblib.load(out/'model.joblib')
    for original,restored in zip(models,loaded['models']):
        np.testing.assert_allclose(original.predict_proba(X.iloc[:10]),restored.predict_proba(X.iloc[:10]))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    (out/'split_audit.json').write_text(json.dumps(audit))
    fold_frame = pd.DataFrame(rows)
    summary = dict(mean_f1_non_toxic=float(fold_frame.f1_non_toxic.mean()),
                   std_f1_non_toxic=float(fold_frame.f1_non_toxic.std(ddof=0)),
                   mean_recall_toxic=float(fold_frame.recall_toxic.mean()),
                   min_recall_toxic=float(fold_frame.recall_toxic.min()))
    if recall_target is not None:
        summary['mean_targets_met'] = bool(summary['mean_f1_non_toxic'] >= .85 and
                                           summary['mean_recall_toxic'] >= recall_target)
        summary['folds_meeting_both_targets'] = int(((fold_frame.f1_non_toxic >= .85) &
                                                     (fold_frame.recall_toxic >= recall_target)).sum())
    result=dict(fold_summary=summary,pooled_outer=pooled,final_threshold=threshold,final_weights=weights.tolist(),
                audited_splits=len(audit),source_unchanged=True,
                limitations=['prior EDA and model-family selection used the full dataset',
                             'provided feature generation provenance and assay/batch leakage unverified',
                             'scaffold grouping does not resolve every tautomer, salt or similarity relationship',
                             'final full-data model has no independent measured score'])
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__ == '__main__':
    # Pickles must reference the importable module, not a transient __main__ class.
    from analysis.train_safe import run as run_training
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recall-target', type=float, default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    run_training(output=args.output, recall_target=args.recall_target)
