"""Recompute the basic facts used in the EDA report without changing source data."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]

def run():
    RDLogger.DisableLog('rdApp.*')
    path = ROOT / 'Data/input_data/train.csv'
    raw = pd.read_csv(path)
    properties = ['MolWt', 'clogp', 'sa_score', 'qed']
    fp_columns = [c for c in raw if c.startswith(('ecfp_', 'fcfp_', 'ptfp_'))]
    canonical = raw.SMILES.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
                               if Chem.MolFromSmiles(s) is not None else None)
    unique = raw.loc[~canonical.duplicated()].copy()
    unique['toxic'] = 1 - unique.label
    quality = [
        {'항목': '결측 셀', '건수': int(raw.isna().sum().sum()), '단위': '셀'},
        {'항목': '완전히 같은 행의 추가 중복', '건수': int(raw.duplicated().sum()), '단위': '행'},
        {'항목': '같은 SMILES 문자열의 추가 중복', '건수': int(raw.SMILES.duplicated().sum()), '단위': '행'},
        {'항목': '같은 분자 구조의 추가 중복', '건수': int(canonical.duplicated().sum()), '단위': '행'},
        {'항목': 'SMILES 파싱 실패', '건수': int(canonical.isna().sum()), '단위': '행'},
        {'항목': '0, 1 이외 label', '건수': int((~raw.label.isin([0, 1])).sum()), '단위': '셀'},
        {'항목': '0, 1 이외 Fingerprint', '건수': int((~raw[fp_columns].isin([0, 1])).sum().sum()), '단위': '셀'},
        {'항목': '유한수가 아닌 물성값', '건수': int((~np.isfinite(raw[properties].to_numpy())).sum()), '단위': '셀'},
    ]
    classes = [{'label': k, 'class': '독성 있음 (0)' if k == 0 else '독성 없음 (1)',
                'count': int(raw.label.eq(k).sum()), 'percent': float(raw.label.eq(k).mean() * 100)} for k in [0, 1]]
    corr = [{'property': p, 'pearson': float(unique[p].corr(unique.toxic)),
             'spearman': float(unique[p].corr(unique.toxic, method='spearman'))} for p in properties]
    fp_summary = []
    for name in ['ecfp', 'fcfp', 'ptfp']:
        block = raw[[c for c in fp_columns if c.startswith(name + '_')]]
        fp_summary.append({'family': name.upper(), 'columns': block.shape[1],
                           'constant_columns': int(block.nunique().le(1).sum()),
                           'active_percent': float(block.to_numpy().mean() * 100)})
    same_label = pd.DataFrame({'canonical': canonical, 'label': raw.label}).groupby('canonical').label.nunique()
    n1 = int(raw.label.eq(1).sum())
    result = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'executed_at': datetime.now(timezone.utc).isoformat(),
              'raw_rows': len(raw), 'columns': raw.shape[1], 'unique_molecules': len(unique),
              'same_structure_label_conflicts': int(same_label.gt(1).sum()),
              'majority_f1': 2 * n1 / (len(raw) + n1),
              'quality': quality, 'class_balance': classes, 'correlations': corr, 'fingerprints': fp_summary}
    (ROOT / 'analysis/eda_basic_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result

if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
