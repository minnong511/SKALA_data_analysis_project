"""Draw actual dataset molecules for the report; source data stays unchanged."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor

ROOT = Path(__file__).resolve().parents[1]

def run():
    RDLogger.DisableLog('rdApp.*')
    path = ROOT / 'Data/input_data/train.csv'
    raw = pd.read_csv(path)
    mols = [Chem.MolFromSmiles(s) for s in raw.SMILES]
    canonical = pd.Series([Chem.MolToSmiles(m, isomericSmiles=True) for m in mols])
    dupkey = canonical.loc[canonical.duplicated()].iloc[0]
    duplicate_ids = canonical.index[canonical.eq(dupkey)].tolist()
    assert len(duplicate_ids) == 2 and raw.loc[duplicate_ids, 'label'].nunique() == 1
    properties = ['MolWt', 'clogp', 'sa_score', 'qed']
    numeric = [c for c in raw if c.startswith(('ecfp_', 'fcfp_', 'ptfp_'))] + properties
    assert raw.loc[98, numeric].equals(raw.loc[99, numeric])
    assert raw.loc[98, 'label'] != raw.loc[99, 'label']
    assert '@' not in raw.loc[98, 'SMILES'] and '@' in raw.loc[99, 'SMILES']
    assets = ROOT / 'reports/eda-review/src/content/assets'
    assets.mkdir(parents=True, exist_ok=True)
    rows = []
    selections = [('molecule-example', 0, '분자 기록 예시'),
                  ('duplicate-a', duplicate_ids[0], '동일 구조의 첫 기록'),
                  ('duplicate-b', duplicate_ids[1], '동일 구조의 두 번째 기록'),
                  ('stereo-unspecified', 98, '공간 배치 정보 없음'),
                  ('stereo-specified', 99, '공간 배치 정보 있음')]
    # Match the coordinates so the specified wedge is easy to compare.
    rdDepictor.Compute2DCoords(mols[98])
    rdDepictor.GenerateDepictionMatching2DStructure(mols[99], mols[98])
    for name, idx, description in selections:
        mol = Chem.Mol(mols[idx])
        if name.startswith('duplicate'):
            mol = Chem.MolFromSmiles(dupkey)
        if not name.startswith('stereo'):
            rdDepictor.Compute2DCoords(mol)
        Draw.MolToImage(mol, size=(840, 400)).save(assets / f'{name}.png')
        first_on = next(i for i in range(1024) if raw.loc[idx, f'ecfp_{i}'] == 1)
        start = max(0, min(1000, first_on - 4))
        rows.append({'example': name, 'description': description, 'record': idx + 1,
                     'SMILES': raw.loc[idx, 'SMILES'], 'label': int(raw.loc[idx, 'label']),
                     **{p: float(raw.loc[idx, p]) for p in properties},
                     'ecfp_display_start': start,
                     'ecfp_display_bits': ''.join(str(int(raw.loc[idx, f'ecfp_{i}'])) for i in range(start, start + 24))})
    result = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'executed_at': datetime.now(timezone.utc).isoformat(), 'rows': rows,
              'drawing': 'RDKit 2D structural drawings from train.csv; not experimentally measured 3D shapes.'}
    (ROOT / 'analysis/eda_report_images.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'records': [r['record'] for r in rows], 'images': len(rows),
                      'same_numeric_input_verified': True, 'source_sha256': result['source_sha256']}, ensure_ascii=False))
    return result

if __name__ == '__main__':
    run()
