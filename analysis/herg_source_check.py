"""Compare structures with public hERG datasets; never use external labels as model inputs.
Run with pandas and RDKit: python analysis/herg_source_check.py
Downloads are cached in the system temporary directory. Raw local data is read-only.
"""
from pathlib import Path
import hashlib, json, tarfile, tempfile, urllib.request
import pandas as pd
from rdkit import Chem, RDLogger, rdBase

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
 'cardiotox': ('https://raw.githubusercontent.com/Abdulk084/CardioTox/master/data/train_validation_cardio_tox_data.tar.xz', 'cardiotox-data.tar.xz'),
 'toxtree': ('https://zenodo.org/records/5807719/files/hERG_Dataset.csv?download=1', 'zenodo-herg.csv'),
}
def canonical(value):
    mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None

def run():
    RDLogger.DisableLog('rdApp.*')
    local_path = ROOT/'Data/input_data/train.csv'
    local = pd.read_csv(local_path, usecols=['SMILES', 'label'])
    local['canonical'] = local.SMILES.map(canonical)
    assert local.canonical.notna().all()
    result = {'local_rows': len(local), 'local_sha256': hashlib.sha256(local_path.read_bytes()).hexdigest(),
              'rdkit_version': rdBase.rdkitVersion,
              'matching': 'canonical isomeric SMILES, default RDKit; no desalting or tautomer normalization',
              'sources': {}}
    for name, (url, filename) in SOURCES.items():
        path = Path(tempfile.gettempdir())/filename
        if not path.exists(): urllib.request.urlretrieve(url, path)
        if name == 'cardiotox':
            with tarfile.open(path) as archive:
                with archive.extractfile('train_validation_cardio_tox_data.csv') as stream:
                    public = pd.read_csv(stream, usecols=['smiles','ACTIVITY']).rename(columns={'smiles':'SMILES'})
        else:
            public = pd.read_csv(path)
        public['canonical'] = public.SMILES.map(canonical)
        matched = local.merge(public.dropna(subset=['canonical']), on='canonical', suffixes=('_local','_public'))
        stats = {'url':url, 'download_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'public_rows':len(public), 'public_parse_failures':int(public.canonical.isna().sum()),
                 'matched_local_rows':int(local.canonical.isin(set(public.canonical.dropna())).sum()),
                 'matched_local_unique_molecules':int(local.loc[local.canonical.isin(set(public.canonical.dropna())), 'canonical'].nunique()),
                 'joined_pairs':len(matched)}
        if name == 'cardiotox':
            stats['inverse_label_agreement_pairs'] = int((matched.label == 1-matched.ACTIVITY).sum())
            stats['inverse_label_agreement_percent'] = float((matched.label == 1-matched.ACTIVITY).mean()*100)
            matched.loc[matched.label != 1-matched.ACTIVITY].to_csv(ROOT/'analysis/herg_label_disagreements.csv', index=False)
            stats['public_activity_counts'] = {str(k):int(v) for k,v in public.ACTIVITY.value_counts().items()}
        else:
            stats['label0_vs_pIC50_ge5_agreement_percent_pairs'] = float((matched.label.eq(0) == matched.PIC50.ge(5)).mean()*100)
            stats['note'] = 'Exploratory threshold comparison, not proof of local assay threshold; duplicate public keys can multiply pairs.'
        result['sources'][name] = stats
    result['conclusion'] = 'Strong evidence of hERG-related labels; exact local lineage and assay cut-off remain unverified. Do not relabel disagreements.'
    (ROOT/'analysis/herg_source_evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
if __name__ == '__main__': run()
