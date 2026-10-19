"""Reproducible robustness checks for EDA; source CSV is read-only.
Run from repository root with pandas, numpy, scipy, RDKit, scikit-learn.
"""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

ROOT=Path(__file__).resolve().parents[1]
def run():
    RDLogger.DisableLog('rdApp.*')
    path=ROOT/'Data/input_data/train.csv'
    raw=pd.read_csv(path)
    mols=[Chem.MolFromSmiles(s) for s in raw.SMILES]
    assert all(m is not None for m in mols)
    canonical=pd.Series([Chem.MolToSmiles(m,isomericSmiles=True) for m in mols])
    keep=~canonical.duplicated().to_numpy()
    d=raw.loc[keep].copy();d['toxic']=d.label.eq(0).astype(int)
    scaffolds=np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
                        for m,k in zip(mols,keep) if k])
    names,group=np.unique(scaffolds,return_inverse=True)
    toxic=d.toxic.to_numpy()
    bins,edges=pd.qcut(d.clogp,5,labels=False,retbins=True)
    low=bins.to_numpy()==0;high=bins.to_numpy()==4
    low_rate=float(toxic[low].mean());high_rate=float(toxic[high].mean())
    observed=high_rate-low_rate
    agg=np.column_stack([np.bincount(group,weights=low*toxic),np.bincount(group,weights=low),
                         np.bincount(group,weights=high*toxic),np.bincount(group,weights=high)])
    rng=np.random.default_rng(20261019)
    replicates=[]
    for _ in range(20):
        sampled=rng.integers(0,len(names),size=(100,len(names)))
        totals=agg[sampled].sum(axis=1)
        assert (totals[:,[1,3]]>0).all()
        replicates.extend(totals[:,2]/totals[:,3]-totals[:,0]/totals[:,1])
    cluster_ci=np.quantile(replicates,[.025,.975])
    iid=(rng.binomial(high.sum(),high_rate,size=2000)/high.sum()
         -rng.binomial(low.sum(),low_rate,size=2000)/low.sum())
    iid_ci=np.quantile(iid,[.025,.975])
    intervals=[{'method':'분자별 재표집','difference_pp':observed*100,'low_pp':float(iid_ci[0]*100),
                'high_pp':float(iid_ci[1]*100),'replicates':2000},
               {'method':'골격별 재표집','difference_pp':observed*100,'low_pp':float(cluster_ci[0]*100),
                'high_pp':float(cluster_ci[1]*100),'replicates':2000}]
    # Reuse the original discovery-selected nine candidates; no new significance selection.
    bits=['ecfp_807','ecfp_650','ecfp_893','fcfp_18','fcfp_445','fcfp_146','ptfp_284','ptfp_53','ptfp_756']
    reference={b:(-1 if b in ['ecfp_807','ecfp_650','ecfp_893','ptfp_53','ptfp_756'] else 1) for b in bits}
    splitrows=[]
    for seed in range(20261020,20261040):
        _,ids=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(d,groups=scaffolds))
        other=np.setdiff1d(np.arange(len(d)),ids)
        assert not(set(scaffolds[ids])&set(scaffolds[other]))
        yy=toxic[ids]
        for bit in bits:
            present=d[bit].to_numpy()[ids]==1
            n1=int(present.sum());n0=int((~present).sum())
            diff=float(yy[present].mean()-yy[~present].mean()) if n1 and n0 else None
            splitrows.append({'seed':seed,'bit':bit,'check_n':len(ids),'active_n':n1,'inactive_n':n0,
              'difference_pp':diff*100 if diff is not None else None,
              'same_direction':bool(np.sign(diff)==reference[bit]) if diff is not None else False})
    splitdf=pd.DataFrame(splitrows)
    split_summary=[]
    for bit,g in splitdf.groupby('bit',sort=False):
        split_summary.append({'bit':bit,'direction_consistency_percent':float(g.same_direction.mean()*100),
          'repeats':len(g),'min_active':int(g.active_n.min()),'min_inactive':int(g.inactive_n.min()),
          'median_difference_pp':float(g.difference_pp.median()),'min_difference_pp':float(g.difference_pp.min()),
          'max_difference_pp':float(g.difference_pp.max())})
    # Row-quarter sensitivity describes file composition, not measured collection batches.
    quarters=pd.cut(d.index,bins=[-1,2087,4174,6261,8348],labels=['앞 25%','25~50%','50~75%','뒤 25%'])
    quarter_rows=[]
    for name in quarters.categories:
        mask=np.asarray(quarters==name)
        l=mask&low;h=mask&high
        assert l.sum()>30 and h.sum()>30
        quarter_rows.append({'quarter':name,'low_n':int(l.sum()),'high_n':int(h.sum()),
          'low_rate_percent':float(toxic[l].mean()*100),'high_rate_percent':float(toxic[h].mean()*100),
          'difference_pp':float((toxic[h].mean()-toxic[l].mean())*100)})
    # Verify conflict/column counts independent of the earlier notebook outputs.
    fps=[c for c in raw if c.startswith(('ecfp_','fcfp_','ptfp_'))]
    fp=raw[fps].to_numpy(dtype=np.uint8)
    packed=np.packbits(fp,axis=1)
    props=raw[['MolWt','clogp','sa_score','qed']].to_numpy(dtype=np.float64);props[props==0]=0.
    groups={}
    for i,(a,b) in enumerate(zip(packed,props)):groups.setdefault(a.tobytes()+b.tobytes(),[]).append(i)
    labels=raw.label.to_numpy()
    mixed=[ids for ids in groups.values() if len(set(labels[ids]))>1]
    assert len(mixed)==33 and sum(map(len,mixed))==73
    result={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'executed_at':datetime.now(timezone.utc).isoformat(),'raw_n':len(raw),'unique_n':len(d),
      'scaffold_n':len(names),'clogp_low':low_rate*100,'clogp_high':high_rate*100,
      'clogp_edges':edges.tolist(),'bootstrap':intervals,'split_stability':split_summary,
      'split_details':splitrows,'row_quarters':quarter_rows,
      'scope':{'bootstrap':'동일 골격의 모든 분자를 함께 복원추출; 2,000회; 분위 경계 고정',
               'prediction_input_available':(ROOT/'Data/input_data/predict_input.csv').exists(),
               'independent_validation':False}}
    out=ROOT/'analysis/eda_review_results.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['split_details']},ensure_ascii=False,indent=2))
    return result
if __name__=='__main__':run()
