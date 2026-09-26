"""Tables for `containment_eval`: stored π vs π+ (containment fix) vs Anchors, per instance.

Prints the tables in `../results/containment_fix/SUMMARY.md`; uses every seed file present.

    python -m revision.containment_report [--cfix_dir ../results/containment_fix]
"""
import argparse, json, glob, collections, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.metrics import paired_wilcoxon
from revision.paper_stats import DATASETS
ap=argparse.ArgumentParser(); ap.add_argument('--cfix_dir',default=str(REPO.parent/'results'/'containment_fix')); R=ap.parse_args().cfix_dir.rstrip('/')+'/'
nm=lambda s:{'folktables_income_CA_2018':'folktables','wyodot_kvdw_labeled':'wyodot'}.get(s,s)
T=('pi','pi_contained','anchors')
out=[]
for arm in ('rlda','mada'):
    per=collections.defaultdict(lambda: collections.defaultdict(list))
    for ds in DATASETS:
        for f in sorted(glob.glob(f'{R}{ds}__{arm}__seed*.json')):
            r=json.load(open(f))
            for t in T:
                xs=[x[t] for x in r['rows'] if x.get(t)]
                if not xs: continue
                d=per[ds]
                d[t+'|contain'].append(np.mean([x['contains_x'] for x in xs])); d[t+'|cond'].append(np.nanmean([x['cond_fid'] for x in xs]))
                d[t+'|ok'].append(np.mean([x['cond_fid']>=0.9 for x in xs])); d[t+'|cov'].append(np.mean([x['coverage'] for x in xs]))
                d[t+'|emp'].append(np.nanmean([x['emp_fid'] for x in xs])); d[t+'|act'].append(np.mean([x['n_active'] for x in xs]))
                if t=='anchors': d['self'].append(np.nanmean([x['self_reported_precision'] if x['self_reported_precision'] is not None else np.nan for x in xs]))
            per[ds]['n'].append(r['n'])
    print(f"\n=== {arm.upper()} per-instance, seeds 42-43, same test points; Anchors-style D(z|A) conditional Fid for all ===")
    print(f"{'dataset':12s}{'n':>5s} | {'contain π/π+/A':>18s} | {'condFid π/π+/A':>20s} | {'≥0.90 π/π+/A':>17s} | {'cov π/π+/A':>20s} | {'conds π+/A':>10s} | {'A self':>7s}")
    cols=collections.defaultdict(list)
    for ds,d in per.items():
        m={k:np.mean(v) for k,v in d.items()}
        for k,v in m.items(): cols[k].append(v)
        print(f"{nm(ds):12s}{int(np.sum(d['n'])):5d} | {m['pi|contain']:.2f}/{m['pi_contained|contain']:.2f}/{m['anchors|contain']:.2f}     | {m['pi|cond']:.3f}/{m['pi_contained|cond']:.3f}/{m['anchors|cond']:.3f}  | {m['pi|ok']:.2f}/{m['pi_contained|ok']:.2f}/{m['anchors|ok']:.2f}   | {m['pi|cov']:.3f}/{m['pi_contained|cov']:.3f}/{m['anchors|cov']:.3f}  | {m['pi_contained|act']:.1f}/{m['anchors|act']:.1f}   | {m['self']:.3f}")
    m={k:np.mean(v) for k,v in cols.items()}
    print(f"{'mean':12s}{'':5s} | {m['pi|contain']:.2f}/{m['pi_contained|contain']:.2f}/{m['anchors|contain']:.2f}     | {m['pi|cond']:.3f}/{m['pi_contained|cond']:.3f}/{m['anchors|cond']:.3f}  | {m['pi|ok']:.2f}/{m['pi_contained|ok']:.2f}/{m['anchors|ok']:.2f}   | {m['pi|cov']:.3f}/{m['pi_contained|cov']:.3f}/{m['anchors|cov']:.3f}  | {m['pi_contained|act']:.1f}/{m['anchors|act']:.1f}   | {m['self']:.3f}")
    for a,b,k,lab in (('pi_contained','pi','cond','cond Fid: fixed vs stored π'),('pi_contained','pi','cov','coverage: fixed vs stored π'),('pi_contained','anchors','cond','cond Fid: fixed π vs Anchors'),('pi_contained','anchors','ok','share ≥0.90: fixed π vs Anchors'),('pi_contained','anchors','cov','coverage: fixed π vs Anchors')):
        x=cols[f'{a}|{k}']; y=cols[f'{b}|{k}']; w=paired_wilcoxon(x,y)
        print(f"  {lab}: A higher {(np.array(x)>np.array(y)).sum()}/{len(x)}, mean Δ {np.mean(np.array(x)-np.array(y)):+.3f}, p={w['pvalue']:.4f}")
    x=cols['self']; y=cols['anchors|cond']; print(f"  Anchors self-reported vs measured precision: mean {np.mean(x):.3f} vs {np.mean(y):.3f}")
