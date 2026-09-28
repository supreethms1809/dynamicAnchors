"""Tables for `tree_instance_eval`: plain-surrogate leaves vs RL (π+) vs Anchors, per instance.

Reads `containment_eval` / `tree_instance_eval` output; uses every seed with all
three files present (42-43 locally, 42-46 once spark's files are copied in).

    python -m revision.instance_surrogate_report [--cfix_dir ../results/containment_fix]
"""
import argparse, glob, json, collections, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.metrics import paired_wilcoxon
from revision.paper_stats import DATASETS
ap=argparse.ArgumentParser(); ap.add_argument('--cfix_dir',default=str(REPO.parent/'results'/'containment_fix')); A=ap.parse_args()
R=A.cfix_dir.rstrip('/')+'/'
SEEDS=sorted({int(p.split('seed')[-1][:-5]) for p in glob.glob(R+'tree/*__seed*.json')})
SEEDS=[s for s in SEEDS if all(Path(f'{R}{ds}__{a}__seed{s}.json').is_file() and Path(f'{R}tree/{ds}__seed{s}.json').is_file() for ds in DATASETS for a in ('rlda','mada'))]
print('seeds',SEEDS)
nm=lambda s:{'folktables_income_CA_2018':'folktables','wyodot_kvdw_labeled':'wyodot','breast_cancer':'breast cancer','uci_credit':'uci credit','uci_adult':'uci adult'}.get(s,s)
M=['RLDA','MADA','Anchors','Tree d3','Tree d5','Tree tuned']
Q=['cond','ok','cov','act','contain','emp','agree']
per={ds:collections.defaultdict(list) for ds in DATASETS}; mism=0; ntot=0; depths=collections.defaultdict(list); serve=[];fit=[]
def add(d,m,xs):
    d[m+'|cond'].append(np.nanmean([x['cond_fid'] for x in xs])); d[m+'|ok'].append(np.mean([x['cond_fid']>=0.9 for x in xs]))
    d[m+'|cov'].append(np.mean([x['coverage'] for x in xs])); d[m+'|act'].append(np.mean([x['n_active'] for x in xs]))
    d[m+'|contain'].append(np.mean([x['contains_x'] for x in xs])); d[m+'|emp'].append(np.nanmean([x['emp_fid'] for x in xs]))
    if 'tree_label_matches_y_hat' in xs[0]: d[m+'|agree'].append(np.mean([x['tree_label_matches_y_hat'] for x in xs]))
for ds in DATASETS:
    for s in SEEDS:
        r=json.load(open(f'{R}{ds}__rlda__seed{s}.json'))['rows']; a=json.load(open(f'{R}{ds}__mada__seed{s}.json'))['rows']; t=json.load(open(f'{R}tree/{ds}__seed{s}.json'))
        assert [x['index'] for x in r]==[x['index'] for x in t['rows']]==[x['index'] for x in a]
        d=per[ds]
        add(d,'RLDA',[x['pi_contained'] for x in r if x.get('pi_contained')]); add(d,'MADA',[x['pi_contained'] for x in a if x.get('pi_contained')])
        add(d,'Anchors',[x['anchors'] for x in r if x.get('anchors')])
        for m,k in (('Tree d3','depth 3'),('Tree d5','depth 5'),('Tree tuned','val-tuned depth')): add(d,m,[x[k] for x in t['rows']])
        mism+=sum(x['y_hat']!=x.get('y_hat_original_units',x['y_hat']) for x in t['rows']); ntot+=len(t['rows'])
        depths[ds].append(t['val_tuned_depth']); serve.append(t['serve_seconds_per_input']['depth 3']); fit.append(t['fit_seconds'])
        d['n'].append(len(t['rows']))
V={ds:{k:np.mean(v) for k,v in per[ds].items()} for ds in DATASETS}
L=[]
L.append("| Method | Pert precision (D(z|A)) | Share ≥ 0.90 | Coverage | Conditions | Contains x* | Tree label = ŷ(x*) |")
L.append("|---|---:|---:|---:|---:|---:|---:|")
for m in M:
    g=lambda q: np.mean([V[ds].get(f'{m}|{q}',np.nan) for ds in DATASETS])
    L.append(f"| {m} | {g('cond'):.3f} | {g('ok'):.0%} | {g('cov'):.3f} | {g('act'):.1f} | {g('contain'):.0%} | {'—' if not m.startswith('Tree') else f'{g(chr(97)+chr(103)+chr(114)+chr(101)+chr(101)):.1%}'} |")
T1=L; print("\n".join(L))
def holm(ps):
    o=np.argsort(ps); n=len(ps); adj=np.empty(n); run=0
    for r,i in enumerate(o): run=max(run,min(1,(n-r)*ps[i])); adj[i]=run
    return adj
pairs=[('Tree d3','RLDA'),('Tree d3','MADA'),('Tree d3','Anchors'),('Tree tuned','RLDA'),('Tree tuned','MADA'),('Tree tuned','Anchors')]
L=["| Contrast | Pert precision Δ (tree higher; p_Holm) | Share ≥ 0.90 Δ (tree higher; p_Holm) | Coverage Δ (tree higher; p_Holm) |","|---|---|---|---|"]
res={}
for q in ('cond','ok','cov'):
    ps=[];cells=[]
    for a,b in pairs:
        x=np.array([V[ds][f'{a}|{q}'] for ds in DATASETS]); y=np.array([V[ds][f'{b}|{q}'] for ds in DATASETS])
        ps.append(paired_wilcoxon(list(x),list(y))['pvalue']); cells.append(((x-y).mean(),(x>y).sum()))
    res[q]=[(c,p) for c,p in zip(cells,holm(np.array(ps)))]
for i,(a,b) in enumerate(pairs):
    L.append(f"| {a} vs {b} | "+" | ".join(f"{res[q][i][0][0]:+.3f} ({res[q][i][0][1]}/12; {res[q][i][1]:.3f})" for q in ('cond','ok','cov'))+" |")
T2=L; print("\n".join(L))
L=["| Dataset | n | RLDA | MADA | Anchors | Tree d3 | Tree d5 | Tree tuned (depths) | Tree d3 label = ŷ |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for ds in DATASETS:
    v=V[ds]; L.append(f"| {nm(ds)} | {int(sum(per[ds]['n']))} | "+" | ".join(f"{v[m+'|cond']:.3f}" for m in M[:5])+f" | {v['Tree tuned|cond']:.3f} ({','.join(str(x) for x in depths[ds])}) | {v['Tree d3|agree']:.0%} |")
T3=L; print("\n".join(L))
print('y_hat unit vs original mismatches',mism,'of',ntot,'| depth-3 serve s/input',np.mean(serve),'fit s',np.mean(fit))
hdr=("# Instance explanations: plain global surrogate vs RL vs Anchors\n\nGenerated by `revision/instance_surrogate_report.py` from `revision/tree_instance_eval.py` output. Same test points as `containment_fix`, seeds "+','.join(map(str,SEEDS))+". "
"Tree explanation of x* = the leaf of a DecisionTreeClassifier fit to f_hat(D_train) that contains x* (open faces). RLDA/MADA = policy boxes with the containment fix (π+). "
"All scored with Anchors' D(z|A) over D_train, 2,000 draws, target ŷ(x*) (unit-space ŷ, as for RL/Anchors). Means over datasets of per-seed means.\n\n")
open(R+'tree/SUMMARY.md','w').write(hdr+"\n".join(T1)+"\n\nPaired Wilcoxon over 12 datasets, Holm over the 6 contrasts per metric\n\n"+"\n".join(T2)+"\n\nDataset-wise pert precision\n\n"+"\n".join(T3)+f"\n\nŷ on unit vs original inputs differs for {mism} of {ntot} points. Depth-3 serving: {np.mean(serve)*1e3:.2f} ms per input; fit {np.mean(fit):.1f} s per cell including the D_val depth search.\n")
