"""Tables for `cart_global_surrogate_pert` (perturbation Fid, plain / pert-fit tree vs RL).

    python -m revision.global_surrogate_pert_report
"""
import json,numpy as np,sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.metrics import paired_wilcoxon
from revision.paper_stats import DATASETS
O=str(REPO.parent)+'/results/paper_final_cart_fixed/global_surrogate_pert/'
S=[42,43,44,45,46]
C={(ds,s):json.load(open(f'{O}{ds}__seed{s}.json')) for ds in DATASETS for s in S}
print('rebuild failures',sum(v['rebuild_failures'] for c in C.values() for v in c['rl'].values()))
nm=lambda s:{'folktables_income_CA_2018':'folktables','wyodot_kvdw_labeled':'wyodot','breast_cancer':'breast cancer','uci_credit':'uci credit','uci_adult':'uci adult'}.get(s,s)
TB=json.load(open(str(REPO.parent)+'/results/paper_final_cart_fixed/trackb_rules_emp_tc0p10.json'))
def get(c,m,q):
    return c['trees'][m][q] if m in c['trees'] else c['rl'][m].get(q,np.nan)
def dm(m,q): return {ds:np.nanmean([get(C[(ds,s)],m,q) for s in S]) for ds in DATASETS}
def mean(m,q): return np.nanmean(list(dm(m,q).values()))
TREES=[f'{f} / {d}' for f in ('plain','pert fit') for d in ('depth 3','depth 5','val-tuned depth','fully grown')]
RL=['rlda_emp','mada_emp','rlda_pert','mada_pert']
lab={'rlda_emp':'RLDA, trained on emp Fid','mada_emp':'MADA, trained on emp Fid','rlda_pert':'RLDA, trained on pert Fid','mada_pert':'MADA, trained on pert Fid'}
L=["| Method | Pert Fid (row-weighted) | Pert Fid (mean over rules) | Rows whose rule has pert Fid ≥ 0.90 | Emp Fid | Coverage | Rules (mean / median) | Conditions / rule |","|---|---:|---:|---:|---:|---:|---:|---:|"]
for m in TREES:
    L.append(f"| Tree, {m} | {mean(m,'pert_fid_rows'):.3f} | {mean(m,'pert_fid_rule_mean'):.3f} | {mean(m,'rows_pert_ge_tau'):.0%} | {mean(m,'emp_fid'):.3f} | 1.000 | {mean(m,'n_leaves'):.0f} / {np.median(list(dm(m,'n_leaves').values())):.0f} | {mean(m,'cond_per_leaf'):.1f} |")
for m in RL:
    L.append(f"| {lab[m]} | {mean(m,'pert_fid_rows'):.3f} | {mean(m,'pert_fid_rule_mean'):.3f} | {mean(m,'rows_pert_ge_tau'):.0%} | {mean(m,'emp_fid'):.3f} | {mean(m,'coverage'):.3f} | {mean(m,'n_rules'):.1f} | {mean(m,'cond_per_rule'):.1f} |")
ga=np.mean([np.mean([r['fid_anchors'] for r in TB if r['dataset']==ds and r['method']=='greedy_anchors']) for ds in DATASETS])
L.append(f"| Greedy anchors (Track B, 2000 draws, reference) | — | {ga:.3f} | — | | | | |")
print("\n".join(L)); T1=L
def holm(ps):
    o=np.argsort(ps); n=len(ps); adj=np.empty(n); run=0
    for r,i in enumerate(o): run=max(run,min(1,(n-r)*ps[i])); adj[i]=run
    return adj
CMP=['plain / depth 3','pert fit / depth 3','plain / val-tuned depth','pert fit / val-tuned depth']
rows=[];P=[]
for a in RL:
    for b in CMP:
        x=np.array(list(dm(a,'pert_fid_rows').values())); y=np.array(list(dm(b,'pert_fid_rows').values()))
        P.append(paired_wilcoxon(list(x),list(y))['pvalue']); rows.append((a,b,np.mean(x-y),(x>y).sum()))
adj=holm(np.array(P))
L=["| RL | vs tree | Δ pert Fid (RL − tree) | RL higher | p_Holm (16 tests) |","|---|---|---:|---:|---:|"]
for (a,b,d,w),p in zip(rows,adj): L.append(f"| {lab[a]} | {b} | {d:+.3f} | {w}/12 | {p:.3f} |")
print("\n".join(L)); T2=L
L=["| Dataset | RLDA emp | MADA emp | RLDA pert | MADA pert | plain d3 | pert-fit d3 | plain tuned | pert-fit tuned |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
cols=RL+['plain / depth 3','pert fit / depth 3','plain / val-tuned depth','pert fit / val-tuned depth']
D={m:dm(m,'pert_fid_rows') for m in cols}
for ds in DATASETS: L.append(f"| {nm(ds)} | "+" | ".join(f"{D[m][ds]:.3f}" for m in cols)+" |")
print("\n".join(L)); T3=L
hdr=("# Plain CART global surrogate on perturbation fidelity\n\nGenerated 2026-09-26 by `revision/cart_global_surrogate_pert.py` (`revision/global_surrogate_pert_report.py`). "
"Every rule (tree leaf, or RL selected rule) scored with Anchors' D(z|B) (`sample_anchors_conditional`, 512 draws; matches the Track B file within MC noise). "
"Row-weighted = each rule weighted by the D_test rows it covers (for a tree, the leaf of each test row). Plain fit = f_hat(D_train); pert fit = D_train + an equal number of recombined rows (independent train marginals) labelled by f_hat, as `run_cart` does for the perturbed estimator. "
"RL from `paper_final_valtb` {emp,pert}_tc0p10, k = 1. 12 datasets × 5 seeds, means over datasets.\n\n")
open(O+'SUMMARY.md','w').write(hdr+"## Overall\n\n"+"\n".join(T1)+"\n\n## Paired tests on row-weighted pert Fid\n\n"+"\n".join(T2)+"\n\n## Dataset-wise row-weighted pert Fid\n\n"+"\n".join(T3)+"\n")
