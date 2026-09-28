"""Tables for `cart_global_surrogate` (empirical Fid, plain tree vs RL).

    python -m revision.global_surrogate_report
"""
import json,glob,numpy as np,sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.metrics import paired_wilcoxon
from revision.paper_stats import DATASETS
O=str(REPO.parent)+'/results/paper_final_cart_fixed/global_surrogate/'
S=[42,43,44,45,46]
C={(ds,s):json.load(open(f'{O}{ds}__seed{s}.json')) for ds in DATASETS for s in S}
bad=[(k,m,v['failures'],v['cov_rebuilt'],v['cov']) for k,c in C.items() for m,v in c['rl'].items() if v['failures'] or abs(v['cov_rebuilt']-v['cov'])>1e-9]
print('rebuild problems',len(bad),bad[:5])
nm=lambda s:{'folktables_income_CA_2018':'folktables','wyodot_kvdw_labeled':'wyodot','breast_cancer':'breast cancer','uci_credit':'uci credit','uci_adult':'uci adult'}.get(s,s)
T=['depth 2','depth 3','depth 4','depth 5','depth 8','val-tuned depth','fully grown']
def dm(f): return {ds:np.nanmean([f(C[(ds,s)]) for s in S]) for ds in DATASETS}
def mean(f): d=dm(f); return np.nanmean(list(d.values()))
def med(f): d=dm(f); return np.median(list(d.values()))
L=[]
L.append("| Tree | Fidelity (= Eff) | Coverage | Leaves (mean / median) | Depth | Conditions / leaf | Conditions / test row | Rows in a leaf with Fid < 0.90 | Accuracy vs y |")
L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for t in T:
    g=lambda q:(lambda c:c['trees'][t][q])
    L.append(f"| {t} | {mean(g('fid')):.3f} | 1.000 | {mean(g('n_leaves')):.0f} / {med(g('n_leaves')):.0f} | {mean(g('depth')):.1f} | {mean(g('cond_per_leaf')):.1f} | {mean(g('cond_per_row')):.1f} | {mean(g('rows_in_weak_leaf')):.0%} | {mean(g('acc')):.3f} |")
for m in ('rlda','mada'):
  for k in (1,5):
    L.append(f"| {m.upper()} k={k} | {mean(lambda c:c['rl'][f'{m}_k{k}']['fid']):.3f} | {mean(lambda c:c['rl'][f'{m}_k{k}']['cov']):.3f} | Eff {mean(lambda c:c['rl'][f'{m}_k{k}']['eff']):.3f} | | | | | |")
print("\n".join(L)); T1=L
# same rows
L=["| Rule set | RL Fid on its decided rows | depth 3 | depth 5 | val-tuned | fully grown | depth 3 on RL's abstained rows | val-tuned on abstained |","|---|---:|---:|---:|---:|---:|---:|---:|"]
P=[]
for m in ('rlda','mada'):
  for k in (1,5):
    key=f'{m}_k{k}'; rl=dm(lambda c:c['rl'][key]['fid'])
    cells=[]
    for t in ('depth 3','depth 5','val-tuned depth','fully grown'):
        tr=dm(lambda c:c['trees'][t][f'fid_on_{key}_decided'])
        x=np.array([rl[d] for d in DATASETS]); y=np.array([tr[d] for d in DATASETS])
        p=paired_wilcoxon(list(x),list(y))['pvalue']; P.append(p)
        cells.append(f"{y.mean():.3f} ({(x>y).sum()}/12 RL higher; p={p:.3f})")
    ab3=mean(lambda c:c['trees']['depth 3'][f'fid_on_{key}_abstained'] if c['trees']['depth 3'][f'fid_on_{key}_abstained'] is not None else np.nan)
    abt=mean(lambda c:c['trees']['val-tuned depth'][f'fid_on_{key}_abstained'] if c['trees']['depth 3'][f'fid_on_{key}_abstained'] is not None else np.nan)
    L.append(f"| {m.upper()} k={k} | {np.mean(list(rl.values())):.3f} | "+" | ".join(cells)+f" | {ab3:.3f} | {abt:.3f} |")
print("\n".join(L)); T2=L
L=["| Dataset | classes | RLDA k=1 Fid / Cov | depth 3 Fid (leaves) | val-tuned Fid (depth; leaves) | fully grown Fid (leaves) | depth 3 on RLDA-decided rows |","|---|---:|---:|---:|---:|---:|---:|"]
for ds in DATASETS:
    c0=C[(ds,42)]; f=lambda q:np.mean([q(C[(ds,s)]) for s in S])
    L.append(f"| {nm(ds)} | {c0['n_classes']} | {f(lambda c:c['rl']['rlda_k1']['fid']):.3f} / {f(lambda c:c['rl']['rlda_k1']['cov']):.3f} | {f(lambda c:c['trees']['depth 3']['fid']):.3f} ({f(lambda c:c['trees']['depth 3']['n_leaves']):.0f}) | {f(lambda c:c['trees']['val-tuned depth']['fid']):.3f} ({','.join(str(C[(ds,s)]['val_tuned_depth']) for s in S)}; {f(lambda c:c['trees']['val-tuned depth']['n_leaves']):.0f}) | {f(lambda c:c['trees']['fully grown']['fid']):.3f} ({f(lambda c:c['trees']['fully grown']['n_leaves']):.0f}) | {f(lambda c:c['trees']['depth 3']['fid_on_rlda_k1_decided']):.3f} |")
print("\n".join(L)); T3=L
# classes missing at depth 2/3
for t in ('depth 2','depth 3'):
    miss=[(nm(ds),s) for ds in DATASETS for s in S if C[(ds,s)]['trees'][t]['classes_with_leaf']<C[(ds,s)]['n_classes']]
    print(t,'cells with a class that has no leaf:',len(miss),sorted(set(m[0] for m in miss)))
# class-wise precision at depth 3 and tuned, per dataset
L=["| Dataset | class | depth 3 precision / recall (leaves) | val-tuned precision / recall (leaves) |","|---|---:|---:|---:|"]
for ds in DATASETS:
    for c in range(C[(ds,42)]['n_classes']):
        def g(t,q):
            v=[C[(ds,s)]['trees'][t]['per_class'][str(c)][q] for s in S]; v=[x for x in v if x is not None]
            return np.mean(v) if v else np.nan
        L.append(f"| {nm(ds) if c==0 else ''} | {c} | {g('depth 3','precision'):.3f} / {g('depth 3','recall'):.3f} ({g('depth 3','n_leaves'):.1f}) | {g('val-tuned depth','precision'):.3f} / {g('val-tuned depth','recall'):.3f} ({g('val-tuned depth','n_leaves'):.0f}) |")
T4=L
print(C[('breast_cancer',42)]['trees']['depth 3']['rules_preview'])
hdr="# CART as a plain global surrogate\n\nGenerated 2026-09-26 by `revision/cart_global_surrogate.py` (`revision/global_surrogate_report.py`). One DecisionTreeClassifier fit to f_hat(D_train) in original units, every leaf is a rule, no top-k, no abstention: coverage = 1 and Fid = Eff = agreement with f_hat on D_test. 12 datasets × 5 seeds, means over datasets of 5-seed means. RL from `paper_final_valtb` (emp τ_C = 0.10; k = 5 from the k-sweep).\n\n"
open(O+'SUMMARY.md','w').write(hdr+"## Overall\n\n"+"\n".join(T1)+"\n\n## Same rows: agreement on the D_test rows each RL rule set decides (Wilcoxon over 12 datasets, unadjusted)\n\n"+"\n".join(T2)+"\n\n## Dataset-wise\n\n"+"\n".join(T3)+"\n\n## Class-wise (precision = Fid of the class's leaves; recall = share of f_hat's class-c rows the tree labels c)\n\n"+"\n".join(T4)+"\n")
