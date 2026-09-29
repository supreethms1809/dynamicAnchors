"""Tables for `containment_eval`: stored π vs π+ (containment fix) vs π∨ (policy OR) vs Anchors.

Prints the tables in `../results/containment_fix/SUMMARY.md`; uses every seed file present.

    python -m revision.containment_report [--cfix_dir ../results/containment_fix] [--fid cond_fid_all]

--fid cond_fid (default) resamples only faces narrower than 95% of the span;
cond_fid_all holds every face of the box (cells written after the exact-bins fix).
Test points whose anchor is empty (the class prior, no conditions) are dropped
from every arm unless --keep_empty.

π∨ (`pi_or`, cells written after the per-policy fix) is the OR of every policy's
contained box that clears D_val Fid >= 0.90: MADA's agents for the class, RLDA's one
policy. A point where no box clears it is unexplained; its precision columns
average over explained points only, and 'expl' is the explained share.
"""
import argparse, json, glob, collections, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.metrics import paired_wilcoxon
from revision.paper_stats import DATASETS
ap = argparse.ArgumentParser(); ap.add_argument('--cfix_dir', default=str(REPO.parent / 'results' / 'containment_fix'))
ap.add_argument('--fid', default='cond_fid', choices=['cond_fid', 'cond_fid_all']); ap.add_argument('--keep_empty', action='store_true')
A = ap.parse_args(); R = A.cfix_dir.rstrip('/') + '/'; F = A.fid
nm = lambda s: {'folktables_income_CA_2018': 'folktables', 'wyodot_kvdw_labeled': 'wyodot'}.get(s, s)
SEEDS = sorted({p.split('seed')[-1][:-5] for p in glob.glob(R + '*__rlda__seed*.json')})
HAS_OR = any('pi_or' in r for p in glob.glob(R + '*__seed*.json')[:3] for r in json.load(open(p))['rows'])
T = ('pi', 'pi_contained', 'pi_or', 'anchors') if HAS_OR else ('pi', 'pi_contained', 'anchors')
LAB = {'pi': 'π', 'pi_contained': 'π+', 'pi_or': 'π∨', 'anchors': 'A'}
tl = '/'.join(LAB[t] for t in T)
for arm in ('rlda', 'mada'):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for ds in DATASETS:
        for f in sorted(glob.glob(f'{R}{ds}__{arm}__seed*.json')):
            r = json.load(open(f))
            rows = [x for x in r['rows'] if A.keep_empty or not x.get('anchors_empty')]
            per[ds]['n_empty'].append(len(r['rows']) - len(rows))
            for t in T:
                xs = [x[t] for x in rows if x.get(t)]
                d = per[ds]
                if t == 'pi_or':
                    d['expl'].append(np.mean([bool(x.get('pi_or')) for x in rows]) if rows else np.nan)
                    d['boxes'].append(np.mean([x['n_boxes'] for x in xs]) if xs else np.nan)
                if not xs:
                    continue
                d[t + '|contain'].append(np.mean([x['contains_x'] for x in xs])); d[t + '|cond'].append(np.nanmean([x[F] for x in xs]))
                d[t + '|ok'].append(np.mean([x[F] >= 0.9 for x in xs])); d[t + '|cov'].append(np.mean([x['coverage'] for x in xs]))
                # conditions: n_cond (a face excludes a D_train row) when the cell has it
                d[t + '|act'].append(np.mean([x.get('n_cond', x['n_active']) for x in xs]))
                if t == 'anchors':
                    d['self'].append(np.nanmean([x['self_reported_precision'] if x['self_reported_precision'] is not None else np.nan for x in xs]))
            per[ds]['n'].append(len(rows))
    print(f"\n=== {arm.upper()} per-instance, seeds {','.join(SEEDS)}, same test points; Anchors-style D(z|A) conditional Fid ({F}) for all; "
          f"{int(sum(sum(d['n_empty']) for d in per.values()))} x* with an empty anchor {'kept' if A.keep_empty else 'dropped'} ===")
    extra = f" | {'expl':>5s} {'boxes':>5s}" if HAS_OR else ''
    print(f"{'dataset':12s}{'n':>5s} | {'contain ' + tl:>22s} | {'condFid ' + tl:>28s} | {'≥0.90 ' + tl:>22s} | {'cov ' + tl:>28s} | {'conds ' + tl:>18s} | {'A self':>7s}{extra}")
    cols = collections.defaultdict(list)

    def fmt(m, key, w):
        return '/'.join(f"{m.get(t + '|' + key, np.nan):.{w}f}" for t in T)

    for ds, d in per.items():
        m = {k: np.nanmean(v) for k, v in d.items()}
        for k, v in m.items():
            cols[k].append(v)
        ex = f" | {m['expl']:5.2f} {m['boxes']:5.2f}" if HAS_OR else ''
        print(f"{nm(ds):12s}{int(np.sum(d['n'])):5d} | {fmt(m, 'contain', 2):>22s} | {fmt(m, 'cond', 3):>28s} | {fmt(m, 'ok', 2):>22s} | "
              f"{fmt(m, 'cov', 3):>28s} | {fmt(m, 'act', 1):>18s} | {m['self']:.3f}{ex}")
    m = {k: np.nanmean(v) for k, v in cols.items()}
    ex = f" | {m['expl']:5.2f} {m['boxes']:5.2f}" if HAS_OR else ''
    print(f"{'mean':12s}{'':5s} | {fmt(m, 'contain', 2):>22s} | {fmt(m, 'cond', 3):>28s} | {fmt(m, 'ok', 2):>22s} | "
          f"{fmt(m, 'cov', 3):>28s} | {fmt(m, 'act', 1):>18s} | {m['self']:.3f}{ex}")
    tests = [('pi_contained', 'pi', 'cond', 'cond Fid: fixed vs stored π'), ('pi_contained', 'pi', 'cov', 'coverage: fixed vs stored π'),
             ('pi_contained', 'anchors', 'cond', 'cond Fid: fixed π vs Anchors'), ('pi_contained', 'anchors', 'ok', 'share ≥0.90: fixed π vs Anchors'),
             ('pi_contained', 'anchors', 'cov', 'coverage: fixed π vs Anchors')]
    if HAS_OR:
        tests += [('pi_or', 'pi_contained', 'cond', 'cond Fid: π∨ vs π+'), ('pi_or', 'pi_contained', 'cov', 'coverage: π∨ vs π+'),
                  ('pi_or', 'anchors', 'cond', 'cond Fid: π∨ vs Anchors'), ('pi_or', 'anchors', 'ok', 'share ≥0.90: π∨ vs Anchors'),
                  ('pi_or', 'anchors', 'cov', 'coverage: π∨ vs Anchors')]
    for a, b, k, lab in tests:
        x = cols[f'{a}|{k}']; y = cols[f'{b}|{k}']; w = paired_wilcoxon(x, y)
        print(f"  {lab}: A higher {(np.array(x) > np.array(y)).sum()}/{len(x)}, mean Δ {np.mean(np.array(x) - np.array(y)):+.3f}, p={w['pvalue']:.4f}")
    x = cols['self']; y = cols['anchors|cond']; print(f"  Anchors self-reported vs measured precision: mean {np.mean(x):.3f} vs {np.mean(y):.3f}")
