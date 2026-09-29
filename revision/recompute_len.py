"""Recompute rule length (Len) and the printed rules of stored cells under the new criterion.

A condition counts when it excludes at least one D_train row
(`utils.metrics.LEN_CRITERION`), decided in the space each box is scored in, and
the printed rule shows exactly those conditions (`utils.rule_print.RulePrinter`).
The old Len kept a feature whose interval was narrower than 95% of its range.
Len is a pure function of the stored boxes and D_train, so nothing is re-run and
no Fid/Cov/Eff number changes.

Rule sets: each selected rule's box is rebuilt with the float32 face repair of
`revision.rescore_boxes.rebuild` (the box that reproduces the stored D_val and
D_test row counts). With --apply every cell gains, in place:
  per_class.*.selected_rules[].n_conditions, display_rule (re-printed; the old
  string is kept as display_rule_stored), per_class.*.compactness.mean_conditions,
  and compactness.mean_conditions / len_criterion at the top level
- the same fields `revision.evaluate` and `revision.baselines` now write.

Instances (`containment_fix_exact_*`): every explanation's box gains `n_cond`
(pi_or: summed over its boxes), as `revision.containment_eval` now writes.

Trees: the surrogate trees are refit exactly as `revision.cart_global_surrogate`
fits them. Their Len was already the number of distinct features on the leaf's
path, which is this criterion (every split excludes training rows); refitting
gives the per-class average the paper uses and checks that identity.

    python -m revision.recompute_len --apply              # rule sets + instances
    python -m revision.recompute_len --trees              # surrogate trees
    python -m revision.recompute_len --report > ../results/len_recompute/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.cov_tau_class_gated import ORIGINAL_UNIT_METHODS, _loader_for  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import SeedData, Split, rebuild  # noqa: E402
from utils.metrics import LEN_CRITERION, active_feature_mask, condition_mask, train_span  # noqa: E402
from utils.rule_print import RulePrinter  # noqa: E402

RES = REPO.parent / "results"
OUT = RES / "len_recompute"
GRIDS = ("emp_tc0p10", "emp_tc0p20", "pert_tc0p10", "pert_tc0p20")
ALGO = {"rlda": "ddpg", "mada": "maddpg"}
# group -> folders of rule-set cells (methods inferred from the file names); the
# report reads the first five, the rest give every cell of those trees the same fields
GROUPS: Dict[str, List[Path]] = {
    "rl_perpolicy": [RES / "paper_final_perpolicy" / g / "results" / a for g in GRIDS for a in ALGO.values()],
    "rl_paper": [RES / "paper_final_valtb" / g / "results" / a for g in GRIDS for a in ALGO.values()],
    "anchors_pool20": [RES / "paper_final_anchorfix" / "pool20" / "k1"],
    "anchors_pool5": [RES / "paper_final_anchorfix" / "baselines_emp"],
    # CART and random search; this folder's Anchors cells are the superseded unfixed ones
    "cart_random": [RES / "paper_final" / "baselines_emp"],
    "rl_perpolicy_other": sorted({p.parent for p in (RES / "paper_final_perpolicy").rglob("*.json")
                                  if p.parts[len(RES.parts) + 1] in ("k_sweep", "ablations")}),
    "rl_nofloor": sorted({p.parent for p in (RES / "paper_final_perpolicy_nofloor").rglob("*.json")}),
    "anchors_other": sorted({p.parent for p in (RES / "paper_final_anchorfix").rglob("*_anchors__*.json")
                             if p.parent.name != "baselines_emp" and p.parent.parent.name != "logs"}
                            - {RES / "paper_final_anchorfix" / "pool20" / "k1"}),
}
GROUP_METHODS = {"cart_random": {"cart", "random_search"}}
INSTANCE_DIRS = [RES / "containment_fix_exact_emp", RES / "containment_fix_exact_pert"]
SURROGATE = RES / "paper_final_cart_fixed" / "global_surrogate"

_LOADERS: Dict[Any, Any] = {}
_PRINTERS: Dict[Any, RulePrinter] = {}


def loader(ds: str, seed: int):
    if (ds, seed) not in _LOADERS:
        _LOADERS.clear()   # one dataset at a time keeps memory flat (cells are sorted)
        _PRINTERS.clear()
        _LOADERS[(ds, seed)] = _loader_for(ds, seed)
    return _LOADERS[(ds, seed)]


def printer(ds: str, seed: int, space: str) -> RulePrinter:
    key = (ds, seed, space)
    if key not in _PRINTERS:
        _PRINTERS[key] = RulePrinter.for_loader(loader(ds, seed), space)
    return _PRINTERS[key]


def seed_data(ds: str, seed: int) -> SeedData:
    """Splits only: `rebuild` needs the rows, not f_hat."""
    L = loader(ds, seed)

    def split(unit, orig, y):
        return Split(np.asarray(unit, np.float32), np.asarray(orig, np.float32), np.asarray(y), None)

    return SeedData(ds, seed, L, split(L.X_val_unit, L.X_val, L.y_val),
                    split(L.X_test_unit, L.X_test, L.y_test), Path("."))


def old_count(lo, hi, method: str, L) -> int:
    """The 95%-width count exactly as the cell stored it."""
    if method in ORIGINAL_UNIT_METHODS:
        return int(active_feature_mask(lo, hi, 0.95, np.min(L.X_train, axis=0), np.max(L.X_train, axis=0)).sum())
    return int(active_feature_mask(lo, hi, 0.95).sum())


def process_cell(path: Path, apply: bool) -> Optional[Dict[str, Any]]:
    cell = json.loads(path.read_text())
    ds, seed, method = cell["dataset"], int(cell["seed"]), cell["method"]
    space = "original" if method in ORIGINAL_UNIT_METHODS else "unit"
    pr = printer(ds, seed, space)
    L = loader(ds, seed)
    rb = rebuild(cell, seed_data(ds, seed))
    # Cells written by the new code (spark s44-46) already carry Len: leave them as
    # they are and count where this recomputation disagrees with what they store.
    native = (cell.get("compactness") or {}).get("len_criterion") == LEN_CRITERION
    write = apply and not native
    per_class, old_stored_mismatch, native_mismatch = [], 0, 0
    for key, blk in (cell.get("per_class") or {}).items():
        if not isinstance(blk, dict):
            continue
        cls = (blk.get("union") or {}).get("target_class")
        cls = int(cls) if cls is not None else int(str(key).split("_")[-1])
        cr = rb.classes.get(cls)
        rules = [r for r in blk.get("selected_rules") or [] if r.get("lower_bounds") is not None]
        if cr is None or len(cr.boxes) != len(rules):
            continue
        comp = blk.get("compactness") or {}
        stored_old = [r.get("n_active_features") for r in comp.get("per_rule") or []]
        rows = []
        for i, (rule, (lo, hi)) in enumerate(zip(rules, cr.boxes)):
            n_new, n_old, shown = pr.count(lo, hi), old_count(lo, hi, method, L), pr(lo, hi)
            if i < len(stored_old) and stored_old[i] is not None and stored_old[i] != n_old:
                old_stored_mismatch += 1
            if native:
                native_mismatch += int(rule.get("n_conditions") != n_new or rule.get("display_rule") != shown)
            rows.append({"n_new": n_new, "n_old": n_old, "printed": shown,
                         "stored": rule.get("display_rule_stored", rule.get("display_rule"))})
            if write:
                rule.setdefault("display_rule_stored", rule.get("display_rule"))
                rule["display_rule"], rule["n_conditions"] = shown, n_new
                if i < len(comp.get("per_rule") or []):
                    comp["per_rule"][i]["n_conditions"] = n_new
        if write and rows:
            comp.update(mean_conditions=float(np.mean([r["n_new"] for r in rows])),
                        total_conditions=int(sum(r["n_new"] for r in rows)), len_criterion=LEN_CRITERION)
            blk["compactness"] = comp
            best = blk.get("best") or {}
            for rule, r in zip(rules, rows):
                if rule.get("rule_id") == best.get("rule_id"):
                    best.setdefault("display_rule_stored", best.get("display_rule"))
                    best["display_rule"] = r["printed"]
        per_class.append({"cls": cls, "rules": rows})
    if not per_class:
        return None
    len_new = float(np.mean([np.mean([r["n_new"] for r in c["rules"]]) for c in per_class]))
    len_old = float(np.mean([np.mean([r["n_old"] for r in c["rules"]]) for c in per_class]))
    if write:
        top = cell.get("compactness") or {}
        top.update(mean_conditions=len_new, len_criterion=LEN_CRITERION)
        cell["compactness"] = top
        path.write_text(json.dumps(cell, indent=2))
    return {"path": str(path.relative_to(RES)), "dataset": ds, "seed": seed, "method": method,
            "len_new": len_new, "len_old": len_old,
            "len_old_stored": (cell.get("compactness") or {}).get("mean_active_features"),
            "old_stored_mismatch": old_stored_mismatch, "rebuild_failures": len(rb.failures),
            "native": native, "native_mismatch": native_mismatch,
            "n_rules": sum(len(c["rules"]) for c in per_class), "n_classes": len(per_class),
            "per_class": per_class}


def cells_of(group: str) -> List[Path]:
    out = []
    for d in GROUPS[group]:
        for p in sorted(d.glob("*__seed*__tp*.json")):
            m = p.name.split("__")[1]
            if m in GROUP_METHODS.get(group, {m}):
                out.append(p)
    return out


def _ds_seed(p: Path):
    return p.name.split("__")[0], int(p.name.split("__seed")[1][:2])


def run_rule_sets(apply: bool, groups=None, datasets=None) -> None:
    """All groups in one pass ordered by (dataset, seed), so each split is loaded once."""
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(g, p) for g in (groups or GROUPS) for p in cells_of(g)
            if datasets is None or _ds_seed(p)[0] in datasets]
    todo.sort(key=lambda gp: (_ds_seed(gp[1]), gp[0], gp[1].name))
    results = defaultdict(list)
    for i, (g, p) in enumerate(todo):
        r = process_cell(p, apply)
        if r is not None:
            results[g].append(r)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(todo)} cells", flush=True)
    for g, rs in results.items():
        (OUT / f"{g}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        print(f"{g}: {len(rs)} cells, rebuild failures {sum(r['rebuild_failures'] for r in rs)}, "
              f"old count != stored on {sum(r['old_stored_mismatch'] for r in rs)} rules; "
              f"{sum(r['native'] for r in rs)} cells already had Len, recomputation differs on "
              f"{sum(r['native_mismatch'] for r in rs)} of their rules", flush=True)


def run_instances(apply: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for d in INSTANCE_DIRS:
        total_mism = 0
        for p in sorted(d.glob("*__seed*.json")):
            J = json.loads(p.read_text())
            ds, seed = J["dataset"], int(J["seed"])
            span = {sp: train_span(loader(ds, seed).X_train_unit if sp == "unit" else loader(ds, seed).X_train)
                    for sp in ("unit", "original")}
            agg = defaultdict(list)
            mism = 0
            for row in J["rows"]:
                for tag, sp in (("pi", "unit"), ("pi_contained", "unit"), ("anchors", "original")):
                    b = row.get(tag)
                    if b and b.get("lower") is not None:
                        n = int(condition_mask(b["lower"], b["upper"], span[sp]).sum())
                        mism += int("n_cond" in b and b["n_cond"] != n)
                        b["n_cond"] = n
                        agg[tag].append((n, b.get("n_active")))
                b = row.get("pi_or")
                if b and b.get("boxes"):
                    n = int(sum(condition_mask(x["lower"], x["upper"], span["unit"]).sum() for x in b["boxes"]))
                    mism += int("n_cond" in b and b["n_cond"] != n)
                    b["n_cond"] = n
                    agg["pi_or"].append((n, b.get("n_active")))
            rows_out.append({"path": str(p.relative_to(RES)), "arm": J["arm"], "dataset": ds, "seed": seed,
                             **{f"{t}|new": float(np.mean([a for a, _ in v])) for t, v in agg.items()},
                             **{f"{t}|old": float(np.mean([b for _, b in v])) for t, v in agg.items()}})
            total_mism += mism
            if apply:
                p.write_text(json.dumps(J, indent=2))
        print(f"{d.name}: done; stored n_cond differs from the recomputation on {total_mism} explanations",
              flush=True)
    (OUT / "instances.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_out))


def run_trees() -> None:
    """Refit the depth-3 and D_val-tuned surrogate trees and count their leaves' conditions."""
    from sklearn.tree import DecisionTreeClassifier

    from revision.cart_global_surrogate import leaf_conditions
    from revision.cov_tau_class_gated import _predictions
    from revision.rescore_boxes import load_seed_data
    from revision.cart_global_surrogate_pert import leaf_boxes

    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(SURROGATE.glob("*__seed*.json")):
        S = json.loads(p.read_text())
        ds, seed = S["dataset"], int(S["seed"])
        ref = json.loads((RES / "paper_final_cart_fixed/precision_constrained/baselines_emp"
                          / f"{ds}__cart__seed{seed}__tp0p90__tc0p10.json").read_text())
        sd = load_seed_data(ds, seed, Path(ref["extra"]["classifier_path"]))
        L = sd.loader
        X_tr = np.asarray(L.X_train, np.float32)
        y_hat_tr = _predictions(L, L.X_train_scaled)
        span = train_span(X_tr)
        rec = {"dataset": ds, "seed": seed}
        for name, depth in (("tree_d3", 3), ("tree_tuned", S["val_tuned_depth"])):
            tr = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X_tr, y_hat_tr)
            stored = S["trees"]["depth 3" if name == "tree_d3" else "val-tuned depth"]
            path_feats = leaf_conditions(tr)
            by_class = defaultdict(list)
            mismatch = 0
            for leaf, (lo, hi, c) in leaf_boxes(tr, X_tr.shape[1]).items():
                n = int(condition_mask(lo, hi, span).sum())
                mismatch += int(n != path_feats[leaf])
                by_class[int(c)].append(n)
            per_leaf = [n for v in by_class.values() for n in v]
            rec[name] = {"len_class_mean": float(np.mean([np.mean(v) for v in by_class.values()])),
                         "len_leaf_mean": float(np.mean(per_leaf)), "n_leaves": len(per_leaf),
                         "total_conditions": int(sum(per_leaf)),
                         "stored_cond_per_leaf": stored["cond_per_leaf"],
                         "stored_n_leaves": stored["n_leaves"], "path_mismatch": mismatch}
        out.append(rec)
        print(f"{ds} s{seed}: d3 {rec['tree_d3']['len_class_mean']:.2f} "
              f"tuned {rec['tree_tuned']['len_class_mean']:.2f}", flush=True)
    (OUT / "trees.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the new fields into the cells")
    ap.add_argument("--rule_sets", action="store_true")
    ap.add_argument("--instances", action="store_true")
    ap.add_argument("--trees", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--groups", nargs="*", choices=list(GROUPS))
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    if a.report:
        from revision.recompute_len_report import main as report
        return report()
    both = not (a.rule_sets or a.instances or a.trees)
    if a.rule_sets or both:
        run_rule_sets(a.apply, a.groups, a.datasets)
    if a.instances or both:
        run_instances(a.apply)
    if a.trees:
        run_trees()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
