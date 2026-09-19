#!/usr/bin/env python3
"""Independent re-derivation of every reported metric, plus a rule-pool audit.

Nothing here calls utils.metrics / utils.eval_harness: masks, Fid, Pur, Cov and
the global rule-set numbers are recomputed from the raw arrays with the formulas
written out longhand, then compared against what the artifact claims. The only
shared code is dataset/classifier loading.

Definitions used here (B = the box, c = target class, f_hat = black box):
  Fid   = |{x in B : f_hat(x) = c}| / |B|
  Pur   = |{x in B : y(x)   = c}| / |B|
  Cov   = |{x in B : y(x)   = c}| / |{x : y(x) = c}|      (class-conditional)
  CovM  = |B| / n_eval                                     (marginal)
  union = OR of the selected rules' masks, scored by the same formulas
  global: each class fires its union; 0 fired -> abstain, 2+ -> conflict (the
          class with the highest union Fid wins); Fid/Pur over DECIDED rows;
          Cov = 1 - abstention; Eff = Fid x Cov

  python revision/verify_metrics.py --results <result.json> [...]
  python revision/verify_metrics.py --sample        # a spread of methods
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))

from revision.evaluate import _load_classifier, make_tabular_loader  # noqa: E402

UNIT_METHODS = {"rlda", "mada", "random_search"}
TOL = 1e-6
_LOADERS: dict = {}


def get_loader(ds: str, seed: int, clf_path, fallback_file: str):
    key = (ds, int(seed), str(clf_path or fallback_file))
    if key not in _LOADERS:
        loader = make_tabular_loader(ds, random_state=int(seed))
        loader.load_dataset()
        loader.preprocess_data()
        if clf_path and Path(clf_path).exists():
            loader.classifier = loader.load_classifier(filepath=clf_path, device="cpu")
        else:
            loader.classifier = _load_classifier(loader, fallback_file, int(seed), ds)
        _LOADERS[key] = (loader, predictions(loader, loader.X_test_scaled),
                         predictions(loader, loader.X_val_scaled))
    return _LOADERS[key]


def predictions(loader, X_std: np.ndarray) -> np.ndarray:
    """argmax f_hat, computed here rather than trusted from the artifact."""
    import torch
    from utils.networks import predict_proba_torch

    clf = loader.classifier
    if hasattr(clf, "eval"):
        clf.eval()
    out = np.empty(X_std.shape[0], dtype=np.int64)
    with torch.no_grad():
        for s in range(0, X_std.shape[0], 20000):
            chunk = torch.from_numpy(np.asarray(X_std[s:s + 20000], dtype=np.float32))
            out[s:s + 20000] = predict_proba_torch(clf, chunk).cpu().numpy().argmax(axis=1)
    return out


def mask_of(X, lo, up):
    lo = np.asarray(lo, dtype=np.float32).reshape(-1)
    up = np.asarray(up, dtype=np.float32).reshape(-1)
    return np.all((X >= lo) & (X <= up), axis=1)


def metrics_of(mask, y, y_hat, cls, basis="true_label"):
    """basis: coverage denominator, y = c ("true_label") or f_hat = c ("predicted")."""
    n_cov = int(mask.sum())
    ref = y_hat if basis == "predicted" else y
    n_class = int((ref == cls).sum())
    return {
        "n_covered": n_cov,
        "n_covered_class": int((mask & (ref == cls)).sum()),
        "fidelity": float((y_hat[mask] == cls).mean()) if n_cov else float("nan"),
        "purity": float((y[mask] == cls).mean()) if n_cov else float("nan"),
        "coverage": float((mask & (ref == cls)).sum() / n_class) if n_class else float("nan"),
        "coverage_marginal": float(n_cov / mask.shape[0]),
    }


def cmp(tag, got, want, problems, tol=TOL):
    if want is None and (got is None or (isinstance(got, float) and math.isnan(got))):
        return
    if want is None or got is None:
        problems.append(f"{tag}: artifact={want} recomputed={got}")
        return
    if isinstance(got, float) and math.isnan(got):
        got = None
    if got is None or abs(float(got) - float(want)) > tol:
        problems.append(f"{tag}: artifact={want} recomputed={got}")


def verify(path: Path) -> tuple[list[str], dict]:
    r = json.loads(Path(path).read_text())
    ds, method, seed = r["dataset"], r["method"], int(r["seed"])
    space = "unit" if method in UNIT_METHODS else "original"
    extra = r.get("extra") or {}
    loader, y_hat, _ = get_loader(ds, seed, extra.get("classifier_path"),
                                  extra.get("rules_file") or str(path))
    y = np.asarray(loader.y_test)
    basis = extra.get("coverage_basis", "true_label")
    X = np.asarray(loader.X_test_unit if space == "unit" else loader.X_test, dtype=np.float32)
    problems: list[str] = []

    # classifier accuracy the artifact reports, recomputed
    acc = (r.get("classifier_accuracy") or {}).get("test_accuracy")
    cmp("classifier test_accuracy", float((y_hat == y).mean()), acc, problems, tol=1e-9)

    union_masks, union_fid = {}, {}
    for key, blk in sorted((r.get("per_class") or {}).items()):
        cls = int(key.split("_")[1])
        rules = blk.get("selected_rules") or []
        if not rules:
            problems.append(f"{key}: no selected_rules stored")
            continue
        masks = []
        for rule in rules:
            lo, up = rule.get("lower_bounds"), rule.get("upper_bounds")
            if lo is None or up is None:
                problems.append(f"{key} {rule.get('rule_id')}: no bounds stored")
                continue
            m = mask_of(X, lo, up)
            masks.append(m)
            rep = rule.get("report_metrics") or {}
            mine = metrics_of(m, y, y_hat, cls, basis)
            for f in ("n_covered", "n_covered_class", "fidelity", "purity",
                      "coverage", "coverage_marginal"):
                cmp(f"{key} rule {rule.get('rule_id')} {f}", mine[f], rep.get(f), problems)
        if not masks:
            continue
        umask = np.zeros(X.shape[0], dtype=bool)
        for m in masks:
            umask |= m
        mine = metrics_of(umask, y, y_hat, cls, basis)
        for f in ("n_covered", "n_covered_class", "fidelity", "purity",
                  "coverage", "coverage_marginal"):
            cmp(f"{key} UNION {f}", mine[f], (blk.get("union") or {}).get(f), problems)
        # union must dominate every member on coverage and on covered rows
        for i, m in enumerate(masks):
            mm = metrics_of(m, y, y_hat, cls, basis)
            if mine["n_covered"] + 1e-9 < mm["n_covered"]:
                problems.append(f"{key}: union covers fewer rows than member {i}")
            if mine["coverage"] + 1e-9 < mm["coverage"]:
                problems.append(f"{key}: union coverage < member {i} coverage")
        union_masks[cls] = umask
        union_fid[cls] = mine["fidelity"] if math.isfinite(mine["fidelity"]) else -np.inf

    # ---- global rule set ---------------------------------------------------
    g = r.get("global_ruleset") or {}
    if union_masks:
        classes = sorted(union_masks)
        stacked = np.stack([union_masks[c] for c in classes], axis=1)
        n_fired = stacked.sum(axis=1)
        abstain, conflict = n_fired == 0, n_fired >= 2
        pred = np.full(X.shape[0], -1, dtype=int)
        single = n_fired == 1
        if single.any():
            pred[single] = np.array(classes)[stacked[single].argmax(axis=1)]
        if conflict.any():
            fid_vec = np.array([union_fid[c] for c in classes], dtype=np.float64)
            pick = np.where(stacked[conflict], fid_vec[None, :], -np.inf).argmax(axis=1)
            pred[conflict] = np.array(classes)[pick]
        decided = ~abstain
        n_dec = int(decided.sum())
        gfid = float((pred[decided] == y_hat[decided]).mean()) if n_dec else float("nan")
        gpur = float((pred[decided] == y[decided]).mean()) if n_dec else float("nan")
        cmp("global n_decided", float(n_dec), g.get("n_decided"), problems, tol=0.5)
        cmp("global abstention_rate", float(abstain.mean()), g.get("abstention_rate"), problems)
        cmp("global conflict_rate", float(conflict.mean()), g.get("conflict_rate"), problems)
        cmp("global coverage", float(1.0 - abstain.mean()), g.get("coverage"), problems)
        cmp("global_fidelity", gfid, g.get("global_fidelity"), problems)
        cmp("global_purity", gpur, g.get("global_purity"), problems)
        eff = g.get("effectiveness")
        if eff is not None and math.isfinite(gfid):
            cmp("effectiveness", gfid * (1.0 - float(abstain.mean())), eff, problems)
    return problems, {"dataset": ds, "method": method, "seed": seed, "space": space}


# ---------------------------------------------------------------------------
# Pool audit: how many generated rules reach selection, and why the rest do not
# ---------------------------------------------------------------------------

def audit_pool(result_path: Path) -> dict:
    r = json.loads(Path(result_path).read_text())
    rf = (r.get("extra") or {}).get("rules_file")
    if not rf or not Path(rf).exists():
        return {}
    rules = json.loads(Path(rf).read_text())
    pc = rules.get("per_class_results", rules)
    ds, seed, method = r["dataset"], int(r["seed"]), r["method"]
    loader, _, yhv = get_loader(ds, seed, (r.get("extra") or {}).get("classifier_path"), rf)
    Xv = np.asarray(loader.X_val_unit, dtype=np.float32)
    yv = np.asarray(loader.y_val)
    min_support = int(r.get("min_support") or 10)

    out = {}
    for key, blk in sorted(pc.items()):
        if not isinstance(blk, dict) or blk.get("class") is None:
            continue
        cls = int(blk["class"])
        raw = []
        for k in ("all_anchors", "anchors"):
            raw.extend(blk.get(k) or [])
        cb = blk.get("class_based_results") or {}
        for v in (cb.values() if isinstance(cb, dict) else []):
            if isinstance(v, dict):
                raw.extend(v.get("anchors") or v.get("all_anchors") or [])
        stats = defaultdict(int)
        stats["raw_entries"] = len(raw)
        seen, kept = set(), []
        for a in raw:
            lo = a.get("lower_bounds_normalized")
            up = a.get("upper_bounds_normalized")
            if lo is None or up is None:
                stats["no_unit_bounds"] += 1
                continue
            lo = np.asarray(lo, dtype=float)
            up = np.asarray(up, dtype=float)
            if not np.any((up - lo) < 0.95):          # empty rule: nothing tightened
                stats["empty_rule"] += 1
                continue
            keyb = (tuple(np.round(lo, 6)), tuple(np.round(up, 6)))
            if keyb in seen:
                stats["duplicate_box"] += 1
                continue
            seen.add(keyb)
            kept.append((lo, up))
        stats["unique_scorable"] = len(kept)
        for lo, up in kept:
            m = mask_of(Xv, lo, up)
            n = int(m.sum())
            if n == 0:
                stats["val_covers_nothing"] += 1
            elif n < min_support:
                stats["below_min_support"] += 1
            else:
                stats["eligible"] += 1
        blk_out = (r.get("per_class") or {}).get(f"class_{cls}") or {}
        stats["selected"] = int(blk_out.get("n_selected") or 0)
        stats["k"] = int(blk_out.get("k") or 0)
        out[f"class_{cls}"] = dict(stats)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=None)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--pool", action="store_true", help="also audit the candidate pools")
    ap.add_argument("--all", action="store_true", help="every artifact in the final comparison")
    args = ap.parse_args()

    paths = [Path(p) for p in (args.results or [])]
    if args.all:
        sys.path.insert(0, str(REPO / "revision"))
        import final_comparison as FC
        paths = []
        for ds in FC.DATASETS:
            for _label, root, method, _csv in FC.METHODS:
                f = Path(root(ds)) / f"{ds}__{method}__seed{FC.SEED}__tp0p90__tc0p10.json"
                if f.exists():
                    paths.append(f)
    if args.sample or not paths:
        P = MAIN / "runs" / "paper_fiveseed_overlap075" / "results"
        R = REPO / "runs"
        paths = [
            P / "ddpg" / "iris__rlda__seed42__tp0p90__tc0p10.json",
            P / "baselines" / "iris__cart__seed42__tp0p90__tc0p10.json",
            P / "baselines" / "uci_credit__greedy_anchors__seed42__tp0p90__tc0p10.json",
            P / "baselines" / "housing__random_search__seed42__tp0p90__tc0p10.json",
            R / "perturb_fid_seed42/results/ddpg/heloc__rlda__seed42__tp0p90__tc0p10.json",
            R / "perturb_fid_seed42/results/maddpg/sick__mada__seed42__tp0p90__tc0p10.json",
            R / "paper_mada_perclass_seed42/results/maddpg/wyodot_kvdw_labeled__mada__seed42__tp0p90__tc0p10.json",
            R / "k_sweep_variants/k5/RLDA-pert/synthetic__rlda__seed42__tp0p90__tc0p10.json",
        ]
    total = 0
    for p in paths:
        if not p.exists():
            print(f"MISSING {p}")
            continue
        problems, info = verify(p)
        total += len(problems)
        tag = f"{info['dataset']}/{info['method']}/seed{info['seed']} [{info['space']}]"
        print(f"{'OK  ' if not problems else 'FAIL'} {tag}  ({p.name})")
        for msg in problems[:12]:
            print(f"      {msg}")
        if args.pool:
            for cls, st in (audit_pool(p) or {}).items():
                print(f"      pool {cls}: " + " ".join(f"{k}={v}" for k, v in st.items()))
    print(f"\n{total} mismatch(es)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
