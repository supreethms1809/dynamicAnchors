#!/usr/bin/env python3
"""Write docs/RESULTS_paper_final_predicted.md and the old-lock comparison.

  python revision/write_paper_final_tables.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
sys.path.insert(0, str(REPO))

from revision.paper_stats import (  # noqa: E402
    BOOT_SEED,
    DATASETS,
    METHODS,
    N_BOOT,
    SEEDS,
    _finite,
    boot_ci_mean,
    dataset_means,
    headline_seed_means,
    holm,
    vec,
    wilcoxon_rows,
)
from utils.metrics import track_a_cov_tau, track_a_eff  # noqa: E402

PF_ROOT = REPO / "runs" / "paper_final"
PF = PF_ROOT / "results" / "predicted"
OLD_PAPER = MAIN / "runs" / "paper_fiveseed_overlap075" / "results"
OLD_WYODOT = MAIN / "runs" / "wyodot_fiveseed_overlap075" / "dnn" / "results"

# Live seed-major trees. Collector copies into PF/<label>/ as a fallback.
_LIVE_RL = {
    ("rlda", "emp", "0p10"): PF_ROOT / "emp_tc0p10" / "results" / "ddpg",
    ("mada", "emp", "0p10"): PF_ROOT / "emp_tc0p10" / "results" / "maddpg",
    ("rlda", "emp", "0p20"): PF_ROOT / "emp_tc0p20" / "results" / "ddpg",
    ("mada", "emp", "0p20"): PF_ROOT / "emp_tc0p20" / "results" / "maddpg",
    ("rlda", "pert", "0p10"): PF_ROOT / "pert_tc0p10" / "results" / "ddpg",
    ("mada", "pert", "0p10"): PF_ROOT / "pert_tc0p10" / "results" / "maddpg",
    ("rlda", "pert", "0p20"): PF_ROOT / "pert_tc0p20" / "results" / "ddpg",
    ("mada", "pert", "0p20"): PF_ROOT / "pert_tc0p20" / "results" / "maddpg",
}
_LIVE_BASE = {
    "emp": PF_ROOT / "baselines_emp",
    "pert": PF_ROOT / "baselines_pert",
}
_COPY_FOLDER = {
    ("rlda", "emp", "0p10"): "RLDA-emp",
    ("mada", "emp", "0p10"): "MADA-emp",
    ("rlda", "emp", "0p20"): "RLDA-emp-tc020",
    ("mada", "emp", "0p20"): "MADA-emp-tc020",
    ("rlda", "pert", "0p10"): "RLDA-pert",
    ("mada", "pert", "0p10"): "MADA-pert",
    ("rlda", "pert", "0p20"): "RLDA-pert-tc020",
    ("mada", "pert", "0p20"): "MADA-pert-tc020",
    ("cart", "emp", "0p10"): "CART-emp",
    ("random_search", "emp", "0p10"): "RandS-emp",
    ("sp_anchors", "emp", "0p10"): "SP-Anch",
    ("greedy_anchors", "emp", "0p10"): "GreedyAnch",
    ("cart", "pert", "0p10"): "CART-pert",
    ("random_search", "pert", "0p10"): "RandS-pert",
    ("cart", "emp", "0p20"): "CART-emp-tc020",
    ("random_search", "emp", "0p20"): "RandS-emp-tc020",
    ("cart", "pert", "0p20"): "CART-pert-tc020",
    ("random_search", "pert", "0p20"): "RandS-pert-tc020",
}
OUT1 = REPO / "docs" / "RESULTS_paper_final_predicted.md"
OUT2 = REPO / "docs" / "RESULTS_old_lock_vs_paper_final.md"

LABEL = {
    "rlda": "RLDA-emp",
    "mada": "MADA-emp",
    "cart": "CART-emp",
    "random_search": "RandS-emp",
    "sp_anchors": "SP-Anch*",
    "greedy_anchors": "GreedyAnch*",
}
FOLDER = {
    "rlda": "RLDA-emp",
    "mada": "MADA-emp",
    "cart": "CART-emp",
    "random_search": "RandS-emp",
    "sp_anchors": "SP-Anch",
    "greedy_anchors": "GreedyAnch",
}
DISP = {
    "rlda": "RLDA",
    "mada": "MADA",
    "cart": "CART",
    "random_search": "random search",
    "sp_anchors": "sp-anchors",
    "greedy_anchors": "greedy-anchors",
}
PERT_FOLDERS = {
    "rlda": "RLDA-pert",
    "mada": "MADA-pert",
    "cart": "CART-pert",
    "random_search": "RandS-pert",
}


def _gr(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()).get("global_ruleset") or {}


def locate(method: str, ds: str, seed: int, tau: str = "0p10",
           estimator: str = "emp") -> Optional[Path]:
    """Prefer the live 2×2 cell; fall back to a collector copy under results/predicted/."""
    name = f"{ds}__{method}__seed{seed}__tp0p90__tc{tau}.json"
    if method in ("rlda", "mada"):
        live_dir = _LIVE_RL.get((method, estimator, tau))
        if live_dir is not None:
            p = live_dir / name
            if p.is_file():
                return p
    else:
        live_dir = _LIVE_BASE.get(estimator)
        if live_dir is not None:
            p = live_dir / name
            if p.is_file():
                return p
    folder = _COPY_FOLDER.get((method, estimator, tau)) or FOLDER.get(method)
    if folder:
        p = PF / folder / name
        if p.is_file():
            return p
    return None


def _record(path: Path) -> Dict[str, Optional[float]]:
    blob = json.loads(path.read_text())
    gr = blob.get("global_ruleset") or {}
    extra = blob.get("extra") or {}
    compact = blob.get("compactness") or {}
    rec: Dict[str, Optional[float]] = {
        "fid": _finite(gr.get("global_fidelity")),
        "cov": _finite(gr.get("coverage")),
        "conf": _finite(gr.get("conflict_rate")),
        "pur": _finite(gr.get("global_purity")),
        "eff": track_a_eff(gr),
        "cov_tau": track_a_cov_tau(gr),
        "n_active": _finite(compact.get("mean_active_features")),
        "basis": extra.get("coverage_basis"),
    }
    unions = []
    for blk in (blob.get("per_class") or {}).values():
        u = (blk or {}).get("union") or {}
        unions.append({
            "fid": _finite(u.get("fidelity")),
            "cov": _finite(u.get("coverage")),
        })
    rec["n_classes"] = float(len(unions))
    rec["_unions"] = unions  # type: ignore[assignment]
    rec["_blob"] = blob  # type: ignore[assignment]
    return rec


def ingest_paper_final() -> Dict[Tuple[str, str, int], Dict[str, Optional[float]]]:
    """Headline ingest: empirical Fid, τ_C=0.10, C=predicted (emp_tc0p10 + baselines_emp)."""
    out: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
    for method in FOLDER:
        est = "emp"
        for ds in DATASETS:
            for seed in SEEDS:
                p = locate(method, ds, seed, tau="0p10", estimator=est)
                if p is None:
                    continue
                out[(ds, method, seed)] = _record(p)
    return out


def ingest_old() -> Dict[Tuple[str, str, int], Dict[str, Optional[float]]]:
    from revision.paper_stats import _ingest
    out: Dict[Tuple[str, str, int], Dict[str, Optional[float]]] = {}
    _ingest(OLD_PAPER, out)
    _ingest(OLD_WYODOT, out)
    return out


def f3(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def msd_cell(mu: Optional[float], sd: Optional[float]) -> str:
    if mu is None:
        return "—"
    if sd is None:
        return f"{mu:.3f}"
    return f"{mu:.3f} ± {sd:.3f}"


def fmt_ci(mu: float, lo: Optional[float], hi: Optional[float]) -> str:
    if lo is None or hi is None:
        return f"{mu:.3f}"
    return f"{mu:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_w(r: Dict[str, Any], adjusted: bool) -> str:
    ps = f"{r['p']:.4f}" if r["p"] is not None else "—"
    rs = f"{r['r']:+.3f}" if r["r"] is not None else "—"
    mds = f"{r['mean_diff']:+.3f}" if r["mean_diff"] is not None else "—"
    ci = (
        f"[{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}]"
        if r["diff_ci_low"] is not None else "—"
    )
    if adjusted:
        pa = f"{r['p_holm']:.4f}" if r["p_holm"] is not None else "—"
        return (f"| `{r['a']}` vs `{r['b']}` | {r['metric']} | {r['n']} | {ps} | "
                f"{pa} | {rs} | {mds} | {ci} | {r['sig_holm']} |")
    return (f"| `{r['a']}` vs `{r['b']}` | {r['metric']} | {r['n']} | {ps} | "
            f"{rs} | {mds} | {ci} | {r['sig']} |")


def seed42_pert_rows() -> List[str]:
    L = [
        "",
        "### Seed 42 only — perturbed Fid objective",
        "",
        "P1 has not finished. These four methods exist for seed 42 under the "
        "predicted basis. Do not average them with the 5-seed empirical table.",
        "",
        "| method | Fid | Cov | Conf | Eff |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in PERT_FOLDERS:
        fids, covs, confs, effs = [], [], [], []
        n = 0
        for ds in DATASETS:
            p = locate(method, ds, 42, tau="0p10", estimator="pert")
            if p is None:
                continue
            gr = _gr(p)
            n += 1
            if _finite(gr.get("global_fidelity")) is not None:
                fids.append(float(gr["global_fidelity"]))
            if _finite(gr.get("coverage")) is not None:
                covs.append(float(gr["coverage"]))
            if _finite(gr.get("conflict_rate")) is not None:
                confs.append(float(gr["conflict_rate"]))
            e = track_a_eff(gr)
            if e is not None:
                effs.append(e)
        if n == 0:
            continue
        tag = {"rlda": "RLDA-pert", "mada": "MADA-pert", "cart": "CART-pert",
               "random_search": "RandS-pert"}[method]
        L.append(
            f"| `{tag}` "
            f"| {st.mean(fids) if fids else float('nan'):.3f} "
            f"| {st.mean(covs) if covs else float('nan'):.3f} "
            f"| {st.mean(confs) if confs else float('nan'):.3f} "
            f"| {st.mean(effs) if effs else float('nan'):.3f} |"
        )
    L.append("")
    L.append(f"Mean over {len(DATASETS)} datasets, seed 42. Not a 5-seed result.")
    return L


def per_class_table(cells) -> List[str]:
    L = [
        "",
        "### Per-class union (predicted basis, mean over 5 seeds)",
        "",
        r"Class Cov here is \(P(x \in B \mid \hat f(x)=c)\) on \(D_{test}\), "
        "the selection/reporting basis. Global Cov in the tables above is "
        r"\(1-\)abstention and is a different object.",
        "",
        "| dataset | class | RLDA Fid / Cov | MADA Fid / Cov | CART Fid / Cov |",
        "|---|---:|---|---|---|",
    ]
    for ds in DATASETS:
        by_cls: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for method in ("rlda", "mada", "cart"):
            for seed in SEEDS:
                rec = cells.get((ds, method, seed)) or {}
                for i, u in enumerate(rec.get("_unions") or []):
                    fid, cov = u.get("fid"), u.get("cov")
                    if fid is None and cov is None:
                        continue
                    by_cls[str(i)][method].append(
                        (float(fid) if fid is not None else float("nan"),
                         float(cov) if cov is not None else float("nan"))
                    )
        for cls in sorted(by_cls, key=lambda x: int(x)):
            cells_m = []
            for method in ("rlda", "mada", "cart"):
                pairs = by_cls[cls].get(method) or []
                if not pairs:
                    cells_m.append("—")
                    continue
                fids = [p[0] for p in pairs if p[0] == p[0]]
                covs = [p[1] for p in pairs if p[1] == p[1]]
                cells_m.append(
                    f"{st.mean(fids):.3f} / {st.mean(covs):.3f}" if fids and covs else "—"
                )
            L.append(f"| `{ds}` | {cls} | {cells_m[0]} | {cells_m[1]} | {cells_m[2]} |")
    return L


def write_new_tables(cells) -> None:
    n_cells = sum(1 for method in METHODS for ds in DATASETS for seed in SEEDS
                  if (ds, method, seed) in cells)
    n_expect = len(METHODS) * len(DATASETS) * len(SEEDS)
    means = dataset_means(cells)
    rows = wilcoxon_rows(means)
    head = headline_seed_means(cells)
    bases = {(cells[(ds, m, s)] or {}).get("basis")
             for m in METHODS for ds in DATASETS for s in SEEDS
             if (ds, m, s) in cells}
    empty_rs = [
        (ds, seed)
        for ds in DATASETS
        for seed in SEEDS
        if (cells.get((ds, "random_search", seed)) or {}).get("fid") is None
        and (cells.get((ds, "random_search", seed)) or {}).get("cov") == 0.0
    ]

    L = [
        "# Paper-final results (predicted coverage basis)",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from "
        f"`runs/paper_final/results/predicted/` by "
        f"`revision/write_paper_final_tables.py`.",
        "",
        "This is the **fixed protocol**, not the manuscript lock in "
        "`docs/RESULTS_comparison.md`. Compare the two in "
        "`docs/RESULTS_old_lock_vs_paper_final.md`.",
        "",
        "## Protocol",
        "",
        "| | |",
        "|---|---|",
        "| Source | `runs/paper_final/results/predicted/` |",
        "| Seeds | 42–46 |",
        "| Datasets | 12 (11-set + `wyodot_kvdw_labeled` DNN) |",
        "| k / τ_P / τ_C | 1 / 0.90 / 0.10 |",
        "| Coverage basis | **predicted**: class Cov = "
        r"\(P(x\in B\mid \hat f(x)=c)\). "
        "Used for D_val ranking and D_test class tables. "
        "Headline Fid/Cov/Eff below are still **global** "
        "(Fid among decided, Cov = 1−abstention, Eff = Fid×Cov). |",
        "| RLDA | top-K cap 5 after dedupe+NMS (`rlda_emp_capped_seed*`) |",
        "| MADA | per-class checkpoint, `final_live_obs_rows` |",
        "| Baselines | empirical Fid; `budget_per_class=5` |",
        f"| Files | **{n_cells}/{n_expect}** empirical cells |",
        f"| `extra.coverage_basis` | {', '.join(sorted(str(b) for b in bases))} |",
        "",
        "Perturbed-Fid methods are seed 42 only (P1 still running) and sit in "
        "their own subsection. Do not mix them into the 5-seed headline.",
        "",
        "## Headline (5 seed-means)",
        "",
        "Each seed is the mean of 12 datasets. ± is seed-to-seed sd.",
        "",
        "| method | Fid | Cov | Cov_τ | Conf | Eff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        h = head[method]
        L.append(
            f"| `{LABEL[method]}` | {msd_cell(h['fid'], h['fid_sd'])} | "
            f"{msd_cell(h['cov'], h['cov_sd'])} | "
            f"{msd_cell(h['cov_tau'], h['cov_tau_sd'])} | "
            f"{msd_cell(h['conf'], h['conf_sd'])} | "
            f"{msd_cell(h['eff'], h['eff_sd'])} |"
        )

    L += [
        "",
        f"Mean over the **12 datasets** (Wilcoxon unit) with a percentile "
        f"bootstrap 95% CI, {N_BOOT:,} resamples (seed {BOOT_SEED}).",
        "",
        "| method | Fid [95% CI] | Cov [95% CI] | Cov_τ [95% CI] | Conf [95% CI] | Eff [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells_m = []
        for met in ("fid", "cov", "cov_tau", "conf", "eff"):
            xs = vec(means, method, met)
            mu = sum(xs) / len(xs)
            lo, hi = boot_ci_mean(xs)
            cells_m.append(fmt_ci(mu, lo, hi))
        L.append(f"| `{LABEL[method]}` | " + " | ".join(cells_m) + " |")

    L += [
        "",
        "Share of the **60** dataset×seed cells with Fid ≥ 0.90.",
        "",
        "| method | cells Fid ≥ 0.90 |",
        "|---|---:|",
    ]
    for method in METHODS:
        n_clear = 0
        for ds in DATASETS:
            for seed in SEEDS:
                fid = (cells.get((ds, method, seed)) or {}).get("fid")
                if fid is not None and float(fid) + 1e-12 >= 0.90:
                    n_clear += 1
        L.append(f"| `{LABEL[method]}` | {n_clear}/60 |")

    empty = ", ".join(f"{ds} seed{s}" for ds, s in empty_rs) if empty_rs else "none"
    L += [
        "",
        f"Empty random-search cells (Eff and Cov_τ set to 0): {empty}.",
        "",
        "## Paired Wilcoxon (dataset seed-means, n=12)",
        "",
        "Unit = one number per dataset (mean of seeds 42–46). Two-sided Wilcoxon. "
        "Rank-biserial is Kerby (2014). Holm is within the 9 Eff contrasts only.",
        "",
        "**Confirmatory family (Eff)**",
        "",
        "| A vs B | metric | n | p | p_holm | Kerby r | mean Δ (A−B) | Δ 95% CI | sig (Holm) |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        if r["family"] == "primary":
            L.append(_fmt_w(r, True))
    L += [
        "",
        "**Cov_τ (unadjusted)**",
        "",
        "| A vs B | metric | n | p | Kerby r | mean Δ (A−B) | Δ 95% CI | sig (raw) |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        if r["family"] == "constrained":
            L.append(_fmt_w(r, False))
    L += [
        "",
        "**Exploratory (unadjusted)**",
        "",
        "| A vs B | metric | n | p | Kerby r | mean Δ (A−B) | Δ 95% CI | sig (raw) |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        if r["family"] == "secondary":
            L.append(_fmt_w(r, False))

    survivors = [
        f"`{r['a']}` vs `{r['b']}`"
        for r in rows
        if r["family"] == "primary" and r["p_holm"] is not None and r["p_holm"] < 0.05
    ]
    L += [
        "",
        "Surviving Holm at α=0.05 on Eff: "
        + (", ".join(survivors) if survivors else "none")
        + ".",
        "",
        "## Per-dataset (5-seed means, global metrics)",
        "",
        "| dataset | RLDA Fid/Cov/Eff | MADA Fid/Cov/Eff | CART Fid/Cov/Eff | SP-Anch Fid/Cov/Eff | Greedy Fid/Cov/Eff | RandS Fid/Cov/Eff |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        def cell(m):
            rec = means[ds][m]
            return f"{rec['fid']:.3f}/{rec['cov']:.3f}/**{rec['eff']:.3f}**"
        L.append(
            f"| `{ds}` | {cell('rlda')} | {cell('mada')} | {cell('cart')} | "
            f"{cell('sp_anchors')} | {cell('greedy_anchors')} | {cell('random_search')} |"
        )

    L += [
        "",
        "Conflict rate (5-seed mean).",
        "",
        "| dataset | RLDA | MADA | CART | SP-Anch | Greedy | RandS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        confs = " | ".join(f"{means[ds][m]['conf']:.3f}" for m in METHODS)
        L.append(f"| `{ds}` | {confs} |")

    # compactness
    L += [
        "",
        "### Compactness (mean active features, 5-seed × 12-dataset mean)",
        "",
        "| method | mean active features |",
        "|---|---:|",
    ]
    for method in METHODS:
        xs = []
        for ds in DATASETS:
            for seed in SEEDS:
                v = (cells.get((ds, method, seed)) or {}).get("n_active")
                if v is not None:
                    xs.append(float(v))
        L.append(f"| `{LABEL[method]}` | {st.mean(xs):.2f} |" if xs else f"| `{LABEL[method]}` | — |")

    L += per_class_table(cells)
    L += seed42_pert_rows()
    L += [
        "",
        "## Not in this file",
        "",
        "- k-sweep and pool-20 under the new protocol (P1 k-sweep is seed 42; "
        "5-seed matched pool is not queued).",
        "- 5-seed perturbed Fid (P1 in flight).",
        "- Ablations (wait for P1 COMPLETE).",
        "- Cost / query tables (instrumentation unchanged; perturbed training "
        "cost is not yet regenerated).",
        "",
    ]
    OUT1.write_text("\n".join(L) + "\n")


def write_comparison(new_cells, old_cells) -> None:
    new_means = dataset_means(new_cells)
    old_means = dataset_means(old_cells)
    new_head = headline_seed_means(new_cells)
    old_head = headline_seed_means(old_cells)
    new_w = {(r["a"], r["b"], r["met"]): r for r in wilcoxon_rows(new_means)}
    old_w = {(r["a"], r["b"], r["met"]): r for r in wilcoxon_rows(old_means)}

    def dlt(a: Optional[float], b: Optional[float]) -> str:
        if a is None or b is None:
            return "—"
        return f"{a - b:+.3f}"

    L = [
        "# Old paper lock vs paper-final (predicted basis)",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "| | Old lock (`docs/RESULTS_comparison.md`) | New (`runs/paper_final/results/predicted/`) |",
        "|---|---|---|",
        "| MADA checkpoint | global; scored from dead `obs[-1][-1]` | per-class; `final_live_obs_rows` |",
        "| RLDA pool | uncapped (~15 boxes/class, seed 42) | top-K 5 after NMS |",
        r"| Class Cov used to **select** on D_val | true-label \(P(x\in B\mid y=c)\) | predicted \(P(x\in B\mid \hat f=c)\) |",
        "| Headline Fid / Cov / Eff | global (same definition) | global (same definition) |",
        "| k / τ / seeds / datasets | 1 / 0.90/0.10 / 42–46 / 12 | same |",
        "| Anchor pool | `budget_per_class=5` | same |",
        "",
        "Global Fid, Cov, Eff, Conf are the **same estimands**. Numbers move "
        "because a different rule is selected (predicted-basis ranking, capped "
        "RLDA pool, fixed MADA checkpoint), not because the metric changed.",
        "",
        "Do not mix k=5 / pool-20 cells from the old lock with these k=1 new cells.",
        "",
        "## Headline (5 seed-means)",
        "",
        "Δ = new − old. Positive Eff means the fixed protocol covers more of the "
        "test set correctly.",
        "",
        "| method | old Fid | new Fid | Δ Fid | old Cov | new Cov | Δ Cov | old Eff | new Eff | Δ Eff | old Conf | new Conf | Δ Conf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        o, n = old_head[method], new_head[method]
        L.append(
            f"| `{DISP[method]}` | {f3(o['fid'])} | {f3(n['fid'])} | {dlt(n['fid'], o['fid'])} | "
            f"{f3(o['cov'])} | {f3(n['cov'])} | {dlt(n['cov'], o['cov'])} | "
            f"{f3(o['eff'])} | {f3(n['eff'])} | {dlt(n['eff'], o['eff'])} | "
            f"{f3(o['conf'])} | {f3(n['conf'])} | {dlt(n['conf'], o['conf'])} |"
        )

    L += [
        "",
        "## Wilcoxon Family 1 (Eff, n=12, Holm)",
        "",
        "| contrast | old Δ | new Δ | old p | new p | old p_holm | new p_holm | old sig | new sig |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for a, b, met, _lab in [
        ("mada", "rlda", "eff", "Eff"),
        ("rlda", "cart", "eff", "Eff"),
        ("mada", "cart", "eff", "Eff"),
        ("rlda", "sp_anchors", "eff", "Eff"),
        ("mada", "sp_anchors", "eff", "Eff"),
        ("rlda", "greedy_anchors", "eff", "Eff"),
        ("mada", "greedy_anchors", "eff", "Eff"),
        ("rlda", "random_search", "eff", "Eff"),
        ("mada", "random_search", "eff", "Eff"),
    ]:
        o, n = old_w[(a, b, met)], new_w[(a, b, met)]
        L.append(
            f"| `{a}` vs `{b}` | {o['mean_diff']:+.3f} | {n['mean_diff']:+.3f} | "
            f"{o['p']:.4f} | {n['p']:.4f} | "
            f"{o['p_holm']:.4f} | {n['p_holm']:.4f} | "
            f"{o['sig_holm']} | {n['sig_holm']} |"
        )

    gained, lost, held = [], [], []
    for a, b, met, lab in [
        ("mada", "rlda", "eff", "MADA vs RLDA"),
        ("rlda", "cart", "eff", "RLDA vs CART"),
        ("mada", "cart", "eff", "MADA vs CART"),
        ("rlda", "sp_anchors", "eff", "RLDA vs sp-anchors"),
        ("mada", "sp_anchors", "eff", "MADA vs sp-anchors"),
        ("rlda", "greedy_anchors", "eff", "RLDA vs greedy"),
        ("mada", "greedy_anchors", "eff", "MADA vs greedy"),
        ("rlda", "random_search", "eff", "RLDA vs random"),
        ("mada", "random_search", "eff", "MADA vs random"),
    ]:
        o_sig = old_w[(a, b, met)]["p_holm"] < 0.05
        n_sig = new_w[(a, b, met)]["p_holm"] < 0.05
        if o_sig and n_sig:
            held.append(lab)
        elif (not o_sig) and n_sig:
            gained.append(lab)
        elif o_sig and not n_sig:
            lost.append(lab)
    L += [
        "",
        f"- Holm significance **held**: {', '.join(held) if held else 'none'}.",
        f"- **Gained** (ns → sig): {', '.join(gained) if gained else 'none'}.",
        f"- **Lost** (sig → ns): {', '.join(lost) if lost else 'none'}.",
        "",
        "## Per-dataset Eff (5-seed means)",
        "",
        "| dataset | old RLDA | new RLDA | Δ | old MADA | new MADA | Δ | old CART | new CART | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        o_r, n_r = old_means[ds]["rlda"]["eff"], new_means[ds]["rlda"]["eff"]
        o_m, n_m = old_means[ds]["mada"]["eff"], new_means[ds]["mada"]["eff"]
        o_c, n_c = old_means[ds]["cart"]["eff"], new_means[ds]["cart"]["eff"]
        L.append(
            f"| `{ds}` | {o_r:.3f} | {n_r:.3f} | {n_r-o_r:+.3f} | "
            f"{o_m:.3f} | {n_m:.3f} | {n_m-o_m:+.3f} | "
            f"{o_c:.3f} | {n_c:.3f} | {n_c-o_c:+.3f} |"
        )

    L += [
        "",
        "## Per-dataset Fid and Cov (RLDA / MADA)",
        "",
        "| dataset | old RLDA Fid/Cov | new RLDA Fid/Cov | old MADA Fid/Cov | new MADA Fid/Cov |",
        "|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        def fc(means, m):
            return f"{means[ds][m]['fid']:.3f}/{means[ds][m]['cov']:.3f}"
        L.append(
            f"| `{ds}` | {fc(old_means,'rlda')} | {fc(new_means,'rlda')} | "
            f"{fc(old_means,'mada')} | {fc(new_means,'mada')} |"
        )

    # largest movers
    movers = []
    for ds in DATASETS:
        for arm, name in (("rlda", "RLDA"), ("mada", "MADA")):
            d = new_means[ds][arm]["eff"] - old_means[ds][arm]["eff"]
            movers.append((abs(d), d, name, ds))
    movers.sort(reverse=True)
    L += [
        "",
        "## Largest Eff moves (absolute)",
        "",
        "| arm | dataset | Δ Eff |",
        "|---|---|---:|",
    ]
    for _abs, d, name, ds in movers[:12]:
        L.append(f"| {name} | `{ds}` | {d:+.3f} |")

    # CART / anchors should barely move if only RL training changed; they CAN
    # move because predicted-basis selection re-ranks the same candidate pools.
    L += [
        "",
        "## Did the baselines move?",
        "",
        "CART, random search, and the anchor reducers were **not retrained**. "
        "They can still change because D_val ranking uses predicted class Cov, "
        "so a different rule can win at k=1.",
        "",
        "| method | Δ Fid | Δ Cov | Δ Eff |",
        "|---|---:|---:|---:|",
    ]
    for method in ("cart", "random_search", "sp_anchors", "greedy_anchors"):
        o, n = old_head[method], new_head[method]
        L.append(
            f"| `{DISP[method]}` | {dlt(n['fid'], o['fid'])} | "
            f"{dlt(n['cov'], o['cov'])} | {dlt(n['eff'], o['eff'])} |"
        )

    L += [
        "",
        "## What this means for the manuscript",
        "",
        "- Tables 3 (k=1), 4 Family 1, and 5 should be replaced from "
        "`docs/RESULTS_paper_final_predicted.md` if you cite the checkpoint / "
        "pool / predicted-basis fixes.",
        "- Table 3 five-rule columns and Table 4 Family 2 still come from the "
        "old `k_sweep` / `pool20` lock. Do not splice them next to these k=1 cells.",
        "- P1 (perturbed 5-seed) and a 5-seed k-sweep are **not** in either file yet.",
        "",
    ]
    OUT2.write_text("\n".join(L) + "\n")


def main() -> int:
    new_cells = ingest_paper_final()
    old_cells = ingest_old()
    n_new = sum(1 for m in METHODS for ds in DATASETS for s in SEEDS if (ds, m, s) in new_cells)
    n_old = sum(1 for m in METHODS for ds in DATASETS for s in SEEDS if (ds, m, s) in old_cells)
    print(f"new empirical cells {n_new}/360; old lock {n_old}/360")
    write_new_tables(new_cells)
    write_comparison(new_cells, old_cells)
    print(f"wrote {OUT1}")
    print(f"wrote {OUT2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
