"""
Paper figures from revision result JSONs (C-12, C-27, C-31).

Usage:
    python paper/make_figures.py --results_dir revision/results --out_dir paper/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.metrics import break_even_n  # noqa: E402

AMORTIZED = {
    "rlda", "mada", "ddpg", "sac", "maddpg", "masac", "cart", "random_search",
}
PER_INSTANCE = {"anchors", "sp_anchors", "greedy_anchors"}


def _load_results(results_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if data.get("schema") == "rlda_mada_result_v1":
            rows.append(data)
    return rows


def _mean(vals) -> Optional[float]:
    arr = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(arr)) if arr else None


def _fixed_and_marginal(row: Dict[str, Any]) -> Tuple[float, float]:
    q = row.get("queries") or {}
    n_bb = float(q.get("n_blackbox_queries") or 0.0)
    n_rep = float(q.get("n_reporting_queries") or 0.0)
    n_test = float((row.get("classifier_accuracy") or {}).get("n") or 0.0)
    method = str(row.get("method", "")).lower()
    if method in PER_INSTANCE:
        n_explained = n_test if n_test > 0 else 1.0
        return 0.0, n_bb / n_explained
    per = (n_rep / n_test) if n_test > 0 else 0.0
    return n_bb, per


def break_even_curves(rows: List[Dict[str, Any]], n_max: int = 400) -> Dict[str, Any]:
    """Cumulative query cost vs number of explained instances."""
    by_method = defaultdict(list)
    for r in rows:
        by_method[(r["dataset"], r["method"])].append(r)

    ns = np.arange(1, n_max + 1, dtype=float)
    series = {}
    for (dataset, method), group in sorted(by_method.items()):
        fixed = _mean([_fixed_and_marginal(r)[0] for r in group]) or 0.0
        marg = _mean([_fixed_and_marginal(r)[1] for r in group]) or 0.0
        series[f"{dataset}:{method}"] = {
            "n": ns.tolist(),
            "cost": (fixed + marg * ns).tolist(),
            "fixed": fixed,
            "marginal": marg,
            "amortized": method.lower() in AMORTIZED,
        }

    crossovers = []
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        amortized = [k for k in series if k.startswith(ds + ":") and series[k]["amortized"]]
        baselines = [k for k in series if k.startswith(ds + ":") and not series[k]["amortized"]]
        for a in amortized:
            for b in baselines:
                n_star = break_even_n(
                    series[a]["fixed"], series[a]["marginal"], series[b]["marginal"]
                )
                crossovers.append({
                    "dataset": ds, "method": a.split(":", 1)[1],
                    "baseline": b.split(":", 1)[1], "n_break_even": n_star,
                })
        # If no real baseline, overlay a typical Anchors cost (Ribeiro et al.: ~thousands of queries).
        if not baselines:
            typical = 2000.0
            for a in amortized:
                n_star = break_even_n(series[a]["fixed"], series[a]["marginal"], typical)
                crossovers.append({
                    "dataset": ds, "method": a.split(":", 1)[1],
                    "baseline": "anchors_typical_2000q", "n_break_even": n_star,
                })
            series[f"{ds}:anchors_typical_2000q"] = {
                "n": ns.tolist(),
                "cost": (typical * ns).tolist(),
                "fixed": 0.0,
                "marginal": typical,
                "amortized": False,
                "hypothetical": True,
            }
    return {"series": series, "crossovers": crossovers, "n_max": n_max}


def frontier_points(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fid_∪ vs Cov_∪ at each (τ_P, τ_C) — C-31. Empty until a threshold grid exists."""
    out = []
    for r in rows:
        for ck, block in (r.get("per_class") or {}).items():
            union = block.get("union") or {}
            out.append({
                "dataset": r["dataset"],
                "method": r["method"],
                "class": ck,
                "tau_p": r["tau_p"],
                "tau_c": r["tau_c"],
                "fid_union": union.get("fidelity"),
                "cov_union": union.get("coverage"),
            })
    return out


def _plot(curves: Dict[str, Any], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_ds = defaultdict(list)
    for key, ser in curves["series"].items():
        ds, method = key.split(":", 1)
        by_ds[ds].append((method, ser))

    n_ds = max(1, len(by_ds))
    fig, axes = plt.subplots(n_ds, 1, figsize=(7.5, 3.2 * n_ds), squeeze=False)
    for ax, (ds, items) in zip(axes[:, 0], sorted(by_ds.items())):
        for method, ser in sorted(items, key=lambda x: x[0]):
            ls = "--" if ser.get("hypothetical") else "-"
            ax.plot(ser["n"], ser["cost"], ls, label=method)
        for cr in curves["crossovers"]:
            if cr["dataset"] != ds or cr["n_break_even"] is None:
                continue
            ax.axvline(cr["n_break_even"], color="0.5", lw=0.8, alpha=0.7)
            ax.annotate(
                f"break-even n={cr['n_break_even']:.0f}",
                xy=(cr["n_break_even"], 0),
                xytext=(4, 8),
                textcoords="offset points",
                fontsize=8,
                rotation=90,
                va="bottom",
            )
        ax.set_title(f"{ds}: cumulative black-box queries")
        ax.set_xlabel("Instances explained")
        ax.set_ylabel("Cumulative queries")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(1, curves["n_max"])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="revision/results")
    ap.add_argument("--out_dir", default="paper/figures")
    ap.add_argument("--n_max", type=int, default=400)
    args = ap.parse_args()
    rows = _load_results(Path(args.results_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No result files in {args.results_dir}")
        return
    curves = break_even_curves(rows, n_max=args.n_max)
    (out_dir / "break_even.json").write_text(json.dumps(curves, indent=2))
    (out_dir / "frontier.json").write_text(json.dumps(frontier_points(rows), indent=2))
    _plot(curves, out_dir / "break_even.png")
    print(f"Wrote {out_dir / 'break_even.png'} and JSON summaries")
    for cr in curves["crossovers"]:
        print(
            f"  {cr['dataset']} {cr['method']} vs {cr['baseline']}: "
            f"n*={cr['n_break_even']}"
        )


if __name__ == "__main__":
    main()
