"""Print a one-leg (dataset × method × seed) summary from eval JSON + extracted rules."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _fmt(x: Any, nd: int = 3) -> str:
    v = _f(x)
    return "—" if v is None else f"{v:.{nd}f}"


def _metrics_block(obj: Any) -> Dict[str, Any]:
    """Union is a flat BoxMetrics dict; best nests that dict under 'fidelity'."""
    if not isinstance(obj, dict):
        return {}
    inner = obj.get("fidelity")
    if isinstance(inner, dict):
        return inner
    return obj


def _class_summary_line(cls: str, cd: Dict[str, Any]) -> str:
    b = _metrics_block(cd.get("best") or {})
    u = _metrics_block(cd.get("union") or {})
    n_best = b.get("n_covered")
    n_union = u.get("n_covered")
    return (
        f"  {cls}: best Fid={_fmt(b.get('fidelity'))} Pur={_fmt(b.get('purity'))} "
        f"clsC={_fmt(b.get('coverage'), 2)} n={n_best} | "
        f"union Fid={_fmt(u.get('fidelity'))} Pur={_fmt(u.get('purity'))} "
        f"clsC={_fmt(u.get('coverage'), 2)} n={n_union}"
    )


def _collect_anchors(rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for a in obj.get("anchors") or obj.get("all_anchors") or []:
                if isinstance(a, dict):
                    out.append(a)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(rules.get("per_class_results") or {})
    return out


def _rollout_stats(rules_file: Optional[Path]) -> str:
    if rules_file is None or not rules_file.exists():
        return "  rollouts: (no extracted_rules file)"
    data = json.loads(rules_file.read_text())
    ancs = _collect_anchors(data)
    steps, nbox, P, C = [], [], [], []
    for a in ancs:
        if a.get("n_steps") is not None:
            steps.append(int(a["n_steps"]))
        if a.get("n_in_box") is not None:
            nbox.append(int(a["n_in_box"]))
        if a.get("precision_rollout_estimated") is not None:
            P.append(float(a["precision_rollout_estimated"]))
        cov = a.get("coverage_class_conditional_rollout_estimated")
        if cov is None:
            cov = a.get("coverage_rollout_estimated")
        if cov is not None:
            C.append(float(cov))
    lines = [f"  rollouts n={len(ancs)}"]
    if steps:
        s = np.array(steps)
        lines.append(
            f"  n_steps med={int(np.median(s))} frac==2={np.mean(s == 2):.2f} "
            f"frac==100={np.mean(s == 100):.2f} unique={sorted(set(int(x) for x in s))[:8]}"
        )
    else:
        lines.append("  n_steps: not stored")
    if nbox:
        n = np.array(nbox)
        lines.append(
            f"  n_in_box min/med/max={n.min()}/{np.median(n):.0f}/{n.max()} "
            f"frac==1={np.mean(n == 1):.2f} frac==0={np.mean(n == 0):.2f}"
        )
    if P:
        p = np.array(P)
        lines.append(f"  P_roll med={np.median(p):.3f} frac>=0.9={np.mean(p >= 0.9):.2f}")
    if C:
        c = np.array(C)
        lines.append(f"  C_roll med={np.median(c):.3f} max={c.max():.3f} frac>=0.2={np.mean(c >= 0.2):.2f}")
    return "\n".join(lines)


def summarize(dataset: str, method: str, seed: int, rules_file: Optional[str] = None) -> str:
    tp, tc = "0p90", "0p20"
    dest = REPO / "revision" / "results" / f"{dataset}__{method}__seed{seed}__tp{tp}__tc{tc}.json"
    if not dest.exists():
        return f"[{dataset} {method.upper()} seed {seed}] no eval JSON at {dest}"
    d = json.loads(dest.read_text())
    g = d.get("global_ruleset") or {}
    s = d.get("success_rate") or {}
    lines = [
        f"=== {dataset} {method.upper()} seed {seed}  TEST (rank on val) ===",
        f"  Fid={_fmt(g.get('global_fidelity'))} Pur={_fmt(g.get('global_purity'))} "
        f"abst={_fmt(g.get('abstention_rate'))} conf={_fmt(g.get('conflict_rate'))} "
        f"cov={_fmt(g.get('coverage'))}",
        f"  success={_fmt(s.get('success_rate'))} "
        f"({s.get('n_success')}/{s.get('n_episodes')} episodes hit τ_P=0.9 and τ_C=0.2)",
    ]
    for cls, cd in (d.get("per_class") or {}).items():
        if not isinstance(cd, dict):
            continue
        lines.append(_class_summary_line(cls, cd))
    rf = Path(rules_file) if rules_file else None
    lines.append(_rollout_stats(rf))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True, choices=("rlda", "mada"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rules_file", default=None)
    args = p.parse_args()
    print(summarize(args.dataset, args.method, args.seed, args.rules_file), flush=True)


if __name__ == "__main__":
    main()
