"""Write paper-scoped results and rules docs (default) or lab-notebook appendices.

Default `--apply` **overwrites** `docs/RESULTS_comparison.md` and `docs/RULES.md`
with the manuscript sources: 5-seed headline (`paper_fiveseed_overlap075`),
seed-42 ablations, scale/Track B, and paper-lock example rules. It does not
include the obsolete full-budget cluster sweep (`runs/sweep_dnn`, `runs/sweep_rf`).

`--lab` splices the full lab notebook (original-reward Track A grids, τ_C=0.05,
smokes, sweep inventory) at the `<!-- LOCAL-APPENDIX -->` / `<!-- ABLATION-APPENDIX -->`
markers. Do not use `--lab` output as paper tables.

    python -m revision.make_local_docs --apply          # paper rewrite docs
    python -m revision.make_local_docs --apply --lab  # full lab notebook
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from utils.metrics import class_cov_tau, track_a_cov_tau, track_a_eff  # noqa: E402
from revision.paper_stats import markdown_section as paper_wilcoxon_section  # noqa: E402

MAIN = os.path.join(REPO, "runs", "low_budget_seed42")
TAUC05 = os.path.join(REPO, "runs", "low_budget_seed42_tp090_tc005")
CONFLICT = os.path.join(REPO, "runs", "low_budget_seed42_conflict_align")
SAC_ROOT = os.path.join(REPO, "runs", "conflict_align_sac_seed42")
PAPER_FIVESEED = os.path.join(REPO, "runs", "paper_fiveseed_overlap075")
WYODOT = os.path.join(REPO, "runs", "wyodot_fiveseed_overlap075")
WYODOT_SEEDS = ("42", "43", "44", "45", "46")
WYODOT_CLASS_NAMES = {
    "class_0": "Dry",
    "class_1": "Ice",
    "class_2": "Slush",
    "class_3": "Snow",
    "class_4": "Wet",
}
OVERLAP_ROOTS = {
    "w075_cross": os.path.join(REPO, "runs", "mada_overlap075_seed42"),
    "w100_cross": os.path.join(REPO, "runs", "mada_overlap100_seed42"),
    "w075_nocross": os.path.join(REPO, "runs", "mada_overlap075_nocross_seed42"),
    "w100_nocross": os.path.join(REPO, "runs", "mada_overlap100_nocross_seed42"),
}
COORD_ROOT = os.path.join(REPO, "runs", "coord_ablation_seed42")
REWARD_ROOT = os.path.join(REPO, "runs", "reward_ablation_seed42")
SMOKES = [os.path.join(REPO, "runs", "smoke_iris"), os.path.join(REPO, "runs", "smoke_iris2")]
ALL_DATASETS = [
    "iris", "synthetic", "wine", "sick", "breast_cancer", "mammography",
    "housing", "heloc", "uci_credit", "uci_adult", "folktables_income_CA_2018",
]

# Big-to-small: the scalability answer reads top-down.
DATASETS = [
    "folktables_income_CA_2018", "uci_adult", "housing", "mammography", "heloc",
    "sick", "synthetic", "uci_credit", "breast_cancer", "wine", "iris",
]
# The 11-set tables above read the `paper_fiveseed_overlap075` tree, which holds
# 11 datasets. The EDA is a property of the data, not of a run tree, so it covers
# all 12 — WyoDOT included, ordered by size like the rest.
EDA_DATASETS = [
    "folktables_income_CA_2018", "uci_adult", "wyodot_kvdw_labeled", "housing",
    "mammography", "heloc", "sick", "synthetic", "uci_credit", "breast_cancer",
    "wine", "iris",
]
METHODS = ["mada", "rlda", "cart", "greedy_anchors", "sp_anchors", "random_search"]
PAPER_COMPARE_METHODS = [
    "mada", "rlda", "greedy_anchors", "sp_anchors", "cart", "random_search",
]
RL = {"mada", "rlda"}
RL_TRACK_A_METHODS = frozenset(RL)
NEW_DATASETS = ["folktables_income_CA_2018", "heloc", "mammography", "sick"]
FNAME = re.compile(r"^(?P<ds>.+?)__(?P<m>[a-z_]+)__seed(?P<seed>\d+)__tp")

BEGIN = "<!-- LOCAL-APPENDIX -->"
END = "<!-- /LOCAL-APPENDIX -->"
ABL_BEGIN = "<!-- ABLATION-APPENDIX -->"
ABL_END = "<!-- /ABLATION-APPENDIX -->"

# Datasets where SAC/MASAC diverge most from DDPG/MADDPG (for rules appendix).
ABLATION_RULE_DATASETS = ["iris", "sick", "wine", "housing"]


def load(root: str, backend: str = "dnn") -> Dict[Any, Dict[str, Any]]:
    """(dataset, method) -> result dict, for one run root and black box."""
    out: Dict[Any, Dict[str, Any]] = {}
    for path in glob.glob(os.path.join(root, backend, "results", "*", "*.json")):
        base = os.path.basename(path)
        if "__instances__" in base:
            continue
        m = FNAME.match(base)
        if not m:
            continue
        with open(path) as fh:
            out[(m["ds"], m["m"])] = json.load(fh)
    return out


def load_instances(root: str, backend: str = "dnn") -> Dict[Any, Dict[str, Any]]:
    out: Dict[Any, Dict[str, Any]] = {}
    for path in glob.glob(os.path.join(root, backend, "results", "*", "*__instances__*.json")):
        with open(path) as fh:
            d = json.load(fh)
        out[(d["dataset"], d["method"])] = d
    return out


def g(d: Optional[Dict[str, Any]], *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def f(v, n=3, comma=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return f"{v:,}" if comma else str(v)


def branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def commit_of(data: Dict[Any, Dict[str, Any]]) -> str:
    for res in data.values():
        c = res.get("git_commit")
        if c:
            return c[:7]
    return "unknown"


# ------------------------------------------------------------------ results doc

def results_appendix(main, tauc05, smokes, inst) -> List[str]:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [
        BEGIN,
        "",
        "---",
        "",
        "# Appendix — local single-seed sweep (this machine)",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`. "
        f"Branch `{branch()}`, code `{commit_of(main)}`.",
        "",
        "The tables above come from the multi-seed cluster sweep. This appendix "
        "covers the runs made on the development machine, which use a **much "
        "smaller budget and one seed**, so their numbers are *not* folded into "
        "the means above and should not be quoted as the paper's headline result. "
        "What they do buy is **breadth**: four datasets the cluster sweep never "
        "touched, including one with 195,665 rows, plus a Track B instance-level "
        "comparison against classical Anchors and a wall-clock cost accounting.",
        "",
        "| | |",
        "|---|---|",
        "| Arms | MADA (MADDPG), RLDA (DDPG), single seed 42 |",
        "| Black box | DNN (`runs/low_budget_seed42/dnn/`) |",
        "| Budget | MADA 24,000 frames/agent (5 FidCov evals at `evaluation_interval: 4800`); "
        "RLDA 25,000 steps/class in this sweep (later ablations use matched totals "
        "`mada_frames×agents_per_class×n_classes`) |",
        "| τ_P / τ_C | 0.90 / 0.10, with a 0.90 / 0.05 re-evaluation in `runs/low_budget_seed42_tp090_tc005/` |",
        "| Baselines | CART, greedy_anchors, sp_anchors, random_search — same splits, same classifier file |",
        "| Selection | rank on D_val, report on D_test; greedy marginal-gain union |",
        "| Not finished | `covtype` — RLDA inference was killed by the OS (exit −9, out of memory) after training; no result JSON exists for either arm |",
        "",
        "**New datasets in this sweep:** " + ", ".join(f"`{d}`" for d in NEW_DATASETS) +
        ". The first is the reviewers' scale answer; the middle two are the "
        "imbalanced medical sets; `heloc` is the credit-risk set with sentinel "
        "codes (−9/−8/−7), which is why some printed bounds are negative and "
        "correctly so — those are values that occur in the raw data.",
        "",
        "## Scale and black-box accuracy",
        "",
        "One classifier file per dataset, shared by every arm and every baseline, "
        "so all methods explain an identical model.",
        "",
        "| dataset | train | val | test | total | train acc | val acc | test acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        res = main.get((ds, "rlda")) or main.get((ds, "mada"))
        if res is None:
            continue
        ss = g(res, "extra", "split_sizes", default={}) or {}
        acc = res.get("classifier_accuracy") or {}
        total = sum(v for v in ss.values() if isinstance(v, int))
        L.append(
            f"| `{ds}` | {ss.get('train', 0):,} | {ss.get('val', 0):,} | {ss.get('test', 0):,} | "
            f"{total:,} | {f(acc.get('train_accuracy'))} | {f(acc.get('val_accuracy'))} | "
            f"{f(acc.get('test_accuracy'))} |"
        )

    L += [
        "",
        "## Class-level rule sets (Track A)",
        "",
        "**Fid** is fidelity to the black box, P(rule class = f̂(x) | covered). "
        "**Pur** is agreement with the ground-truth label on the same rows. They "
        "are reported side by side on purpose: where they diverge, the rule is "
        "describing the classifier rather than the data, which is what a post-hoc "
        "explanation is supposed to do. Conflict and Abstain are lower-is-better; "
        "Cov + Abstain = 1. Success is the fraction of episodes reaching both τ_P "
        "and τ_C and applies only to the RL arms. Queries are the black-box calls "
        "spent *building* the rule set, excluding policy training.",
        "",
        "`feats/rule` is the mean number of constrained features per rule. The "
        "baseline figures were rescaled by `revision.recompute_compactness` "
        "against each feature's observed X_train range: CART and the Anchors "
        "family build boxes in original feature units, and measuring their "
        "widths against a unit-space [0,1] range read a real one-sided iris "
        "split as 0 active features and all 16 breast_cancer features as "
        "active. `random_search` and both RL arms are unit-space and were "
        "never affected.",
        "",
    ]
    for ds in DATASETS:
        if not any((ds, m) in main for m in METHODS):
            continue
        tag = " *(new in this sweep)*" if ds in NEW_DATASETS else ""
        L += [
            f"### {ds}{tag}",
            "",
            "| method | Fid | Pur | Cov | Conflict | Abstain | Success | extraction queries | feats/rule |",
            "|---|---|---|---|---|---|---|---:|---:|",
        ]
        for m in METHODS:
            r = main.get((ds, m))
            if r is None:
                L.append(f"| {m} | — | — | — | — | — | — | — | — |")
                continue
            gr = r.get("global_ruleset") or {}
            succ = g(r, "success_rate", "success_rate") if m in RL else None
            feats = f(g(r, "compactness", "mean_active_features"), 2)
            L.append(
                f"| {m} | {f(gr.get('global_fidelity'))} | {f(gr.get('global_purity'))} | "
                f"{f(gr.get('coverage'))} | {f(gr.get('conflict_rate'))} | "
                f"{f(gr.get('abstention_rate'))} | {f(succ) if m in RL else '—'} | "
                f"{f(g(r, 'queries', 'n_blackbox_queries'), comma=True)} | {feats} |"
            )
        L.append("")

    L += [
        "## Instance-level comparison against classical Anchors (Track B)",
        "",
    ]
    L += paper_track_b_intro()
    for ds in DATASETS:
        for arm in ("rlda", "mada"):
            d = inst.get((ds, arm))
            if d is None:
                continue
            a = g(d, "pi", "summary", "all", default={}) or {}
            anc = g(d, "anchors", "summary", default={}) or {}
            c = d.get("cost") or {}
            spi, sanc = c.get("wall_s_per_x_pi"), c.get("wall_s_per_x_anchors")
            sp = (sanc / spi) if (spi and sanc) else None
            L.append(
                f"| `{ds}` | {arm.upper()} | {d.get('n_scored')} | {f(a.get('emp_fid'))} | "
                f"{f(a.get('emp_pur'))} | {f(a.get('emp_cov_marginal'))} | "
                f"{f(a.get('perturb_fid'))} | {f(anc.get('perturb_fid'))} | "
                f"{f(anc.get('coverage'))} | {f(c.get('queries_per_x_pi'), 1)} | "
                f"{f(c.get('queries_per_x_anchors'), 1, comma=True)} | {f(spi, 4)} | "
                f"{f(sanc, 4)} | {f(sp, 1)}× |"
            )

    L += [
        "",
        "## Coverage-threshold sensitivity (τ_C 0.10 → 0.05)",
        "",
        "Same checkpoints, same rules, re-evaluated at the looser coverage "
        "threshold. Fid/Cov are unchanged everywhere — the selector keeps the "
        "same rules — so the only thing τ_C moves is the episode success rate.",
        "",
        "| dataset | Success @ τ_C=0.10 | Success @ τ_C=0.05 | Δ |",
        "|---|---|---|---|",
    ]
    for ds in DATASETS:
        a = g(main.get((ds, "rlda")), "success_rate", "success_rate")
        b = g(tauc05.get((ds, "rlda")), "success_rate", "success_rate")
        if a is None and b is None:
            continue
        delta = f"{b - a:+.3f}" if (a is not None and b is not None) else "—"
        L.append(f"| `{ds}` | {f(a)} | {f(b)} | {delta} |")

    L += [
        "",
        "## Repeat runs (iris, both black boxes)",
        "",
        "`runs/smoke_iris` and `runs/smoke_iris2` are two independent executions "
        "of the same pipeline at the same seed. Identical rows mean the pipeline "
        "is deterministic given a seed; the rows that move are the ones with "
        "sampling in the baseline, not in the arms.",
        "",
        "| run | black box | method | Fid | Pur | Cov | Conflict | Success |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for root in smokes:
        name = os.path.basename(root)
        for backend in ("dnn", "rf"):
            data = load(root, backend)
            for m in METHODS:
                r = data.get(("iris", m))
                if r is None:
                    continue
                gr = r.get("global_ruleset") or {}
                succ = g(r, "success_rate", "success_rate") if m in RL else None
                L.append(
                    f"| `{name}` | {'DNN' if backend == 'dnn' else 'RF'} | {m} | "
                    f"{f(gr.get('global_fidelity'))} | {f(gr.get('global_purity'))} | "
                    f"{f(gr.get('coverage'))} | {f(gr.get('conflict_rate'))} | "
                    f"{f(succ) if m in RL else '—'} |"
                )
    L += ["", END, ""]
    return L


# -------------------------------------------------------------------- rules doc

def rules_appendix(main) -> List[str]:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [
        BEGIN,
        "",
        "---",
        "",
        "# Appendix — rules from the local single-seed sweep",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`.",
        "",
        "Seed 42, DNN black box, `runs/low_budget_seed42/`. These are the four "
        "datasets the multi-seed sweep above does not cover. Rules are the "
        "validation-selected union printed in **original feature units** and "
        "scored on the test split; Fid is fidelity to the classifier, Pur is "
        "agreement with the label.",
        "",
        "`heloc` bounds of −9, −8 and −7 are not a denormalization bug: they are "
        "HELOC's own missing-data sentinel codes and occur verbatim in the raw "
        "file. Every printed bound is checked against the observed feature range "
        "before it is written (`assert_bounds_in_observed_range`).",
        "",
    ]
    for ds in NEW_DATASETS:
        L += [f"## {ds}", ""]
        for m in ("mada", "rlda"):
            r = main.get((ds, m))
            if r is None:
                L += [f"**{m.upper()}** — no result", ""]
                continue
            L += [f"**{m.upper()}**", ""]
            for cls, blk in sorted((r.get("per_class") or {}).items()):
                rules = blk.get("selected_rules") or []
                u = blk.get("union") or {}
                L.append(
                    f"- `{cls}` (k={blk.get('k')}, union Fid {f(u.get('fidelity'))}, "
                    f"Pur {f(u.get('purity'))}, Cov {f(u.get('coverage'))}, "
                    f"n={u.get('n_covered')})"
                )
                if not rules:
                    L.append("  - *(no rule selected)*")
                for i, rule in enumerate(rules, 1):
                    rm = rule.get("report_metrics") or {}
                    L.append(
                        f"  {i}. {rule.get('display_rule', '(no display string)')}"
                        f"  — Fid {f(rm.get('fidelity'))}, Pur {f(rm.get('purity'))}, "
                        f"Cov {f(rm.get('coverage'))}, n={rm.get('n_covered')}"
                    )
            L.append("")
    L += [END, ""]
    return L


def count_track_a(
    root: str,
    sub: str,
    methods: frozenset = RL_TRACK_A_METHODS,
) -> int:
    """Count Track A JSONs under results/{sub}/; default counts RL arms only."""
    for base in (
        os.path.join(root, "dnn", "results", sub),
        os.path.join(root, "results", sub),
    ):
        if not os.path.isdir(base):
            continue
        n = 0
        for path in glob.glob(os.path.join(base, "*__*__tp0p90__tc0p10.json")):
            m = FNAME.match(os.path.basename(path))
            if m and m["m"] in methods:
                n += 1
        if n:
            return n
    return 0


def paper_fiveseed_progress() -> str:
    log = Path(PAPER_FIVESEED) / "logs" / "sweep_progress.log"
    if not log.exists():
        return "not started"
    lines = [ln.strip() for ln in log.read_text().splitlines() if ln.strip()]
    return lines[-1] if lines else "no log lines"


def load_scoreboard_file(path: str) -> List[Dict[str, float]]:
    """Single-arm scoreboard (dataset row) -> list of metric dicts."""
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, float]] = []
    with open(path) as fh:
        hdr = fh.readline().strip().split("\t")
        for ln in fh:
            parts = ln.strip().split("\t")
            if len(parts) < 6 or parts[1] in ("", "MISSING"):
                continue
            fid, cov, conf = float(parts[1]), float(parts[5]), float(parts[4])
            rows.append({"fid": fid, "cov": cov, "conf": conf, "eff": fid * cov})
    return rows


def mean_scoreboard_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {k: sum(r[k] for r in rows) / n for k in ("fid", "cov", "conf", "eff")}


def paper_seed_means() -> Dict[str, Dict[str, Dict[str, float]]]:
    """seed -> arm -> mean metrics (only datasets present)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm, sub in (("mada", "maddpg"), ("rlda", "ddpg")):
        pat = os.path.join(PAPER_FIVESEED, "results", sub, "*__*__tp0p90__tc0p10.json")
        acc: Dict[str, Dict[str, List[float]]] = {}
        for path in glob.glob(pat):
            base = os.path.basename(path)
            m = FNAME.match(base)
            if not m:
                continue
            seed = m["seed"]
            with open(path) as fh:
                g = (json.load(fh).get("global_ruleset") or {})
            bucket = acc.setdefault(seed, {})
            fid = g.get("global_fidelity")
            cov = g.get("coverage")
            conf = g.get("conflict_rate")
            try:
                if fid is not None:
                    bucket.setdefault("fid", []).append(float(fid))
            except (TypeError, ValueError):
                pass
            try:
                if cov is not None:
                    bucket.setdefault("cov", []).append(float(cov))
            except (TypeError, ValueError):
                pass
            try:
                if conf is not None:
                    bucket.setdefault("conf", []).append(float(conf))
            except (TypeError, ValueError):
                pass
            eff = track_a_eff(g)
            if eff is not None:
                bucket.setdefault("eff", []).append(eff)
            ct = track_a_cov_tau(g)
            if ct is not None:
                bucket.setdefault("cov_tau", []).append(ct)
        for seed, m in acc.items():
            out.setdefault(seed, {})[arm] = {
                k: sum(v) / len(v) for k, v in m.items() if v
            }
            out[seed][arm]["n"] = len(m.get("eff", []))
    return out


def load_scoreboard_means(path: str, arm_key: str = "arm") -> Dict[str, Dict[str, float]]:
    """Mean Fid/Cov/Conf from a scoreboard.tsv keyed by arm name."""
    p = os.path.join(path, "scoreboard.tsv")
    if not os.path.isfile(p):
        return {}
    acc: Dict[str, Dict[str, List[float]]] = {}
    with open(p) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            arm = row.get(arm_key, row.get("arm", ""))
            if not arm:
                continue
            acc.setdefault(arm, {}).setdefault("fid", []).append(float(row["fid"]))
            acc.setdefault(arm, {}).setdefault("cov", []).append(float(row["cov"]))
            acc.setdefault(arm, {}).setdefault("conf", []).append(float(row["conf"]))
    return {a: {k: sum(v) / len(v) for k, v in m.items()} for a, m in acc.items()}


def sweep_catalog_section(stamp: str) -> List[str]:
    n11 = 11
    paper_ddpg = count_track_a(PAPER_FIVESEED, "ddpg")
    paper_mada = count_track_a(PAPER_FIVESEED, "maddpg")
    paper_done = paper_ddpg + paper_mada
    return [
        "## Sweep inventory (this machine)",
        "",
        f"Catalog updated {stamp}. Paths under `runs/`.",
        "",
        "| sweep | seeds | datasets | budget | arms | status | path |",
        "|---|---:|---:|---|---|---|---|",
        "| **cluster DNN/RF** | 42–44 | 7 | 144k / 270k | MADDPG, DDPG + baselines | complete (remote); **old reward** | `runs/sweep_dnn`, `sweep_rf` |",
        f"| **low_budget** | 42 | 11 | 24k / 25k per-class RLDA | MADDPG, DDPG + baselines | "
        f"**complete** ({count_track_a(MAIN, 'ddpg')}+{count_track_a(MAIN, 'maddpg')} RL Track A) | "
        f"`runs/low_budget_seed42/dnn` |",
        f"| **conflict_align** (w=0.5) | 42 | 11 | low matched | MADDPG, DDPG | "
        f"**complete** ({count_track_a(CONFLICT, 'ddpg')}+{count_track_a(CONFLICT, 'maddpg')} Track A) | "
        f"`runs/low_budget_seed42_conflict_align/dnn` |",
        "| **overlap weight search** | 42 | 11 | low | 4× MADDPG configs | **complete** (4×11) | `runs/mada_overlap*_seed42/` |",
        f"| **coord ablation** | 42 | 11 | low, matched steps | 4× MADDPG | **complete** (4×11) | `runs/coord_ablation_seed42/` |",
        f"| **reward ablation** | 42 | 11 | low | 5× MADDPG | **complete** | `runs/reward_ablation_seed42/` |",
        f"| **SAC/MASAC** | 42 | 11 | low | SAC, MASAC | "
        f"**complete** ({count_track_a(SAC_ROOT, 'sac')}+{count_track_a(SAC_ROOT, 'masac')} Track A) | "
        f"`runs/conflict_align_sac_seed42/dnn` |",
        f"| **5-seed paper (w=0.75)** | 42–46 | 11 | low matched | MADDPG, DDPG | **complete** ({paper_done}/110 Track A) | `runs/paper_fiveseed_overlap075/` |",
        "| τ_C 0.05 re-eval | 42 | 11 | re-eval | — | complete | `runs/low_budget_seed42_tp090_tc005/` |",
        "| ~~5-seed full-budget conflict_align~~ | — | — | — | — | **cancelled** | `runs/conflict_align_sweep_dnn/` |",
        "",
        "**Reviewer package:** headline statistics from **`paper_fiveseed_overlap075`** "
        "(overlap w=0.75, cross-class ON, low matched budget). Use seed-42 ablations "
        "for mechanism decomposition. **Eff** = Fid×Cov is the primary — rule-set "
        "accuracy vs f̂ with abstention as failure (`n_fid_agree/n_eval`). "
        "**Cov_τ** = Cov·1[Fid ≥ 0.90] is reported alongside as a diagnostic, not "
        "headlined. Do not headline conditional Fid alone — `random_search` beats "
        "both RL arms on Fid in many cells at tiny coverage.",
        "",
        "Monitor the active sweep:",
        "```bash",
        "tail -f runs/paper_fiveseed_overlap075/logs/sweep_progress.log",
        "```",
        "",
    ]


def _mean_track_a(data: Dict[Any, Dict[str, Any]], methods: List[str]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}
    for (_ds, m), r in data.items():
        if m not in methods:
            continue
        gr = r.get("global_ruleset") or {}
        for k, key in (
            ("fid", "global_fidelity"), ("pur", "global_purity"), ("cov", "coverage"),
            ("conf", "conflict_rate"), ("abst", "abstention_rate"),
        ):
            acc[m].setdefault(k, []).append(float(gr.get(key, 0.0)))
        eff = track_a_eff(gr)
        if eff is not None:
            acc[m].setdefault("eff", []).append(eff)
        ct = track_a_cov_tau(gr)
        if ct is not None:
            acc[m].setdefault("cov_tau", []).append(ct)
    return {
        m: {k: sum(v) / len(v) for k, v in acc[m].items() if v}
        for m in methods
    }


def _msd(vals: List[float]) -> str:
    if not vals:
        return "—"
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return f"{mu:.3f}"
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return f"{mu:.3f} ± {var ** 0.5:.3f}"


def _finite(vals: List[Any]) -> List[float]:
    out: List[float] = []
    for v in vals:
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def iter_paper_track_a_paths():
    root = os.path.join(PAPER_FIVESEED, "results")
    for sub in ("maddpg", "ddpg", "baselines"):
        for path in glob.glob(os.path.join(root, sub, "*__tp0p90__tc0p10.json")):
            base = os.path.basename(path)
            if "__instances__" in base:
                continue
            m = FNAME.match(base)
            if m:
                yield path, m


def load_paper_track_a_cells() -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for path, m in iter_paper_track_a_paths():
        with open(path) as fh:
            d = json.load(fh)
        cells.append({
            "dataset": m["ds"],
            "method": m["m"],
            "seed": m["seed"],
            "data": d,
        })
    return cells


def count_paper_baselines() -> int:
    n = 0
    for path, m in iter_paper_track_a_paths():
        if m["m"] in {"cart", "random_search", "sp_anchors", "greedy_anchors"}:
            n += 1
    return n


def load_paper_instance_cells() -> List[Dict[str, Any]]:
    """Track B JSONs under paper_fiveseed (includes imported seed-42 reuse)."""
    cells: List[Dict[str, Any]] = []
    root = os.path.join(PAPER_FIVESEED, "results")
    for sub in ("maddpg", "ddpg"):
        for path in glob.glob(os.path.join(root, sub, "*__instances__*.json")):
            with open(path) as fh:
                cells.append(json.load(fh))
    return cells


def count_paper_track_b() -> int:
    return len(load_paper_instance_cells())


def _median(vals: List[Any]) -> Optional[float]:
    xs = sorted(_finite(vals))
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _msd_n(vals: List[Any], n: int = 3) -> str:
    xs = _finite(vals)
    if not xs:
        return "—"
    mu = sum(xs) / len(xs)
    if len(xs) < 2:
        return f"{mu:.{n}f}"
    var = sum((v - mu) ** 2 for v in xs) / (len(xs) - 1)
    return f"{mu:.{n}f} ± {var ** 0.5:.{n}f}"


def _instance_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    a = g(d, "pi", "summary", "all", default={}) or {}
    anc = g(d, "anchors", "summary", default={}) or {}
    c = d.get("cost") or {}
    spi, sanc = c.get("wall_s_per_x_pi"), c.get("wall_s_per_x_anchors")
    try:
        sp = (float(sanc) / float(spi)) if spi and sanc and float(spi) > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        sp = None
    return {
        "n_scored": d.get("n_scored"),
        "emp_fid": a.get("emp_fid"),
        "emp_pur": a.get("emp_pur"),
        "emp_cov": a.get("emp_cov_marginal"),
        "pert_fid": a.get("perturb_fid"),
        "anc_prec": anc.get("perturb_fid"),
        "anc_cov": anc.get("coverage"),
        "q_pi": c.get("queries_per_x_pi"),
        "q_anc": c.get("queries_per_x_anchors"),
        # One-time |D_ref| pass that builds the prediction table both arms read
        # live precision/coverage from. Used to be charged to whichever episode
        # triggered the fill, which is what made MADA's q/x read |D_train|/n_scored.
        "q_ref": c.get("reference_table_queries"),
        "break_even": c.get("break_even_n_queries"),
        "s_pi": spi,
        "s_anc": sanc,
        "speedup": sp,
    }


def paper_track_b_grouped() -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for d in load_paper_instance_cells():
        key = (d.get("dataset"), str(d.get("method", "")).lower())
        grouped.setdefault(key, []).append(d)
    return grouped


def paper_seed42_track_a() -> Dict[Any, Dict[str, Any]]:
    out: Dict[Any, Dict[str, Any]] = {}
    for cell in load_paper_track_a_cells():
        if str(cell.get("seed")) != "42":
            continue
        out[(cell["dataset"], cell["method"])] = cell["data"]
    return out


def paper_cart_position_section() -> List[str]:
    """Everything measured about CART, in one place, so the paper stops guessing."""
    return [
        "",
        "## What can and cannot be claimed against CART",
        "",
        "Four separate tests were run against the surrogate tree. Three came back "
        "negative and one positive; together they fix the framing.",
        "",
        "| Test | Result |",
        "|---|---|",
        "| Fidelity / coverage (k-sweep) | **Negative.** At k=3 CART is Fid 0.917 / Cov 0.913 and Pareto-dominates both RL arms on **6 of 12** datasets, losing on none. The K-leaf-partition excuse covers k=1 only. |",
        "| Minority-class representation | **Negative.** CART emits no rule on 7/155 class×seed cells at k=1, all of them the minority class — but **0/155 at k=3**, and even at k=1 its mean minority Cov_c exceeds RLDA's by 0.093 (p=0.021). See `CLASS_IMBALANCE.md` §9. |",
        "| Construction cost | **Negative.** On distinct black-box evaluations all three are ≈ |train| + |val|. Do not claim CART is more expensive, and do not concede that it is cheaper. |",
        "| Instance-level box vs containing leaf | **Negative.** The policy box has the strictly larger hypothesis space and still Pareto-dominates on **0/12** datasets at k=1 and **1/12** at k=3; it is also less compact (2.5–6.5 active features vs 1.0–2.0). |",
        "| **Abstention-region fidelity** | **Positive** — see below. |",
        "",
        "### The one positive: abstention selects regions of model ambiguity",
        "",
        "On exactly the test rows the RL rule set declines to cover (~23%), CART's "
        "fidelity to \\(\\hat f\\) is markedly worse than on the rows it does cover:",
        "",
        "| k | arm | Δ CART Fid (abstained − covered) | worse on | p |",
        "|---|---|---:|---:|---:|",
        "| 1 | RLDA | **−0.108** | 10/12 | 0.0049 |",
        "| 1 | MADA | **−0.124** | 11/12 | 0.0015 |",
        "| 3 | RLDA | −0.059 | 8/12 | 0.0269 |",
        "| 3 | MADA | −0.104 | 10/12 | 0.0034 |",
        "",
        "**State the confound.** The black box is *also* less accurate on those "
        "rows, so part of the gap is intrinsic region difficulty. Testing CART's "
        "fidelity drop against the classifier's own accuracy drop leaves a "
        "surrogate-specific excess of −0.074 (RLDA, k=1, p=0.021) and −0.060 "
        "(MADA, k=1, p=0.052), shrinking to ns at k=3.",
        "",
        "So the defensible sentence is **\"the rule sets abstain on regions where "
        "the black box itself is least reliable\"** — supported at both k — and "
        "not \"a partition is uniquely unreliable there\", which holds only at "
        "k=1. This is the functional claim against CART; fidelity, coverage, cost "
        "and minority representation are not.",
        "",
        "Sources: `revision/abstention_analysis.py`, "
        "`revision/trackb_vs_cart_leaf.py`, `revision/minority_class_analysis.py`, "
        "`revision/audit_query_counts.py`; consolidated in `docs/DECISION_TESTS.md`.",
        "",
    ]


def paper_marginal_cost_section() -> List[str]:
    """Cost on the only unit that is fair to all four methods."""
    path = os.path.join(REPO, "runs", "marginal_cost", "measurements.json")
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        m = json.load(fh)
    per = m.get("per_dataset") or {}
    order = [d for d in EDA_DATASETS if d in per]

    L = [
        "",
        "## Cost on the marginal-call unit (this replaces the query tables)",
        "",
        "**The old per-method query totals were not measurements** — every RL "
        "figure was an exact closed form in the split sizes, seed-invariant and "
        "independent of the frame budget, while CART was billed one pass for "
        "scoring its leaves on validation and the RL arms 12–51 re-reads of a "
        "cached table for the identical operation. § \"Extraction and training "
        "cost\" keeps them only for provenance.",
        "",
        "The unit that is fair to all four methods:",
        "",
        "> A call is **marginal** if it evaluates rows **not already in the "
        "prediction table**. The table — f̂ on the fixed splits — is a one-time "
        "**shared** setup cost, identical for every method, and is reported "
        "separately rather than folded into per-method totals.",
        "",
        "Measured, not labelled: `utils/query_meter.py` wraps "
        "`SimpleClassifier.forward` at class level and matches each input against "
        "the registered splits **row by row**, so a call on the rows inside a "
        "candidate box is a table read rather than a new query, and only rows "
        "whose exact values appear in no split are charged. All 12 datasets, "
        "seed 42, wired into both inference paths and all four baselines "
        "(`queries.marginal_cost` in every new artifact).",
        "",
        "| dataset | shared table | RLDA | MADA | CART | rand | greedy_anchors | sp_anchors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in order:
        e = per[ds]
        mpe = e["marginal_per_explanation"]
        L.append(
            f"| `{ds}` | {e['table_rows']:,} | "
            + " | ".join(
                (f"{mpe[k]:,.1f}" if k in mpe else "—")
                for k in ("rlda", "mada", "cart", "random_search",
                          "greedy_anchors", "sp_anchors")
            ) + " |"
        )
    L += [
        "",
        "Marginal classifier calls **per explanation**, seed 42.",
        "",
        "### What this settles",
        "",
        "**1. Against classical Anchors the amortisation claim is measured and "
        "large.** RLDA costs **~1 marginal row per explanation** and MADA **~3**, "
        "flat across all 12 datasets and independent of dataset size. "
        "`greedy_anchors` costs **2,220–32,650** rows per explanation, because "
        "`anchor-exp` evaluates f̂ at synthetic perturbations that no table can "
        "hold. That is three to four orders of magnitude, on a unit both sides "
        "are measured on. MADA's 3 against RLDA's 1 is the extra per-class "
        "agent it routes through, not a scaling effect — an earlier coarse "
        "version of the meter reported MADA growing to 467 on housing; that was "
        "whole-matrix matching miscounting sub-batches of tabulated rows, and it "
        "is fixed.",
        "",
        "**2. Break-even is under one explanation.** \\(N^* = \\) shared table / "
        "(anchor marginal − our marginal):",
        "",
        "| dataset | N* RLDA | N* MADA |",
        "|---|---:|---:|",
    ]
    for ds in order:
        be = (per[ds].get("break_even_n") or {})
        L.append(f"| `{ds}` | {be.get('rlda', float('nan')):.2f} | "
                 f"{be.get('mada', float('nan')):.2f} |")
    L += [
        "",
        "Median **0.75** explanations for both arms. The construction cost is "
        "repaid **before the first explanation finishes on 6 of 12 datasets**, and "
        "within ~2 explanations on 8 of 12. The tail is the large sets, where the "
        "shared table is big and classical Anchors is comparatively cheap per "
        "instance: uci_adult 17, folktables 54, wyodot 5.4 — still trivial against "
        "test splits of 9,769 / 39,133 / 6,896 rows. **This supersedes the earlier "
        "break-even figures of 25–1,013**, which divided an inflated fixed cost by "
        "a per-instance cost measured on a different unit.",
        "",
        "**3. Against CART there is no cost claim, in either direction.** CART's "
        "marginal cost is **0.00 on all 12 datasets** — it fits on tabulated train "
        "predictions and ranks leaves on tabulated val predictions, exactly as the "
        "RL arms score boxes on tabulated rows. Both amortise into the same shared "
        "table; neither pays per explanation. `random_search` is also 0. Do not "
        "write that CART is cheaper, and do not claim the reverse. The real "
        "difference against CART is wall-clock training time, which is not a "
        "black-box cost.",
        "",
    ]
    return L


def paper_pool_section() -> List[str]:
    """Matched-pool comparison against the anchor family. This gates the main claim.

    The paper cells build the anchor pool from 5 explained instances per class,
    while the RL arms select their union from ~15 (RLDA) / ~19 (MADA) candidate
    boxes per class. Pool size and union size k are separate knobs; the published
    comparison held neither of them matched. Source: `revision/run_pool20_sweep.py`.
    """
    path = os.path.join(REPO, "runs", "pool20", "curves.json")
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        pool20 = json.load(fh)
    ks_path = os.path.join(REPO, "runs", "k_sweep", "curves.json")
    rl = {}
    if os.path.isfile(ks_path):
        with open(ks_path) as fh:
            rl = json.load(fh)
    ks = ["1", "2", "3", "5", "10", "20"]

    L = [
        "",
        "## Matched-pool comparison against classical Anchors",
        "",
        "**Read this before writing any claim against the anchor family.**",
        "",
        "The paper cells build the anchor pool from **5** explained instances per "
        "class (`budget_per_class=5`). The RL arms select their union from a "
        "candidate pool of **~15 (RLDA) / ~19 (MADA)** boxes per class, measured "
        "post-dedupe across the 12 datasets. **The published comparison was never "
        "matched on pool size** — a 5-candidate pool against a 15–19-candidate "
        "one. Pool size and union size *k* are separate knobs and both must be "
        "stated wherever a comparison appears.",
        "",
        "Below, one pool per (dataset, seed) at **20 anchors/class**, reduced at "
        "every k so the explainer's unseeded sampling is held constant across the "
        "sweep. 60/60 cells, no failures.",
        "",
        "| method | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for label, src, key in (
        ("`sp_anchors` pool 20", pool20, "sp_anchors"),
        ("`greedy_anchors` pool 20", pool20, "greedy_anchors"),
        ("`mada` (pool ~19)", rl, "mada"),
        ("`rlda` (pool ~15)", rl, "rlda"),
    ):
        cells = []
        for k in ks:
            c = (src.get(key) or {}).get(k)
            cells.append(
                f"{c['fid']:.3f}/{c['cov']:.3f}/**{c['eff']:.3f}**" if c else "—"
            )
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    L += [
        "",
        "Fid / Cov / **Eff**. At pool 5 the same reducers reached Eff 0.568 "
        "(`sp_anchors`) and 0.527 (`greedy_anchors`) at k=1. **The published "
        "margins of +0.127 and +0.169 were a pool budget.**",
        "",
        "### Paired Wilcoxon on Eff at matched k (n=12)",
        "",
        "| contrast | RL Eff | anchor Eff | Δ | p |",
        "|---|---:|---:|---:|---:|",
        "| `rlda`(k=1) vs `sp_anchors`(k=1) | 0.682 | 0.620 | +0.062 | 0.092 ns |",
        "| `mada`(k=1) vs `sp_anchors`(k=1) | 0.695 | 0.620 | +0.075 | 0.064 ns |",
        "| `rlda`(k=1) vs `greedy_anchors`(k=1) | 0.682 | 0.575 | +0.107 | **0.027 \\*** |",
        "| `mada`(k=1) vs `greedy_anchors`(k=1) | 0.695 | 0.575 | +0.120 | **0.021 \\*** |",
        "| `rlda`(k=5) vs `sp_anchors`(k=5) | 0.744 | 0.741 | +0.003 | 0.970 ns |",
        "| `mada`(k=5) vs `sp_anchors`(k=5) | 0.762 | 0.741 | +0.021 | 0.791 ns |",
        "| `rlda`(k=5) vs `greedy_anchors`(k=5) | 0.744 | 0.687 | +0.057 | 0.424 ns |",
        "| `mada`(k=5) vs `greedy_anchors`(k=5) | 0.762 | 0.687 | +0.075 | 0.092 ns |",
        "| `rlda`(k=5) vs `sp_anchors`(k=20) | 0.744 | 0.751 | −0.006 | 0.791 ns |",
        "| `mada`(k=5) vs `sp_anchors`(k=20) | 0.762 | 0.751 | +0.011 | 0.970 ns |",
        "",
        "**Against `sp_anchors` at matched pool and matched k the arms are tied.** "
        "The only surviving significant cells are against `greedy_anchors` at k=1 "
        "— the weaker reducer at the smallest union size. Do not write that the "
        "method beats the classical-Anchors aggregations on Eff.",
        "",
        "### The cost claim is NOT currently supported — do not write it",
        "",
        "It was tempting to answer the lost quality margin with a cost margin: the "
        "anchor family needs 875,545 extraction queries at pool 20, against RLDA's "
        "reported 79,992 and MADA's 176,233. **That comparison cannot be made from "
        "the current instrumentation.**",
        "",
        "The anchor and CART figures are measurements: a counter wrapped around "
        "`SimpleClassifier.forward` reconciles them exactly (`cart` and "
        "`greedy_anchors` both delta +0 once reporting passes are included). "
        "`anchor-exp` genuinely pushes every perturbation row through `predict_fn`.",
        "",
        "**The RL figures are not measurements.** They are exact closed forms in "
        "the split sizes, identical across all five seeds and independent of the "
        "frame budget:",
        "",
        "| quantity | closed form | exact on |",
        "|---|---|---|",
        "| MADA `n_training_queries` | \\(|train| + |val|\\) | 12/12 datasets |",
        "| RLDA `n_training_queries` | \\(K \\times (|train| + |val|)\\) | 12/12 |",
        "| CART `n_blackbox_queries` | \\(|train| + |val|\\) | 12/12 |",
        "| MADA `n_blackbox_queries` | \\((12K + 3) \\times |val|\\) | 12/12, zero residual |",
        "| RLDA `n_blackbox_queries` | \\(\\approx 6K \\times |val|\\) + per-instance | 12/12 |",
        "",
        "A training-query count that does not move with the frame budget "
        "(folktables trains at 72k frames/agent, iris at 24k) and does not vary by "
        "seed is not counting anything that happens during training. Direct "
        "measurement confirms it: the environment makes **90 classifier calls at "
        "construction and 0 across 3,200 env steps**.",
        "",
        "And the reported extraction figures do not match a call counter either — "
        "in the *opposite* direction. Full inference on iris seed 42 under the "
        "class-level forward counter:",
        "",
        "| arm | actual rows through f̂ | reported `n_blackbox_queries` |",
        "|---|---:|---:|",
        "| RLDA | **930** | 600 |",
        "| MADA | **1,710** | 1,170 |",
        "",
        "So the RL number is neither the true call count nor a coherent "
        "distinct-rows count. Scoring 12–51 candidate boxes against the validation "
        "split does not need 12–51 passes through f̂ — it needs f̂(val) once, after "
        "which membership is a comparison on stored predictions. The RL arms are "
        "billed for cached reads; CART is billed one pass for the identical "
        "operation.",
        "",
        "**This has now been done** — see § \"Cost on the marginal-call unit\", "
        "which supersedes the numbers above. Measured on that unit: the RL arms "
        "cost **2.00 marginal rows per explanation** (a conservative upper bound; "
        "the true value is ~0) against `greedy_anchors`' **4,506–69,417**, so the "
        "amortisation claim against classical Anchors is real and three to four "
        "orders of magnitude. **CART's marginal cost is 0.00 on every dataset**, "
        "so there is still no cost claim against CART in either direction. Quote "
        "the marginal-unit table, not the extraction totals.",
        "",
    ]
    return L


def paper_frontier_section() -> List[str]:
    """Fidelity–coverage curves from the k-sweep, and what they settle.

    Eff and Cov_τ are two points on a trade-off. k (rules per class union) moves
    each method along its own curve, so the comparison becomes curve-vs-curve
    and stops depending on where a threshold was put. Source:
    `python -m revision.k_sweep --out runs/k_sweep`.
    """
    path = os.path.join(REPO, "runs", "k_sweep", "curves.json")
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        curves = json.load(fh)
    ks = ["1", "2", "3", "5"]
    order = ["mada", "rlda", "cart", "sp_anchors", "greedy_anchors", "random_search"]

    L = [
        "",
        "## Fidelity–coverage frontier (k-sweep)",
        "",
        "**k=1 is the paper lock**; k∈{2,3,5} re-runs selection only (no "
        "retraining, no new rules) and is stored in `runs/k_sweep/`. Each cell is "
        "the mean over 12 datasets of the 5-seed means, so every column is the "
        "same 60 runs re-selected at a different union size. 1,080 cells, no "
        "failures.",
        "",
        "| method | " + " | ".join(f"k={k} Fid / Cov / Eff" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for m in order:
        cells = []
        for k in ks:
            c = (curves.get(m) or {}).get(k)
            cells.append(
                f"{c['fid']:.3f} / {c['cov']:.3f} / **{c['eff']:.3f}**" if c else "—"
            )
        L.append(f"| `{m}` | " + " | ".join(cells) + " |")

    L += [
        "",
        "### What the sweep settles",
        "",
        "**1. The Eff ranking is stable in k — but only against a pool-5 anchor "
        "family.** At the paper's pool the RL arms beat `greedy_anchors` at "
        "every k and MADA beats `sp_anchors` at every k. **That result does not "
        "survive matching the pool** (§ \"Matched-pool comparison\"): at "
        "`budget_per_class=20` and matched k the arms are tied with "
        "`sp_anchors`. Coverage rises with k for every method while fidelity is "
        "near-flat, so k is not the confound — pool size is.",
        "",
        "**2. MADA vs RLDA is ns at every k** — consistent with the headline "
        "test. Do not claim a difference between the arms.",
        "",
        "**3. The coverage ceiling was also a pool artifact.** At pool 5 "
        "`sp_anchors` tops out at Cov 0.730 and `greedy_anchors` at 0.648, "
        "against MADA 0.849 / RLDA 0.840 — but at pool 20 `sp_anchors` reaches "
        "0.859. Do not claim the policies extend a frontier the anchor family "
        "cannot reach; they reach a comparable frontier far more cheaply "
        "(§ \"Matched-pool comparison\").",
        "",
        "**4. CART is a real competitor at k≥2, not a construction artifact — "
        "state this plainly.** At k=1 CART is a bare K-leaf partition (Cov 0.950, "
        "Fid 0.849) and its Eff win is the partition doing what a partition does. "
        "At k=3 it is Fid 0.917 / Cov 0.913, and per dataset it **Pareto-dominates "
        "both RL arms (higher Fid *and* higher Cov) on 6 of 12 datasets, while "
        "neither RL arm dominates CART on any**. The construction argument does "
        "not cover this. The honest claim against CART is interpretability and "
        "the per-class rule form, not fidelity or coverage.",
        "",
        "**5. Weak Pareto results against the anchor family, in MADA's favour.** "
        "At matched k=5, MADA dominates `sp_anchors` on 3/12 datasets and "
        "`greedy_anchors` on 3/12, losing on 0. RLDA is mixed (0/12 and 1/12, "
        "losing 1/12 and 1/12). Most pairs are genuine trades — report the "
        "frontier, not a winner.",
        "",
    ]
    return L


def paper_metric_choice_section() -> List[str]:
    """Why Eff is primary and Cov_τ is not. Settled; kept for the reviewer."""
    return [
        "",
        "## Choosing the primary metric — settled: Eff",
        "",
        "**Eff = Fid×Cov is the primary.** Cov_τ was trialled as the primary and "
        "rejected. Both are computed and reported; this section records why, "
        "because a reviewer will ask and because the trial produced a finding "
        "worth keeping.",
        "",
        "| | **Eff** = Fid×Cov | **Cov_τ** (global gate) | **Cov_τ** (per-class gate) |",
        "|---|---|---|---|",
        "| Question | rule set as a replacement classifier, abstention = miss | coverage of a rule set that clears the precision floor | coverage of the class rules that individually clear it |",
        "| Dispersion over 12 datasets | sd 0.13–0.25 | sd 0.22–0.40 | between |",
        "| Effective n (Wilcoxon) | 12 on every contrast | 10–11 (ties at 0 dropped) | 12 |",
        "| Significant contrasts (Holm) | **9/9** | 1/9 | 1/9 |",
        "| Stable in τ | n/a — no threshold | **no** — winner flips 3× over τ∈[0.80,0.95] | better, still flips 2× |",
        "| Stable in k | **yes** — ordering holds at k=1,2,3,5 | not tested | not tested |",
        "| Known distortion | a K-leaf partition gets Cov≈1 free (see CART below) | one weak class zeroes a whole dataset; a method 0.002 under the floor loses everything | a method with 0.21 less raw coverage still scores higher \"coverage\" |",
        "",
        "**Why Cov_τ is not the primary.** Three reasons, none of them the "
        "win/loss record:",
        "",
        "1. **It is unstable in τ.** The winner changes 2–3 times across "
        "τ∈[0.80, 0.95]. Publishing τ=0.90 alone when τ=0.85 reverses the "
        "ranking is not defensible.",
        "2. **τ_P = 0.90 is the RL arms' own training target.** They are "
        "optimised to sit at the floor, so their class unions cluster just "
        "under it — on folktables an RLDA rule set keeps 0.4 of 2 class unions "
        "above 0.90, against `greedy_anchors`' 2.0 of 2. Gating at the value "
        "one family was tuned to and the other was not is circular.",
        "3. **It resolves almost nothing at n=12.** With ~2× Eff's dispersion "
        "it leaves 1 of 9 contrasts significant under either gate. That is low "
        "power, not evidence of equivalence.",
        "",
        "**What the trial found, and what to keep.** Under *any* gated variant "
        "at *any* τ the RL arms do not lead — `greedy_anchors` is ahead at "
        "every τ ≥ 0.88, CART below that. That is a real property of where the "
        "methods sit on the trade-off, not a metric bug, and it belongs in the "
        "paper as a limitation: **the RL arms' class unions frequently fall "
        "just short of the precision floor they were trained to.** Report "
        "Cov_τ with the classes-kept diagnostic; do not report it as a "
        "headline either way.",
        "",
        "**Why Eff survives the same scrutiny.** It has no threshold to tune, "
        "and the k-sweep (§ frontier) shows its ordering is stable in k. (Stability "
        "in k is not the same as a win: at the paper's anchor pool of 5 the RL "
        "arms beat `greedy_anchors` and `sp_anchors` at every k, but at a matched "
        "pool of 20 only the `greedy_anchors` k=1 cells survive — "
        "§ \"Matched-pool comparison\".) Its one known distortion is CART, and that is "
        "now measured rather than argued: at k=1 CART's win is a K-leaf "
        "partition getting Cov≈1 free, but at k≥2 CART genuinely "
        "Pareto-dominates both RL arms on 6 of 12 datasets. **Do not claim "
        "fidelity or coverage against CART at any k** — claim the per-class "
        "rule form.",
        "",
    ]


def paper_breakeven_section() -> List[str]:
    """Where the fixed construction cost is repaid, in explanations.

    Reads `paper/figures/break_even.json` (written by `paper/make_figures.py`).
    This is the number the amortisation claim actually reduces to: serving is
    free for both arms, so the only question is how many explanations it takes
    to repay training + extraction against a per-instance baseline.
    """
    path = os.path.join(REPO, "paper", "figures", "break_even.json")
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        data = json.load(fh)
    rows: Dict[str, Dict[str, Any]] = {}
    for x in data.get("crossovers") or []:
        if x.get("baseline") != "greedy_anchors":
            continue
        if x.get("method") in ("rlda", "mada"):
            rows.setdefault(x["dataset"], {})[x["method"]] = x.get("n_break_even")
    if not rows:
        return []
    L = [
        "",
        "### Break-even: explanations needed to repay construction",
        "",
        "\\(N^*\\) such that `training + extraction` equals \\(N^*\\) explanations at "
        "the anchor family's per-instance cost. Serving is 0 marginal queries for "
        "both arms, so this is the whole amortisation claim in one number. "
        "Generated by `python paper/make_figures.py`; baseline is "
        "`greedy_anchors` (`sp_anchors` shares its query cost).",
        "",
        "| dataset | RLDA \\(N^*\\) | MADA \\(N^*\\) | test rows |",
        "|---|---:|---:|---:|",
    ]
    sizes = {r["dataset"]: r for r in load_eda()}
    for ds in EDA_DATASETS:
        rec = rows.get(ds)
        if not rec:
            continue
        n_test = (sizes.get(ds) or {}).get("n")
        n_test_s = f"{round(n_test/5):,}" if n_test else "—"   # 60/20/20 split
        L.append(
            f"| `{ds}` | {f(rec.get('rlda'), 1)} | {f(rec.get('mada'), 1)} | {n_test_s} |"
        )
    L += [
        "",
        "**⚠ Superseded — do not quote this table.** These break-even figures "
        "divide an inflated fixed cost (the RL extraction totals, which are "
        "closed forms in the split sizes rather than measurements) by a "
        "per-instance cost measured on a different unit. The corrected version is "
        "§ \"Cost on the marginal-call unit\": median N* = **0.11** explanations, "
        "worst case ~1.7, against the 25–1,013 implied here. Kept only so the "
        "correction is auditable.",
        "",
    ]
    return L


def paper_extraction_cost_section(cells) -> List[str]:
    """One-time construction cost. The amortisation claim is incomplete without it.

    Serving one more explanation is free (§ Track B); that only means something
    next to what was paid once. Three categories, kept apart: training queries,
    extraction (generation + validation selection), and reporting
    (instrumentation, excluded).
    """
    L = [
        "",
        "### Extraction and training cost (5-seed)",
        "",
        "Black-box calls to **build** the rule set, mean ± sd over seeds. "
        "`extraction` is generation + validation selection "
        "(`n_blackbox_queries`); `training` is classifier calls during policy "
        "training (`n_training_queries`, RL arms only); reporting calls on "
        "the test split are instrumentation and excluded. This is the **fixed** "
        "side of the amortisation claim — quote it next to the per-instance "
        "serving cost, never on its own.",
        "",
        "| dataset | method | n | extraction queries | of which reusable table | training queries |",
        "|---|---|---:|---|---|---|",
    ]
    by: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for c in cells:
        if c["method"] not in PAPER_COMPARE_METHODS:
            continue
        q = c["data"].get("queries") or {}
        rec = by.setdefault(c["dataset"], {}).setdefault(
            c["method"], {"ex": [], "tr": [], "ref": []}
        )
        if q.get("n_blackbox_queries") is not None:
            rec["ex"].append(float(q["n_blackbox_queries"]))
        if q.get("n_training_queries") is not None:
            rec["tr"].append(float(q["n_training_queries"]))
        if q.get("n_reference_table_queries") is not None:
            rec["ref"].append(float(q["n_reference_table_queries"]))
    for ds in DATASETS:
        for method in PAPER_COMPARE_METHODS:
            rec = (by.get(ds) or {}).get(method)
            if not rec or not rec["ex"]:
                continue
            L.append(
                f"| `{ds}` | {method} | {len(rec['ex'])} | "
                f"{_msd_n(rec['ex'], 0)} | "
                f"{_msd_n(rec['ref'], 0) if rec['ref'] else '—'} | "
                f"{_msd_n(rec['tr'], 0) if rec['tr'] else '—'} |"
            )
    L += [
        "",
        "**The old \u201cRLDA costs ~5\u00d7 MADA to extract\u201d line is resolved and was "
        "an artifact. Do not repeat it.** Two things caused it, both structural:",
        "",
        "1. **A hoisting difference.** MADA scores the recompute split once, "
        "outside the per-instance loop. The single-agent path had the same block "
        "*inside* the loop, so every instance re-ran the classifier over the whole "
        "split. On iris that was 20 instances \u00d7 3 class shards \u00d7 90 train rows "
        "= 5,400 queries for **90 distinct rows** \u2014 5,670 of the 5,730 reported "
        "(99%) were refills. Fixed by memoising per distinct split; the extracted "
        "anchors are identical as a multiset before and after, so no metric moved.",
        "2. **Class-parallel shards.** RLDA trains and extracts one process per "
        "class, so the process-wide prediction cache cannot dedupe across them and "
        "the per-shard totals are summed. Training shows this exactly: RLDA "
        "= K \u00d7 (|train|+|val|), MADA = 1 \u00d7 the same rows. The **reusable table** "
        "column is that dedupable share; the remainder is genuine per-instance "
        "work (40\u2013100 queries per dataset, i.e. one prediction per explained "
        "instance).",
        "",
        "**Corrected, RLDA extraction is ~0.45\u00d7 MADA on every dataset** (range "
        "0.44\u20130.51), which is what the agent counts predict: RLDA runs K policies "
        "against MADA's K\u00b7M = 3K agents. The ratio is stable enough to state "
        "plainly, but it is a property of the implementations' parallel layout as "
        "much as of the algorithms \u2014 do not sell it as an efficiency result.",
        "",
        "**⚠ The RL columns in this table are not measurements — do not quote them.** "
        "They are exact closed forms in the split sizes (MADA training = |train|+|val|; "
        "RLDA training = K×(|train|+|val|); MADA extraction = (12K+3)×|val| with zero "
        "residual), identical across all five seeds and independent of the frame "
        "budget. A class-level forward counter disagrees with them in the other "
        "direction (iris: RLDA 930 actual vs 600 reported; MADA 1,710 vs 1,170). "
        "The baseline columns *are* measurements and reconcile exactly. See "
        "§ \"Matched-pool comparison\" for the unit that has to replace this.",
        "",
        "**Training performs no new black-box evaluations at all.** A counter wrapped "
        "around the classifier's `forward` sees 90 rows (= |D_train|) at env "
        "construction on iris and **0** during 50, 400 or 3,200 env steps. "
        "`n_training_queries` is the reference-table build, which is why it "
        "tracks dataset size rather than the frame budget. Say it that way — it "
        "is a stronger and more accurate statement than \"training queries\".",
        "",
        "**Do not write that CART is cheaper to build.** That concession rested "
        "on inflated RL totals from per-shard duplicate table builds. The "
        "settled version is in § \"Cost on the marginal-call unit\": per "
        "explanation, CART is 0.0, RLDA ~1 and MADA ~3 marginal rows, so neither "
        "side is meaningfully cheaper and the real difference against CART is "
        "wall-clock training time, not queries. (An intermediate framing — "
        "\"distinct evaluations ≈ |train|+|val| for all three\" — is superseded: "
        "it folded the shared table build into per-method cost, which is the "
        "conflation the marginal unit exists to remove.)",
        "",
        "**Remaining caveat.** MADA's training-query count is a **lower bound**: "
        "the TorchRL collector holds environment copies whose cache hits are not "
        "attributed back.",
        "",
    ]
    return L


def paper_fiveseed_detail_section() -> List[str]:
    """Per-dataset aggregates + per-class unions for RL arms and 5-seed baselines."""
    cells = load_paper_track_a_cells()
    n_bl = count_paper_baselines()
    L: List[str] = [
        "",
        "### 5-seed baselines (same classifiers, k=1, τ=0.90/0.10)",
        "",
        f"`runs/paper_fiveseed_overlap075/results/baselines/` — **{n_bl}/220** JSONs "
        "(11 datasets × 5 seeds × 4 methods). Same DNN files as Track A. "
        "Do not mix with `low_budget_seed42` baseline tables.",
        "",
        "Seed-level rows are the mean across datasets present for that seed "
        "(baselines: 11/11 on every seed).",
        "",
        "| seed | method | n | Fid | Pur | Cov | Cov_τ | Conf | Eff |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_seed_method: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for c in cells:
        if c["method"] not in {"cart", "random_search", "sp_anchors", "greedy_anchors"}:
            continue
        gr = c["data"].get("global_ruleset") or {}
        rec = by_seed_method.setdefault(c["seed"], {}).setdefault(c["method"], {
            "fid": [], "pur": [], "cov": [], "cov_tau": [], "conf": [], "eff": [],
        })
        rec["fid"].append(gr.get("global_fidelity"))
        rec["pur"].append(gr.get("global_purity"))
        rec["cov"].append(gr.get("coverage"))
        rec["cov_tau"].append(track_a_cov_tau(gr))
        rec["conf"].append(gr.get("conflict_rate"))
        rec["eff"].append(track_a_eff(gr))
    for seed in ("42", "43", "44", "45", "46"):
        for method in ("greedy_anchors", "sp_anchors", "cart", "random_search"):
            rec = (by_seed_method.get(seed) or {}).get(method)
            if not rec:
                L.append(f"| {seed} | {method} | — | — | — | — | — | — | — |")
                continue
            n = len(rec["cov"])
            L.append(
                f"| {seed} | {method} | {n} | "
                f"{f(_mean(rec['fid']))} | {f(_mean(rec['pur']))} | "
                f"{f(_mean(rec['cov']))} | {f(_mean(rec['cov_tau']))} | "
                f"{f(_mean(rec['conf']))} | {f(_mean(rec['eff']))} |"
            )
    L += ["", "Mean ± sd over seeds (each seed is a mean across its datasets):", ""]
    for method in ("greedy_anchors", "sp_anchors", "cart", "random_search"):
        seed_eff = []
        seed_fid = []
        seed_cov = []
        seed_ct = []
        seed_conf = []
        for seed in ("42", "43", "44", "45", "46"):
            rec = (by_seed_method.get(seed) or {}).get(method)
            if not rec or not _finite(rec["cov"]):
                continue
            for dest, src in (
                (seed_fid, rec["fid"]),
                (seed_cov, rec["cov"]),
                (seed_ct, rec["cov_tau"]),
                (seed_conf, rec["conf"]),
                (seed_eff, rec["eff"]),
            ):
                mu = _mean(src)
                if mu is not None:
                    dest.append(mu)
        L.append(
            f"- **{method}** ({len(seed_eff)} seeds): "
            f"Fid {_msd(seed_fid)}, Cov {_msd(seed_cov)}, "
            f"Cov_τ {_msd(seed_ct)}, Conf {_msd(seed_conf)}, Eff {_msd(seed_eff)}"
        )

    L += [
        "",
        "### Per-dataset Track A (mean ± sd over seeds)",
        "",
        "Global rule-set metrics on \(D_{\\mathrm{test}}\). **Cov** here is Track A "
        "(any-class fire rate). **Cov_τ** is Cov counted only when that cell's Fid "
        "≥ 0.90. **n** is seeds with a JSON.",
        "",
        "| dataset | method | n | Fid | Pur | Cov | Cov_τ | Conf | Eff |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    by_ds_m: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for c in cells:
        if c["method"] not in PAPER_COMPARE_METHODS:
            continue
        gr = c["data"].get("global_ruleset") or {}
        rec = by_ds_m.setdefault(c["dataset"], {}).setdefault(c["method"], {
            "fid": [], "pur": [], "cov": [], "cov_tau": [], "conf": [], "eff": [],
        })
        rec["fid"].append(gr.get("global_fidelity"))
        rec["pur"].append(gr.get("global_purity"))
        rec["cov"].append(gr.get("coverage"))
        rec["cov_tau"].append(track_a_cov_tau(gr))
        rec["conf"].append(gr.get("conflict_rate"))
        rec["eff"].append(track_a_eff(gr))
    for ds in DATASETS:
        for method in PAPER_COMPARE_METHODS:
            rec = (by_ds_m.get(ds) or {}).get(method)
            if not rec:
                L.append(f"| `{ds}` | {method} | 0 | — | — | — | — | — | — |")
                continue
            n = len(rec["cov"])
            L.append(
                f"| `{ds}` | {method} | {n} | "
                f"{_msd(_finite(rec['fid']))} | {_msd(_finite(rec['pur']))} | "
                f"{_msd(_finite(rec['cov']))} | {_msd(_finite(rec['cov_tau']))} | "
                f"{_msd(_finite(rec['conf']))} | {_msd(_finite(rec['eff']))} |"
            )

    L += [
        "",
        "### Per-class unions (mean ± sd over seeds)",
        "",
        "One row per class union. **Fid / Pur** are \(P(\\hat y=c\\mid x\\in B_c)\) and "
        "\(P(y=c\\mid x\\in B_c)\). **Cov_c** is class-conditional "
        "\(P(x\\in B_c\\mid y=c)\) — not Track A Cov. "
        "**Cov_{τ,c}** is Cov_c counted only when that union's Fid ≥ 0.90. "
        "Use these rows to write majority/minority stories (mammography, sick, "
        "housing bins, iris setosa). Then aggregate to the per-dataset table above "
        "for the headline.",
        "",
        "| dataset | class | method | n | Fid | Pur | Cov_c | Cov_{τ,c} | n_covered |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    by_cls: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
    for c in cells:
        if c["method"] not in PAPER_COMPARE_METHODS:
            continue
        for cls, blk in (c["data"].get("per_class") or {}).items():
            u = (blk or {}).get("union") or {}
            rec = (
                by_cls.setdefault(c["dataset"], {})
                .setdefault(str(cls), {})
                .setdefault(c["method"], {
                    "fid": [], "pur": [], "cov": [], "cov_tau": [], "n": [],
                })
            )
            rec["fid"].append(u.get("fidelity"))
            rec["pur"].append(u.get("purity"))
            rec["cov"].append(u.get("coverage"))
            rec["cov_tau"].append(class_cov_tau(u.get("fidelity"), u.get("coverage")))
            rec["n"].append(u.get("n_covered"))
    for ds in DATASETS:
        classes = sorted((by_cls.get(ds) or {}).keys())
        for cls in classes:
            for method in PAPER_COMPARE_METHODS:
                rec = ((by_cls.get(ds) or {}).get(cls) or {}).get(method)
                if not rec:
                    L.append(f"| `{ds}` | `{cls}` | {method} | 0 | — | — | — | — | — |")
                    continue
                n = max(len(_finite(rec["fid"])), len(_finite(rec["cov"])))
                n_cov = _finite(rec["n"])
                n_cov_s = _msd(n_cov) if n_cov else "—"
                L.append(
                    f"| `{ds}` | `{cls}` | {method} | {n} | "
                    f"{_msd(_finite(rec['fid']))} | {_msd(_finite(rec['pur']))} | "
                    f"{_msd(_finite(rec['cov']))} | {_msd(_finite(rec['cov_tau']))} | "
                    f"{n_cov_s} |"
                )

    L += [
        "",
        "### Compactness and episode success (5-seed, RL only)",
        "",
        "Present in every Track A JSON; tabulated here so the manuscript does not "
        "have to invent them. **Active feats** = mean constrained features per "
        "selected rule (lower = more compact). **Success** = episodes with "
        "Fid ≥ τ_P and Cov_c ≥ τ_C over episodes attempted. Baselines have no "
        "episodes (`—`). Values are mean ± sd over seeds, then the bullet is the "
        "mean of these dataset-means. WyoDOT has its own compactness/success "
        "table in the WyoDOT section — add it before quoting a 12-dataset figure.",
        "",
        "| dataset | method | n | rules/class | mean active feats | success rate |",
        "|---|---|---:|---|---|---|",
    ]
    by_comp: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for c in cells:
        if c["method"] not in {"mada", "rlda"}:
            continue
        compact = c["data"].get("compactness") or {}
        succ = c["data"].get("success_rate") or {}
        rec = by_comp.setdefault(c["dataset"], {}).setdefault(c["method"], {
            "rules": [], "active": [], "succ": [],
        })
        rec["rules"].append(compact.get("mean_rules_per_class"))
        rec["active"].append(compact.get("mean_active_features"))
        rec["succ"].append(succ.get("success_rate"))
    ds_means: Dict[str, Dict[str, List[float]]] = {
        "mada": {"active": [], "succ": []},
        "rlda": {"active": [], "succ": []},
    }
    for ds in DATASETS:
        for method in ("mada", "rlda"):
            rec = (by_comp.get(ds) or {}).get(method)
            if not rec:
                L.append(f"| `{ds}` | {method} | 0 | — | — | — |")
                continue
            n = len(rec["active"])
            L.append(
                f"| `{ds}` | {method} | {n} | "
                f"{_msd(_finite(rec['rules']))} | {_msd(_finite(rec['active']))} | "
                f"{_msd(_finite(rec['succ']))} |"
            )
            act = _finite(rec["active"])
            sr = _finite(rec["succ"])
            if act:
                ds_means[method]["active"].append(sum(act) / len(act))
            if sr:
                ds_means[method]["succ"].append(sum(sr) / len(sr))
    L += ["", f"Mean of these {len(DATASETS)} dataset-means "
          "(WyoDOT's row is in the WyoDOT section — add it before quoting a "
          "12-dataset figure):", ""]
    for method in ("mada", "rlda"):
        L.append(
            f"- **{method.upper()}**: mean active feats "
            f"{_msd(ds_means[method]['active'])}, success "
            f"{_msd(ds_means[method]['succ'])}"
        )
    L.append("")
    L += paper_extraction_cost_section(cells)
    L += paper_breakeven_section()
    L += paper_marginal_cost_section()
    L += paper_pool_section()
    L += paper_cart_position_section()
    L += paper_frontier_section()
    L += paper_metric_choice_section()
    return L


def _mean(vals: List[Any]) -> Optional[float]:
    xs = _finite(vals)
    if not xs:
        return None
    return sum(xs) / len(xs)


def _scoreboard_arm_means(path: str, arms: tuple) -> Dict[str, Dict[str, float]]:
    """Mean Fid/Cov/Conf/Eff per arm from a scoreboard.tsv."""
    acc: Dict[str, Dict[str, List[float]]] = {}
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            a = row["arm"]
            if row.get("fid") in ("", "MISSING"):
                continue
            fid, cov, conf = float(row["fid"]), float(row["cov"]), float(row["conf"])
            acc.setdefault(a, {}).setdefault("fid", []).append(fid)
            acc[a].setdefault("cov", []).append(cov)
            acc[a].setdefault("conf", []).append(conf)
            acc[a].setdefault("eff", []).append(fid * cov)
    return {
        a: {k: sum(v) / len(v) for k, v in acc[a].items()}
        for a in arms if a in acc
    }


def paper_protocol_header() -> List[str]:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    paper_done = count_track_a(PAPER_FIVESEED, "ddpg") + count_track_a(
        PAPER_FIVESEED, "maddpg"
    )
    return [
        "# Results for the paper rewrite",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`.",
        "",
        "This file is the **only** numeric source for the manuscript. It does **not** "
        "include the older full-budget cluster sweep (`runs/sweep_dnn`, `runs/sweep_rf`, "
        "original overlap reward) or RandomForest black-box runs.",
        "",
        "## Protocol",
        "",
        "| | |",
        "|---|---|",
        "| Arms | MADA (MADDPG), RLDA (DDPG) |",
        "| Black box | **DNN** — one `classifier.pth` per dataset, shared by every arm and baseline |",
        "| Datasets | **12, all complete:** iris, wine, breast_cancer, synthetic, "
        "housing, uci_credit, uci_adult, heloc, mammography, sick, "
        "folktables_income_CA_2018 (`runs/paper_fiveseed_overlap075/`) and "
        "`wyodot_kvdw_labeled` (`runs/wyodot_fiveseed_overlap075/dnn/`; do not run "
        "testbed). The 11-set tables below reflect the first tree only; the "
        "Wilcoxon, the EDA and the headline are over all 12. |",
        "| Dropped | covtype (OOM at inference). |",
        "| Out of scope | RandomForest and gradient-boosting black boxes — **this paper is DNN only**. |",
        "| Headline | `runs/paper_fiveseed_overlap075/` — overlap w=**0.75**, cross-class ON, "
        f"seeds 42–46 ({paper_done}/110 Track A JSONs; "
        f"{count_paper_baselines()}/220 baseline JSONs) |",
        "| Budget | MADA 24k frames/agent base; RLDA `mada_frames×3×n_classes` total steps |",
        "| Selection | validation split, greedy marginal-gain union (YAML pool k≤5; paper cells k=1); report on test |",
        "| τ_P / τ_C | 0.90 / 0.10 |",
        "| Primary metric | **Eff** = Fid×Cov — rule-set accuracy vs f̂, abstention counted as a miss. No threshold, and the k-sweep shows the ordering holds at k=1,2,3,5. |",
        "| Reported alongside | **Cov_τ** = Cov·1[Fid ≥ 0.90]. Trialled as primary and rejected: ranking flips 2–3× over τ∈[0.80,0.95], τ_P is the RL arms' own training target, 1/9 contrasts significant. Kept as a diagnostic. |",
        "| ⚠ Comparison caveat | The 5-seed tables below use `budget_per_class=5` for the anchor family against an RL candidate pool of ~15–19. **At a matched pool the Eff advantage over `sp_anchors` disappears** — see § \"Matched-pool comparison against classical Anchors\" before quoting any of it. |",
        "",
        "Seed-42 ablations decompose mechanism only. **Track B** uses the same "
        f"paper-lock checkpoints as Track A (`paper_fiveseed_overlap075`, "
        f"{count_paper_track_b()}/110 instance JSONs; seed 42 imported from "
        "`mada_overlap075_seed42` / `low_budget_seed42_conflict_align`). "
        "Do **not** cite original-reward `low_budget_seed42` instance tables. "
        "**Not in the paper:** original-reward Track A grids, "
        "τ_C=0.05 re-eval, smoke repeats, full-budget `conflict_align_sweep_dnn` "
        "(cancelled), per-dataset SAC/MASAC tables, MASAC as a positive conflict result.",
        "",
    ] + paper_blackbox_section() + paper_budget_section() + paper_baselines_section() \
      + paper_metrics_section()


def paper_blackbox_section() -> List[str]:
    """The model being explained. A reviewer cannot check a fidelity claim without it."""
    return [
        "## The black box being explained",
        "",
        "Every arm and every baseline in this document loads the **same** "
        "`classifier.pth` per (dataset, seed) — the explanation target is fixed "
        "before any rule is generated. Built by "
        "`TabularDatasetLoader.create_classifier` / `train_classifier` "
        "(`BenchMARL/tabular_datasets.py`) and fitted by "
        "`revision/fit_shared_classifier.py`.",
        "",
        "MLP on the StandardScaler’d features, each hidden layer "
        "Linear → BatchNorm → ReLU → Dropout, linear head to \\(K\\) logits. "
        "Width and dropout are picked from the training-split size:",
        "",
        "| train rows | hidden sizes | dropout |",
        "|---|---|---:|",
        "| ≤ 5,000 | 256 → 256 → 128 | 0.3 |",
        "| 5,001 – 10,000 | 256 → 256 → 256 → 128 | 0.3 |",
        "| > 10,000 | 1024 → 1024 → 512 | 0.1 |",
        "",
        "Dropout drops to 0.1 on the large branch because the model was "
        "*under*fitting there (housing, 150 epochs, seed 42: train 0.7245 / test "
        "0.7209 at dropout 0.3 — the regularisation was pure cost).",
        "",
        "Optimiser Adam, lr 1e-3, weight decay 1e-4, batch 256, up to 500 epochs "
        "with early stopping on validation loss (patience 10) and a LR scheduler. "
        "Dropout and BatchNorm are in `eval()` mode for every query counted "
        "anywhere in this document, so \\(\\hat f\\) is deterministic.",
        "",
        "Per-dataset accuracies: § Scale and black-box accuracy. `random_forest` "
        "and `gradient_boosting` exist in the loader but **no result in this "
        "document uses them** — the paper is DNN-only.",
        "",
    ]


def paper_budget_section() -> List[str]:
    """Per-dataset training budget. '24k–72k' is not reproducible; this is."""
    return [
        "## Training budget per dataset",
        "",
        "MADA frames/agent scale with the dataset’s `ma_frames` entry in "
        "`revision/run_mada_pipeline.py`, times `24k/360k`, floored to a multiple "
        "of the 4,800-frame evaluation interval. RLDA gets "
        "`mada_frames × 3 × K` total environment steps — matched to MADA’s "
        "\\(K\\cdot M\\) agents at \\(M=3\\), so neither arm is handed a budget "
        "advantage. Source: `revision/run_overlap075_fiveseed.py`.",
        "",
        "| dataset | K | MADA frames/agent | RLDA total steps |",
        "|---|---:|---:|---:|",
        "| `iris` | 3 | 24,000 | 216,000 |",
        "| `wine` | 3 | 24,000 | 216,000 |",
        "| `synthetic` | 2 | 24,000 | 144,000 |",
        "| `sick` | 2 | 24,000 | 144,000 |",
        "| `breast_cancer` | 2 | 24,000 | 144,000 |",
        "| `mammography` | 2 | 48,000 | 288,000 |",
        "| `heloc` | 2 | 48,000 | 288,000 |",
        "| `uci_credit` | 2 | 48,000 | 288,000 |",
        "| `uci_adult` | 2 | 48,000 | 288,000 |",
        "| `housing` | 4 | 48,000 | 576,000 |",
        "| `wyodot_kvdw_labeled` | 5 | 48,000 | 720,000 |",
        "| `folktables_income_CA_2018` | 2 | 72,000 | 432,000 |",
        "",
        "Shared training hyperparameters (`BenchMARL/conf/base_experiment.yaml`): "
        "lr 5e-5, γ 0.95 (**must** equal the env `discount`, or Ng shaping stops "
        "being policy-invariant), collected frames/batch 4,800, 12 envs/worker, "
        "1,000 optimiser steps per batch, train batch 2,048, exploration annealed "
        "0.8 → 0.01, MLP width 256.",
        "",
    ]


def paper_baselines_section() -> List[str]:
    """Exactly how the four baselines are built. The settings are not the file defaults."""
    return [
        "## Baseline implementations (`revision/baselines.py`)",
        "",
        "**The paper cells do not use the module defaults.** Both sweep launchers "
        "(`revision/run_paper_fiveseed_baselines.py`, "
        "`revision/run_wyodot_fiveseed_baselines.py`) pass "
        "`--k 1 --tau_p 0.90 --tau_c 0.10 --budget_per_class 5 --n_candidates 256`, "
        "against the argparse defaults of `k=5`, `tau_c=0.20`, "
        "`budget_per_class=20`, `n_candidates=512`. Quote the launcher values.",
        "",
        "All four share the RL arms’ pipeline: candidates are scored on "
        "\\(D_{\\mathrm{val}}\\), ranked by Wilson-LCB\\((\\mathrm{Fid})\\times"
        "(1+\\mathrm{Cov}_c)\\), selected by `select_topk_union` at \\(k=1\\) with "
        "`min_support=10`, then re-scored on \\(D_{\\mathrm{test}}\\) and emitted in "
        "the identical JSON schema. They load the same `classifier.pth` as the RL "
        "arms, so the explanation target is identical.",
        "",
        "### `sp_anchors` / `greedy_anchors` — classical Anchors, aggregated",
        "",
        "`anchor-exp`’s `AnchorTabularExplainer`, fitted on `X_train` in **original "
        "units**, called with `threshold = τ_P = 0.90`. For each class \\(c\\), "
        "**5 instances** (`budget_per_class`) are drawn without replacement from "
        "the \\(D_{\\mathrm{val}}\\) rows the black box predicts as \\(c\\), seeded "
        "by the run seed. Each returned predicate list is parsed back into an "
        "axis-aligned box initialised to the `X_train` min/max envelope.",
        "",
        "**A pool of 5 anchors per class is the main reason the anchor family "
        "reaches high Fid at low Cov — and it is why the published Eff margin "
        "does not survive.** At `budget_per_class=20` the same reducers reach "
        "Eff 0.751 / 0.691 and the RL advantage at matched k is not significant "
        "(§ \"Matched-pool comparison\"). The RL arms select from ~15–19 "
        "candidate boxes per class, so the paper cells were never pool-matched. "
        "Report pool size and k as separate knobs, and report both pool sizes.",
        "",
        "The two differ only in how the pool is reduced to a class rule set:",
        "",
        "- `greedy_anchors` — greedy set-cover: repeatedly add the rule adding the "
        "most **not-yet-covered class-\\(c\\) validation rows**, restricted to rules "
        "with \\(\\mathrm{Fid}\\ge\\tau_P\\). If no rule clears \\(\\tau_P\\) it falls "
        "back to the fidelity-sorted pool, so a class always gets a rule set.",
        "- `sp_anchors` — SP-LIME-style submodular pick: greedy on "
        "\\(\\mathrm{Fid}\\times\\#\\{\\text{newly covered class-}c\\text{ rows}\\}\\). "
        "This is the aggregation baseline Reviewer 2 asked for.",
        "",
        "**Predicate parsing (fixed; regenerate before quoting old numbers).** "
        "`anchor-exp` discretizes continuous features and emits two-sided bins "
        "`a < feat <= b` with the value on the *left* of the name. The parser "
        "matched only `feat <op> value`, so every such lower bound was dropped and "
        "the box stayed at the training minimum on that feature — **224 of 1,766 "
        "selected predicates (12.7%)** across the paper cells. The boxes were "
        "therefore strictly wider than the anchors they encode, which inflates Cov "
        "and depresses Fid for both anchor baselines. Fixed in "
        "`_apply_anchor_predicate`; regression test "
        "`tests/test_anchor_predicate_parse.py`; both anchor baselines regenerated "
        "on all 12 datasets × 5 seeds.",
        "",
        "### `cart` — surrogate tree on the model’s predictions",
        "",
        "`DecisionTreeClassifier(max_leaf_nodes=max(K, k·K), random_state=seed)` fit "
        "on \\(\\hat f\\)’s **predictions** over \\(D_{\\mathrm{train}}\\) in original "
        "units. Every root-to-leaf path becomes a box, labelled by the leaf’s "
        "majority prediction; leaves are then ranked and selected like any other "
        "candidate.",
        "",
        "**It is leaf-limited, not depth-limited, and at the paper’s \\(k=1\\) the "
        "limit is \\(K\\) — a \\(K\\)-leaf tree.** Three consequences are structural, "
        "not empirical, and must be stated wherever CART is compared: "
        "the leaves **partition** the feature space, so Cov ≈ 1 by construction; "
        "no point can fall in two leaves, so **Conf ≡ 0** by construction; and a "
        "\\(K\\)-leaf tree needs \\(K-1\\) splits, so its compactness (~1.3 features) "
        "is a property of the leaf budget. **Eff = Fid×Cov therefore collapses to "
        "Fid** for CART and is replacement accuracy, not an explanation score. "
        "CART is a strong coverage-first *surrogate* of \\(\\hat f\\); score it on "
        "Fid (how close the tree is to the DNN). Score RL / Anchors on "
        "**Cov_τ** (coverage that still clears τ_P=0.90). Do not rank CART above "
        "rule extraction on Eff.",
        "",
        "### `random_search` — the sanity floor",
        "",
        "**256 candidates per class** (`n_candidates`) in **unit space**: sample a "
        "seed row from the \\(D_{\\mathrm{val}}\\) rows predicted \\(c\\), draw a "
        "per-dimension width \\(\\sim U(0.05, 0.6)\\), centre the box on the seed "
        "row, clip to \\([0,1]\\), enforce a minimum width of 0.05. Every dimension "
        "is constrained, so its active-feature count is ≈ \\(d\\) (13.75 mean over the "
        "12 sets, exactly 5.0 on WyoDOT) — that is the control confirming the "
        "compactness column discriminates.",
        "",
        "**The anchor baselines are not seed-reproducible.** Our seed fixes which "
        "validation instances enter the pool, but `anchor-exp`’s own "
        "multi-armed-bandit sampling inside `explain_instance` is not seeded by it. "
        "Two identical re-runs of all 12 datasets × 5 seeds at fixed code moved "
        "`greedy_anchors` mean Eff by 0.014 and Cov by 0.019. Treat per-dataset "
        "anchor-family cells as carrying that extra variance on top of the "
        "seed-to-seed spread, and do not read a <0.02 Eff gap against them as real. "
        "The parser fix above is separable from this: it moved `greedy_anchors` Fid "
        "+0.023 against 0.003 run-to-run noise, while its Cov/Conf/Eff shifts were "
        "the expected sign but inside the noise band.",
        "",
        "### Query accounting for the baselines",
        "",
        "CART and `random_search` are charged one pass over train/val (they read "
        "cached \\(\\hat f\\) predictions, never perturbations). The anchor family is "
        "charged every row `anchor-exp` pushes through `predict_fn`, which is where "
        "the hundreds-to-tens-of-thousands per instance come from. Test-split "
        "scoring is counted separately as `n_reporting_queries` and is "
        "instrumentation, not a cost of producing explanations.",
        "",
    ]


def paper_metrics_section() -> List[str]:
    """Canonical metric definitions for the manuscript. Do not collapse the three coverages."""
    return [
        "## Metrics — definitions and equations",
        "",
        "Notation. Held-out test set \(D_{\\mathrm{test}}=\\{(x_i,y_i)\\}_{i=1}^{n}\). "
        "Black box \(\\hat f\), predictions \(\\hat y_i=\\hat f(x_i)\). "
        "A **box** \(B=\\{x:\\ell_j\\le x_j\\le u_j\\ \\forall j\\}\) is axis-aligned. "
        "All Track A numbers below are computed by `evaluate_ruleset_as_classifier` / "
        "`evaluate_box` in `utils/eval_harness.py` and `utils/metrics.py` on "
        "**\(D_{\\mathrm{test}}\)** after selection on \(D_{\\mathrm{val}}\).",
        "",
        "### Three different coverage numbers — never treat them as one column",
        "",
        "| Symbol | Where it appears | Probability space | Question it answers |",
        "|---|---|---|---|",
        "| **Cov** (Track A) | Main tables | \(x\\sim D_{\\mathrm{test}}\) | What fraction of **real test rows** does the **class-level rule set** fire on? |",
        "| **Cov_τ** (Track A) | Reported diagnostic | same | Cov counted only when Fid ≥ τ_P=0.90 |",
        "| **Cov\(_c\)** (per-rule) | `RULES.md`, ranking | \(x\\sim D_{\\mathrm{test}}\\mid y=c\) | What fraction of **class-\(c\) test rows** fall in this box? |",
        "| **Cov\(_{\\mathrm{test}}\)** (Track B, π) | Instance table, `π Cov_test` | \(x\\sim D_{\\mathrm{test}}\) | What fraction of **real test rows** fall in the **instance box** \(B_\\pi(x^*)\)? |",
        "| **Cov\(_D\)** (classical Anchors) | Instance table, `Anchors Cov_D` | \(z\\sim D(z)\\) (Ribeiro perturbation / train-resample) | What mass of the **perturbation neighbourhood** of \(x^*\) satisfies the predicate? |",
        "",
        "**Do not write “RLDA coverage 0.26 vs Anchors coverage 0.18” as a like-for-like win.** "
        "Those are Cov\(_{\\mathrm{test}}\) vs Cov\(_D\). A fair coverage comparison with "
        "classical Anchors is either (i) both on \(D(z)\) — π pert-Fid vs Anchors precision — "
        "or (ii) both as Track A rule-sets on \(D_{\\mathrm{test}}\) — RLDA/MADA vs "
        "`greedy_anchors` / `sp_anchors`. Track B Cov\(_{\\mathrm{test}}\) exists to show "
        "the instance box is **not a 1-row spike** (Reviewer 1 coverage collapse), not to "
        "beat Ribeiro coverage.",
        "",
        "### Track A — class-level rule set (primary paper tables)",
        "",
        "For each class \(c\), let \(B_c\) be the selected union. "
        "Fired(\(x\))=\\(\\{c:x\\in B_c\\}\\). Predictor: abstain if Fired is empty; if "
        "one class, emit it; if two or more, emit the fired class with highest union Fid "
        "(tie-break).",
        "",
        "\\[",
        "\\mathrm{Cov}=\\frac{1}{n}\\sum_i \\mathbf{1}\\big[\\lvert\\mathrm{Fired}(x_i)\\rvert\\ge 1\\big]"
        "=1-\\mathrm{Abstain}",
        "\\]",
        "\\[",
        "\\mathrm{Abstain}=\\frac{1}{n}\\sum_i \\mathbf{1}\\big[\\lvert\\mathrm{Fired}(x_i)\\rvert=0\\big],"
        "\\qquad "
        "\\mathrm{Conf}=\\frac{1}{n}\\sum_i \\mathbf{1}\\big[\\lvert\\mathrm{Fired}(x_i)\\rvert\\ge 2\\big]",
        "\\]",
        "\\[",
        "\\mathrm{Fid}=P\\big(\\mathrm{pred}(x)=\\hat y \\mid \\text{not abstain}\\big)"
        "=\\frac{n_{\\mathrm{fid\\,agree}}}{n_{\\mathrm{decided}}},\\qquad "
        "n_{\\mathrm{decided}}=n-n_{\\mathrm{abstain}}",
        "\\]",
        "\\[",
        "\\mathrm{Pur}=P\\big(\\mathrm{pred}(x)=y \\mid \\text{not abstain}\\big)"
        "=\\frac{n_{\\mathrm{pur\\,agree}}}{n_{\\mathrm{decided}}}",
        "\\]",
        "\\[",
        "\\mathrm{Eff}=\\mathrm{Fid}\\times\\mathrm{Cov}"
        "=\\frac{n_{\\mathrm{fid\\,agree}}}{n}",
        "\\]",
        "\\[",
        "\\mathrm{Cov}_\\tau="
        "\\mathrm{Cov}\\cdot\\mathbf{1}[\\mathrm{Fid}\\ge\\tau_P]"
        "\\qquad(\\tau_P=0.90)",
        "\\]",
        "",
        "**Cov_τ is a reported diagnostic, not the primary.** It is the test-set mass "
        "covered by a rule set that still meets the precision floor: if Fid < 0.90 "
        "that cell's coverage does not count — a 4-leaf housing tree at Fid 0.53 "
        "covering 99.9% of rows scores Cov_τ = 0. Computed per (dataset, seed) "
        "JSON, then averaged; do not threshold the seed-mean Fid. It is worth "
        "reporting because it exposes what Eff hides — the RL arms' class unions "
        "often sit just under the floor they were trained to — but see "
        "§ \"Choosing the primary metric\" for why it is not the headline.",
        "",
        "**Eff is replacement accuracy** against \\(\\hat f\\) with abstention counted "
        "as a miss. It is the right score if the rule set must always answer. For a "
        "\\(K\\)-leaf CART partition Cov ≈ 1, so Eff ≈ Fid: that column is how close "
        "the surrogate tree is to the DNN, not evidence that CART beat rule "
        "extraction. Conditional Fid alone is also not the primary: "
        "`random_search` often wins Fid at Cov 0.10–0.30.",
        "",
        "Conflict is a fraction of **all** test rows, not of covered rows. "
        "Cov and Abstain sum to 1. Conf ≤ Cov.",
        "",
        "### Per-rule / per-class union (what `RULES.md` prints as Cov)",
        "",
        "For a box (or class union) \(B\) targeting class \(c\):",
        "",
        "\\[",
        "\\mathrm{Fid}(B)=P(\\hat y=c\\mid x\\in B)=\\frac{\\#\\{x\\in B:\\hat y=c\\}}{\\#\\{x\\in B\\}}",
        "\\]",
        "\\[",
        "\\mathrm{Pur}(B)=P(y=c\\mid x\\in B)=\\frac{\\#\\{x\\in B:y=c\\}}{\\#\\{x\\in B\\}}",
        "\\]",
        "\\[",
        "\\mathrm{Cov}_c(B)=P(x\\in B\\mid y=c)=\\frac{\\#\\{x\\in B:y=c\\}}{\\#\\{y=c\\}}",
        "\\qquad "
        "\\mathrm{Cov}_{\\mathrm{marg}}(B)=P(x\\in B)=\\frac{\\#\\{x\\in B\\}}{n}",
        "\\]",
        "\\[",
        "\\mathrm{Cov}_{\\tau,c}(B)="
        "\\mathrm{Cov}_c(B)\\cdot\\mathbf{1}[\\mathrm{Fid}(B)\\ge\\tau_P]",
        "\\]",
        "",
        "Ranking uses Wilson LCB of Fid times \((1+\\mathrm{Cov}_c)\). "
        "A class union's Cov\(_c\) is **not** Track A Cov: Track A Cov is the "
        "union-over-classes test-set mass of “any rule fired.” "
        "Cov_{τ,c} zeros a class union that misses τ_P (WyoDOT Slush CART, "
        "adult CART >50K, housing middle bins).",
        "",
        "Wilson interval on Fid/Pur: `wilson_interval` in `utils/metrics.py`.",
        "",
        "### Track B — instance boxes vs classical Anchors",
        "",
        "For each scored test instance \(x^*\) with \(\\hat y^*=\\hat f(x^*)\):",
        "",
        "| Quantity | π (RLDA/MADA instance box \(B_\\pi\)) | Classical Anchors (predicate \(A\)) |",
        "|---|---|---|",
        "| **Precision / Fid on \(D(z)\\)** | `π pert-Fid` \(=P(\\hat f(z)=\\hat y^*\\mid z\\in B_\\pi,\\,z\\sim D(z\\mid A))\) with frozen CRN | `Anchors Prec_D` \(=\) `explainer.precision` (Ribeiro; same family) |",
        "| **Coverage on \(D(z)\\)** | not the Track B headline | **`Anchors Cov_D`** \(=P(A(z)=1\\mid z\\sim D)\) — perturbation/train-resample mass |",
        "| **Coverage on \(D_{\\mathrm{test}}\)** | **`π Cov_test`** \(=\\frac{1}{n}\\sum_i\\mathbf{1}[x_i\\in B_\\pi(x^*)]\) (marginal) | **not computed** in the current JSON (would require applying \(A\) to every test row) |",
        "| **Fid on \(D_{\\mathrm{test}}\)** | `π Fid` \(=P(\\hat y=\\hat y^*\\mid x\\in B_\\pi,\\,x\\sim D_{\\mathrm{test}})\) | — |",
        "| **Pur on \(D_{\\mathrm{test}}\)** | `π Pur` \(=P(y=\\hat y^*\\mid x\\in B_\\pi)\) | — |",
        "",
        "Ribeiro et al. (2018): an *anchor* \(A\) is a predicate with high precision on "
        "the perturbation distribution \(D(z\\mid A)\) around \(x^*\). Their coverage is "
        "how large that neighbourhood is under \(D\), **not** how many held-out rows "
        "the predicate covers. Tabular \(D\) resamples unconstrained features from "
        "the training set.",
        "",
        "**What Track B is allowed to claim.** (1) Collapse is gone: mean `π Cov_test` "
        "is tens of percent, not ~1 test row. (2) Like-for-like precision on \(D(z)\): "
        "`π pert-Fid` vs `Anchors Prec_D` — π usually trails (on-manifold vs off-manifold). "
        "(3) Cost: π serving queries vs Anchors queries per \(x^*\). "
        "**What it must not claim:** “π coverage exceeds Anchors coverage” using "
        "`π Cov_test` vs `Anchors Cov_D`.",
        "",
        "Baselines `greedy_anchors` / `sp_anchors` **are** Track A: they aggregate "
        "classical Anchors into a class-level rule set and are scored with the same "
        "Cov/Fid/Cov_τ/Eff on \(D_{\\mathrm{test}}\) as RLDA/MADA. Use Cov_τ for "
        "the explanation head-to-head; Eff only as the replacement reading.",
        "",
        "### Other reported quantities",
        "",
        "| Metric | Definition | Direction |",
        "|---|---|---|",
        "| **Cov_τ** | Track A coverage counted only when Fid ≥ τ_P | higher (reported diagnostic) |",
        "| **Success** | episodes with Fid ≥ τ_P and Cov\(_c\) ≥ τ_C / episodes attempted | higher (RL only; baselines `—`) |",
        "| **Active feats** | mean number of constrained features per selected rule | lower = more compact |",
        "| **Extraction queries** | black-box calls to **build** the rule set (generation + val selection), not training | lower |",
        "| **Serving queries** | black-box calls per instance **after** extraction | RLDA: 0; MADA: see Track B table; Anchors: hundreds to tens of thousands |",
        "",
        "Do **not** headline conditional Fid alone. Against **CART**, do not claim "
        "fidelity or coverage at all: at k=1 its Eff win is a K-leaf partition "
        "getting Cov≈1 free, and at k≥2 it genuinely Pareto-dominates the RL arms "
        "on half the datasets (§ frontier). The claim against CART is the "
        "per-class rule form.",
        "",
    ]


EDA_JSON = os.path.join(REPO, "docs", "dataset_eda.json")


def load_eda() -> List[Dict[str, Any]]:
    if not os.path.isfile(EDA_JSON):
        return []
    with open(EDA_JSON) as fh:
        return json.load(fh)


def _mix(row: Dict[str, Any]) -> str:
    fr = row.get("class_fracs") or []
    return "/".join(f"{100 * x:.1f}" for x in fr)


def paper_eda_section() -> List[str]:
    """Dataset EDA + how it explains Fid/Pur/conflict/weak rows."""
    rows = load_eda()
    L = [
        "",
        "## Dataset EDA (use this to explain the results)",
        "",
        "All **12** paper datasets, computed from the same loader path training "
        "uses (`utils.dataset_factory`, seed-42 three-way split). **Centroid-NN error** is the fraction of rows "
        "whose nearest class-mean (in min–max unit space) is the wrong class — a "
        "cheap proxy for class overlap. **Imb.** is majority/minority count. "
        "Fid/Pur/Conf/Eff below are seed-42 **paper lock** (MADA w=0.75, matched RLDA), "
        "not 5-seed means.",
        "",
        "| dataset | n | d | K | cat | class mix % | imb. | centroid-NN err | DNN test acc | 1−acc |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    by_name = {r["dataset"]: r for r in rows}
    for ds in EDA_DATASETS:
        r = by_name.get(ds)
        if r is None:
            continue
        acc = r.get("test_acc")
        one_m = (1.0 - acc) if isinstance(acc, (int, float)) else None
        L.append(
            f"| `{ds}` | {r['n']:,} | {r['d']} | {r['K']} | {r['n_cat']} | "
            f"{_mix(r)} | {r['imbalance']:.2f} | {r['nn_mean_error']:.3f} | "
            f"{f(acc)} | {f(one_m)} |"
        )
    L += [
        "",
        "| dataset | MADA Fid | Pur | gap | Conf | Eff | RLDA Fid | Pur | gap | Conf | Eff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in EDA_DATASETS:
        r = by_name.get(ds)
        if r is None:
            continue
        m, a = r.get("mada") or {}, r.get("rlda") or {}
        L.append(
            f"| `{ds}` | {f(m.get('fid'))} | {f(m.get('pur'))} | {f(m.get('gap'))} | "
            f"{f(m.get('conf'))} | {f(m.get('eff'))} | {f(a.get('fid'))} | "
            f"{f(a.get('pur'))} | {f(a.get('gap'))} | {f(a.get('conf'))} | "
            f"{f(a.get('eff'))} |"
        )
    L += [
        "",
        "What each dataset is, and what that implies for the tables above:",
        "",
    ]
    for ds in DATASETS:
        r = by_name.get(ds)
        if r is None:
            continue
        L.append(f"- **`{ds}`** — {r.get('blurb', '')}")
        if ds == "heloc" and r.get("sentinel_row_frac") is not None:
            L.append(
                f"  {100 * r['sentinel_row_frac']:.0f}% of rows contain at least one "
                "−9/−8/−7 sentinel. Negative printed bounds are those codes, not a "
                "denormalisation bug."
            )
        L.append("")
    L += [
        "### How the EDA explains the results",
        "",
        "1. **Fid − Pur tracks black-box error, not rule failure.** Gap ≈ 1 − test acc "
        "on HELOC (acc 0.717, RLDA gap +0.196), `uci_credit` (0.812, +0.150), and is "
        "**exactly 0** on wine (test acc 1.000). Mammography/sick/breast_cancer have "
        "acc ≥ 0.95 and gaps ≤ 0.01. Quote this as the reviewers' own argument "
        "reproduced empirically: purity ≠ fidelity unless f̂ is perfect.",
        "",
        "2. **Housing is overlapping 4-class geometry, not a broken policy.** Prices "
        "are quartile-binned into `very_low/low/medium/high` on the same 8 continuous "
        "features. Centroid-NN error is **0.559** (worst of 11) — adjacent bins are "
        "not separable by axis-aligned boxes. That is why RLDA Fid is 0.444 with "
        "conflict 0.363, why Track B π Fid sits at ~0.50, and why classical Anchors "
        "shrinks to **Anchors Cov_D** 0.022 (perturbation-neighbourhood mass, not "
        "Track A Cov): the explanation target itself is a sliced regression, not "
        "discrete classes.",
        "",
        "3. **Wine RLDA Fid 0.714 is small-n / 3-class, not label noise.** Geometry is "
        "easy (centroid-NN error 0.028) and the DNN is perfect on the 36-row test "
        "split, so Fid = Pur. MADA with 3 agents/class recovers Fid 0.935 and Conf "
        "0.000. Report wine as a sample-size caveat (n=178, 13 features, 3 classes), "
        "not as evidence that the MDP cannot separate cultivars.",
        "",
        "4. **Iris RLDA conflict 0.567 is the versicolor/virginica overlap.** Classes "
        "are balanced 50/50/50; setosa is linearly separable on petal width (MADA "
        "and RLDA both recover a one-predicate setosa rule). RLDA's virginica rule "
        "is a broad sepal-width box (Fid 0.429, Cov 1.0). Paper-lock MADA uses petal "
        "predicates for all three classes, Fid 1.0, Conf 0.0 — the coordination "
        "ablation's within-class ensemble story in miniature.",
        "",
        "5. **Imbalanced medical sets inflate majority-class Fid/Cov.** Mammography "
        "is 97.7/2.3 (imb. 42×); sick is 93.9/6.1 (15×). DNN acc 0.986 / 0.981. "
        "Track A Fid ≥ 0.95 and Cov ≥ 0.81 for both arms. Track B speed-up is "
        "**sub-1× for RLDA on mammography** because classical Anchors is already "
        "cheap (~356 q/x) on the trivial majority — nothing to amortise. "
        "Paper-lock MADA is ~1.2× with π Cov_test ~0.55. Do not headline "
        "mammography as an amortisation win.",
        "",
        "6. **Scale and mixed types are the amortisation / categorical story.** "
        "Folktables (195,665 × 10) and Adult (48,842 × 14, 8 labelled categoricals) "
        "are where 0 vs thousands of serving queries matters. Sick has 22 "
        "categoricals (atom-aligned quantiles). HELOC is 22 continuous risk "
        "features with sentinels on 76% of rows — largest Fid−Pur gap and a "
        "balanced 50/50 label, so the gap is classifier noise, not imbalance.",
        "",
        "7. **Breast cancer (d=30, n=569) is separable** (centroid-NN error 0.062). "
        "High Fid with ~2 active features means coverage is not paid for in rule "
        "length even in the highest-dimensional paper set.",
        "",
        "8. **WyoDOT is housing\u2019s lesson at high classifier accuracy.** Centroid-NN "
        "error **0.411** on 5 adjacent road states (Dry\u2013Wet\u2013Slush\u2013Ice\u2013Snow) "
        "\u2014 worse overlap than every set except housing (0.559) \u2014 yet the DNN reaches "
        "0.968. So the boxes are fighting geometry, not classifier noise: Fid stays "
        "\u22650.96 for both arms while Cov sits at 0.62\u20130.68, and the 2.1% Slush class "
        "is where coverage is lost (CART emits no Slush leaf at all). This is the "
        "cleanest separation in the paper between *the classifier is wrong* (HELOC) "
        "and *axis-aligned boxes cannot tile this space* (housing, WyoDOT).",
        "",
        "`covtype` (581,012 × 54, 7 classes) is omitted: RLDA inference OOM.",
        "",
    ]
    return L


def paper_track_b_intro() -> List[str]:
    """Caption that keeps π Cov_test distinct from Ribeiro Anchors Cov_D."""
    n_b = count_paper_track_b()
    missing = ""
    if n_b < 110:
        missing = (
            " Seed 46 MADA Track B is missing until Track A seed 46 MADA finishes; "
            "RLDA seed 46 folktables is missing if that run has not completed."
        )
    return [
        f"`runs/paper_fiveseed_overlap075/` — **{n_b}/110** instance JSONs "
        "(11 datasets × 5 seeds × 2 arms). Same locked policies as Track A "
        "(w=0.75, matched budget). Seed 42 MADA from `mada_overlap075_seed42`; "
        "seed 42 RLDA from `low_budget_seed42_conflict_align`."
        f"{missing} **n** is seeds. "
        "Values are mean ± sd over those seeds. Do not mix with "
        "`low_budget_seed42` original-reward Track B.",
        "",
        "Every scored test instance is explained by the learned policy and, for "
        "the same instance, by classical Anchors (Ribeiro et al. 2018). "
        "The two coverage columns are **different objects** — see Metrics.",
        "",
        "- **π Cov_test** = \(P(x \\in B_\\pi(x^*) \\mid x \\sim D_{\\mathrm{test}})\) — "
        "marginal mass of the instance box on **real test rows**. This is the "
        "Reviewer-1 collapse check (submitted paper was ~0.002–0.008 ≈ one row).",
        "- **Anchors Cov_D** = `explainer.coverage` = \(P(A(z)=1 \\mid z \\sim D)\) — "
        "mass of the **perturbation / train-resample neighbourhood**, **not** "
        "\(P(x \\in A \\mid x \\sim D_{\\mathrm{test}})\). The current JSON does not "
        "score Anchors predicates on \(D_{\\mathrm{test}}\).",
        "- **π pert-Fid** vs **Anchors Prec_D** *are* like-for-like: both are "
        "precision on \(D(z\\mid A)\).",
        "- **π Fid / π Pur** are empirical on real test rows inside \(B_\\pi\).",
        "",
        "Do **not** write “RLDA beats Anchors coverage 8/11” from this table. "
        "Like-for-like test-set coverage vs classical Anchors is Track A "
        "`greedy_anchors` / `sp_anchors`.",
        "",
        "**Reading the query columns.** Both arms read live precision and coverage "
        "off a table of \(\\hat f\)’s predictions on the reference split. That table "
        "is built **once per process** and shared by every explanation, so it is a "
        "fixed cost — reported as **ref q** — and already paid during training. "
        "**π q/x** is what one *additional* explanation costs on top of it. "
        "Earlier versions of this table charged the whole table to whichever "
        "episode happened to trigger the fill, which made MADA’s column read "
        "\(|D_{\\mathrm{train}}|/n_{\\mathrm{scored}}\) (293.5 on folktables, 3.0 on "
        "the small sets, 22.5 on WyoDOT) and RLDA’s read 0.0 — the asymmetry was "
        "purely which arm warmed the cache before the counter was zeroed, not a "
        "difference in serving cost. Do not quote the old numbers.",
        "",
        "Classical Anchors cannot amortise the same way: it evaluates \(\\hat f\) at "
        "synthetic perturbations of each \(x^*\), which are in no precomputed "
        "table, so its per-instance cost is irreducible. That — not a per-instance "
        "arithmetic gap — is the amortisation claim.",
        "",
        "**Reading the wall-clock columns.** π s/x was re-measured when the query "
        "instrumentation was corrected; the Anchors column was **carried over** "
        "rather than re-run, because `anchor-exp` is stochastic and re-running it "
        "would move Prec_D / Cov_D, numbers that defect never touched. The speed-up "
        "ratio therefore pairs two sessions on the same machine. Re-measurement "
        "moved π s/x between 0.52\u00d7 and 1.22\u00d7 across the 24 dataset\u00d7arm "
        "cells, in both directions and uncorrelated with the arm \u2014 machine load, "
        "not a systematic shift. **Quote wall-clock speed-ups as orders of magnitude, "
        "not to two significant figures**, and lean on the query columns, which are "
        "exact counts, for anything load-bearing.",
        "",
        "| dataset | arm | n | n_scored | π Fid | π Pur | π Cov_test | π pert-Fid | Anchors Prec_D | Anchors Cov_D | ref q (1×) | π q/x | Anchors q/x | π s/x | Anchors s/x | speed-up |",
        "|---|---|---:|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]


def paper_track_b_takeaways(grouped: Dict[Any, List[Dict[str, Any]]]) -> List[str]:
    rlda_cov: List[float] = []
    anc_cov: List[float] = []
    mada_q: List[float] = []
    rlda_q: List[float] = []
    anc_q: List[float] = []
    folk_q = None
    for ds in DATASETS:
        for arm in ("rlda", "mada"):
            rows = grouped.get((ds, arm)) or []
            if not rows:
                continue
            mets = [_instance_metrics(d) for d in rows]
            cov = _mean([m["emp_cov"] for m in mets])
            ac = _mean([m["anc_cov"] for m in mets])
            qp = _mean([m["q_pi"] for m in mets])
            qa = _mean([m["q_anc"] for m in mets])
            if arm == "rlda" and cov is not None:
                rlda_cov.append(cov)
            if ac is not None:
                anc_cov.append(ac)
            if qa is not None:
                anc_q.append(qa)
            if arm == "mada" and qp is not None:
                mada_q.append(qp)
                if ds == "folktables_income_CA_2018":
                    folk_q = qp
            if arm == "rlda" and qp is not None:
                rlda_q.append(qp)
    med_cov = _median(rlda_cov)
    med_anc = _median(anc_cov)
    anc_lo = min(anc_q) if anc_q else None
    anc_hi = max(anc_q) if anc_q else None
    mada_nonfolk = []
    for ds in DATASETS:
        if ds == "folktables_income_CA_2018":
            continue
        rows = grouped.get((ds, "mada")) or []
        if not rows:
            continue
        qp = _mean([_instance_metrics(d)["q_pi"] for d in rows])
        if qp is not None:
            mada_nonfolk.append(qp)
    mada_typ = _median(mada_nonfolk)
    ref_q: List[float] = []
    for ds in DATASETS:
        for arm in ("rlda", "mada"):
            rows = grouped.get((ds, arm)) or []
            v = _mean([_instance_metrics(d)["q_ref"] for d in rows]) if rows else None
            if v is not None:
                ref_q.append(v)
    ref_lo = min(ref_q) if ref_q else None
    ref_hi = max(ref_q) if ref_q else None
    both_zero = (
        _median(rlda_q) is not None and _median(mada_q) is not None
        and abs(_median(rlda_q)) < 1e-9 and abs(_median(mada_q)) < 1e-9
    )
    serving = (
        "**both arms serve an additional explanation for 0 black-box queries** "
        "— the rollout is policy forward passes plus box-membership lookups "
        f"against the reference table ({f(ref_lo, 0, comma=True)}–"
        f"{f(ref_hi, 0, comma=True)} queries, paid once)."
        if both_zero else
        f"RLDA serving is {f(_median(rlda_q), 1)} q/x; MADA is "
        f"~{f(mada_typ, 1)} q/x"
        + (f" and {f(folk_q, 1)} on folktables" if folk_q is not None else "")
        + "."
    )
    return [
        "",
        "**Paper takeaways (do not mix coverage columns):** Collapse is gone — "
        f"RLDA median **π Cov_test {f(med_cov)}** over dataset means "
        f"(tens of percent of \(D_{{\\mathrm{{test}}}}\), not one row). "
        f"**Anchors Cov_D** median {f(med_anc)} is Ribeiro perturbation-mass, "
        "not a comparable coverage win. Like-for-like on \(D(z)\): **π pert-Fid** "
        "trails **Anchors Prec_D** on most datasets (on-manifold faithful, weaker "
        f"off-manifold) — report in limitations. On serving cost, {serving} "
        f"Classical Anchors needs {f(anc_lo, 0, comma=True)}–"
        f"{f(anc_hi, 0, comma=True)} q/x **per instance, every instance** "
        "(dataset-mean range), because its perturbations are not in any table. "
        "State the amortisation claim that way — as a fixed cost versus a "
        "per-instance cost — not as one per-instance number beating another.",
        "",
    ]


def _wyodot_method_dir(backend: str, method: str) -> str:
    sub = {"rlda": "ddpg", "mada": "maddpg"}.get(method, "baselines")
    return os.path.join(WYODOT, backend, "results", sub)


def load_wyodot_track_a(backend: str = "dnn") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """seed -> method -> result JSON."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    methods = ("rlda", "mada", "cart", "random_search", "sp_anchors", "greedy_anchors")
    for method in methods:
        for seed in WYODOT_SEEDS:
            path = os.path.join(
                _wyodot_method_dir(backend, method),
                f"wyodot_kvdw_labeled__{method}__seed{seed}__tp0p90__tc0p10.json",
            )
            if not os.path.isfile(path):
                continue
            with open(path) as fh:
                out.setdefault(seed, {})[method] = json.load(fh)
    return out


def load_wyodot_instances(backend: str = "dnn") -> Dict[str, Dict[str, Any]]:
    """seed -> arm -> instance JSON."""
    out: Dict[str, Dict[str, Any]] = {}
    for arm in ("rlda", "mada"):
        for seed in WYODOT_SEEDS:
            path = os.path.join(
                _wyodot_method_dir(backend, arm),
                f"wyodot_kvdw_labeled__{arm}__instances__seed{seed}.json",
            )
            if not os.path.isfile(path):
                continue
            with open(path) as fh:
                out.setdefault(seed, {})[arm] = json.load(fh)
    return out


def count_wyodot_files(backend: str = "dnn") -> Dict[str, int]:
    ta = load_wyodot_track_a(backend)
    n_rl = sum(1 for s in ta.values() for m in s if m in ("rlda", "mada"))
    n_bl = sum(
        1
        for s in ta.values()
        for m in s
        if m in ("cart", "random_search", "sp_anchors", "greedy_anchors")
    )
    n_b = sum(len(v) for v in load_wyodot_instances(backend).values())
    return {"track_a_rl": n_rl, "baselines": n_bl, "track_b": n_b}


def paper_wyodot_section() -> List[str]:
    """WyoDOT case study. DNN 5-seed is on disk; RF is a separate tree."""
    dnn_n = count_wyodot_files("dnn")
    ta = load_wyodot_track_a("dnn")
    inst = load_wyodot_instances("dnn")
    dnn_done = (
        dnn_n["track_a_rl"] == 10
        and dnn_n["baselines"] == 20
        and dnn_n["track_b"] == 10
    )
    if dnn_done:
        status = (
            "**Complete** (10/10 Track A, 20/20 baselines, 10/10 Track B), DNN black box. "
            "RandomForest is out of scope for this submission."
        )
    else:
        status = (
            f"DNN Track A RL {dnn_n['track_a_rl']}/10, baselines "
            f"{dnn_n['baselines']}/20, Track B {dnn_n['track_b']}/10."
        )

    L: List[str] = [
        "",
        "## WyoDOT (`wyodot_kvdw_labeled`) — the 12th dataset",
        "",
        "Road-surface condition classification. Included in the n=12 Wilcoxon as "
        "**one** dataset seed-mean (DNN), not five extra rows. "
        "Same locked protocol as `paper_fiveseed_overlap075` (overlap w=0.75, "
        "cross-class ON, τ_P=0.90, τ_C=0.10, k=1, seeds 42–46). "
        "Trees: `runs/wyodot_fiveseed_overlap075/{dnn,rf}/`. "
        "Do **not** run `wyodot_testbed`. Do not mix Track B π Cov_test into Track A Cov.",
        "",
        status,
        "",
        "Budget (housing-scale `ma_frames=720k` × 24k/360k): **MADA 48k frames/agent**, "
        "**RLDA 720k** total steps (`48k × 3 × K=5`). YAML: "
        "`runs/wyodot_fiveseed_overlap075/conf/anchor.yaml`. "
        "Do not use `wyodot/run_pipeline.py` (historical SA 250k–750k). "
        "Do not copy README RF accuracy into Track A Fid.",
        "",
        "### Dataset facts (after the loader’s NaN / label clean)",
        "",
        "**`wyodot_kvdw_labeled`** (`KVDW_labeled.csv`) — synoptic weather, labels "
        "transferred from rules learned on the testbed (not SurfaceVue-verified on "
        "every row). Class ids are LabelEncoder order: "
        "0 Dry, 1 Ice, 2 Slush, 3 Snow, 4 Wet.",
        "",
        "| | |",
        "|---|---|",
        "| Raw / after dropna | 39,858 → **34,479** |",
        "| Split | train 20,687 / val 6,896 / test 6,896 |",
        "| Features (d=5) | `air_temp_set_1`, `relative_humidity_set_1`, "
        "`dew_point_temperature_set_1d`, `road_temp_set_1`, `wind_speed_set_1` |",
        "| Classes (K=5) | Snow 13,945 (40.4%), Wet 11,182 (32.4%), Dry 6,484 (18.8%), "
        "Ice 2,152 (6.2%), Slush 716 (2.1%) |",
        "| Imbalance | majority/minority **19.5×** (Snow vs Slush) |",
        "| DNN test acc | 0.968 ± 0.001 (5 seeds) |",
        "",
        "`wyodot_testbed` is in the loader but is **not** a paper dataset.",
        "",
        "Five **adjacent** road states on a weather continuum (Dry–Wet–Slush–Ice–Snow), "
        "same geometry lesson as housing’s quartile-binned prices. **Slush** (2.1%) "
        "is the failure mode: CART never emits a Slush leaf; MADA Cov_c ≈ 0.07; "
        "RLDA Cov_c ≈ 0.22.",
        "",
        "| dataset | n (clean) | d | K | mix % | imb. | status |",
        "|---|---:|---:|---:|---|---:|---|",
        "| `wyodot_kvdw_labeled` | 34,479 | 5 | 5 | 40.4/32.4/18.8/6.2/2.1 | 19.5 | "
        "**5-seed complete** (DNN) |",
        "",
        "### Track A (DNN, seeds 42–46)",
        "",
        "Cov = any-class fire rate on \(D_{\\mathrm{test}}\); "
        "Cov_τ = Cov if Fid ≥ 0.90 else 0; Eff = Fid×Cov "
        "(empty ruleset → 0). Shared DNN per seed.",
        "",
        "| seed | method | Fid | Pur | Cov | Cov_τ | Conf | Eff |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    methods = ("rlda", "mada", "cart", "sp_anchors", "greedy_anchors", "random_search")
    acc: Dict[str, Dict[str, List[float]]] = {m: {} for m in methods}
    for seed in WYODOT_SEEDS:
        for method in methods:
            data = (ta.get(seed) or {}).get(method)
            if not data:
                L.append(f"| {seed} | {method} | — | — | — | — | — | — |")
                continue
            gr = data.get("global_ruleset") or {}
            fid, pur = gr.get("global_fidelity"), gr.get("global_purity")
            cov, conf = gr.get("coverage"), gr.get("conflict_rate")
            ct = track_a_cov_tau(gr)
            eff = track_a_eff(gr)
            L.append(
                f"| {seed} | {method} | {f(fid)} | {f(pur)} | {f(cov)} | "
                f"{f(ct)} | {f(conf)} | {f(eff)} |"
            )
            for key, val in (
                ("fid", fid), ("pur", pur), ("cov", cov), ("cov_tau", ct),
                ("conf", conf), ("eff", eff),
            ):
                if val is None:
                    continue
                try:
                    acc[method].setdefault(key, []).append(float(val))
                except (TypeError, ValueError):
                    pass
    L += ["", "Mean ± sd over seeds:", ""]
    for method in methods:
        L.append(
            f"- **{method}** ({len(acc[method].get('eff') or [])} seeds): "
            f"Fid {_msd(acc[method].get('fid') or [])}, "
            f"Pur {_msd(acc[method].get('pur') or [])}, "
            f"Cov {_msd(acc[method].get('cov') or [])}, "
            f"Cov_τ {_msd(acc[method].get('cov_tau') or [])}, "
            f"Conf {_msd(acc[method].get('conf') or [])}, "
            f"Eff {_msd(acc[method].get('eff') or [])}"
        )
    L += [
        "",
        "CART's Eff is replacement accuracy of a 5-leaf tree (Cov ≈ 0.91 by "
        "construction). On Cov_τ, coverage counts only when Fid ≥ 0.90. "
        "RLDA leads MADA on unconstrained Cov at matched Fid. "
        "Conflict is near zero for both RL arms (unlike SP-Anchors). "
        "WyoDOT enters the n=12 Wilcoxon as one paired dataset mean, not five seed rows.",
        "",
        "### Per-class unions (DNN, Cov_c = class-conditional, not Track A Cov)",
        "",
        "| class | method | n | Fid | Pur | Cov_c | Cov_{τ,c} | n_covered |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    cls_methods = ("rlda", "mada", "cart", "sp_anchors", "greedy_anchors", "random_search")
    cls_acc: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
    for seed in WYODOT_SEEDS:
        for method in cls_methods:
            data = (ta.get(seed) or {}).get(method) or {}
            pc = data.get("per_class") or {}
            for cls in WYODOT_CLASS_NAMES:
                rec = (
                    cls_acc.setdefault(cls, {})
                    .setdefault(method, {
                        "fid": [], "pur": [], "cov": [], "cov_tau": [], "n": [],
                    })
                )
                u = (pc.get(cls) or {}).get("union") or {}
                rec["fid"].append(u.get("fidelity"))
                rec["pur"].append(u.get("purity"))
                rec["cov"].append(u.get("coverage"))
                rec["cov_tau"].append(class_cov_tau(u.get("fidelity"), u.get("coverage")))
                rec["n"].append(u.get("n_covered"))
    for cls in ("class_3", "class_4", "class_0", "class_1", "class_2"):
        name = WYODOT_CLASS_NAMES[cls]
        for method in cls_methods:
            rec = (cls_acc.get(cls) or {}).get(method) or {}
            n_json = sum(1 for v in (rec.get("cov") or []) if v is not None)
            L.append(
                f"| {name} (`{cls}`) | {method} | {n_json} | "
                f"{_msd(_finite(rec.get('fid') or []))} | "
                f"{_msd(_finite(rec.get('pur') or []))} | "
                f"{_msd(_finite(rec.get('cov') or []))} | "
                f"{_msd(_finite(rec.get('cov_tau') or []))} | "
                f"{_msd(_finite(rec.get('n') or []))} |"
            )
    L += [
        "",
        "CART has **no Slush leaf** on any seed (n=0). MADA Slush Cov_c 0.070 ± 0.039; "
        "RLDA 0.217 ± 0.087. Majority Snow/Wet are where unconstrained Cov (and Eff) "
        "is earned; Cov_{τ,c} is 0 whenever the class union misses Fid 0.90.",
        "",
        "### Compactness and episode success (DNN, 5 seeds)",
        "",
        "Same columns as the 11-set table, so WyoDOT can be read as the 12th "
        "dataset rather than a separate study. `random_search` at exactly 5.00 "
        "active features is \\(d\\) — every dimension constrained, by construction.",
        "",
        "| method | n | mean active feats | success rate |",
        "|---|---:|---|---|",
    ]
    comp_methods = ("mada", "rlda", "cart", "sp_anchors", "greedy_anchors", "random_search")
    comp: Dict[str, Dict[str, List[float]]] = {m: {"af": [], "sr": []} for m in comp_methods}
    for seed in WYODOT_SEEDS:
        for method in comp_methods:
            data = (ta.get(seed) or {}).get(method) or {}
            af = (data.get("compactness") or {}).get("mean_active_features")
            if af is not None:
                comp[method]["af"].append(float(af))
            sr = data.get("success_rate")
            if isinstance(sr, dict) and sr.get("success_rate") is not None:
                comp[method]["sr"].append(float(sr["success_rate"]))
    for method in comp_methods:
        af_v, sr_v = comp[method]["af"], comp[method]["sr"]
        L.append(
            f"| {method} | {len(af_v)} | {_msd(af_v) if af_v else '—'} | "
            f"{_msd(sr_v) if sr_v else '—'} |"
        )
    L += [
        "",
        "**WyoDOT is where RLDA’s success rate is highest of any paper dataset** "
        f"({_msd(comp['rlda']['sr'])} vs a 0.00–0.39 range on the 11-set), while "
        f"MADA sits at {_msd(comp['mada']['sr'])}. Do not repeat the "
        "“dataset-means 0.00–0.39” range without WyoDOT in it.",
        "",
        "### Track B (DNN, same checkpoints; do not mix with Track A Cov)",
        "",
        "π Cov_test is the instance-box fire rate on real test rows (~0.09–0.13), "
        "not Track A Cov (~0.62–0.68). Anchors Cov_D is perturbation-neighbourhood mass.",
        "",
        "| seed | arm | n_scored | π Fid | π Pur | π Cov_test | π pert-Fid | Anchors Prec_D | Anchors Cov_D | π q/x | Anchors q/x |",
        "|---:|---|---:|---|---|---|---|---|---|---:|---:|",
    ]
    b_acc: Dict[str, Dict[str, List[float]]] = {"rlda": {}, "mada": {}}
    for seed in WYODOT_SEEDS:
        for arm in ("rlda", "mada"):
            d = (inst.get(seed) or {}).get(arm)
            if not d:
                L.append(f"| {seed} | {arm} | — | — | — | — | — | — | — | — | — |")
                continue
            m = _instance_metrics(d)
            L.append(
                f"| {seed} | {arm} | {m.get('n_scored')} | {f(m.get('emp_fid'))} | "
                f"{f(m.get('emp_pur'))} | {f(m.get('emp_cov'))} | "
                f"{f(m.get('pert_fid'))} | {f(m.get('anc_prec'))} | "
                f"{f(m.get('anc_cov'))} | {f(m.get('q_pi'), 1)} | "
                f"{f(m.get('q_anc'), 0, comma=True)} |"
            )
            for key in (
                "emp_fid", "emp_pur", "emp_cov", "pert_fid",
                "anc_prec", "anc_cov", "q_pi", "q_anc",
            ):
                val = m.get(key)
                if val is None:
                    continue
                try:
                    b_acc[arm].setdefault(key, []).append(float(val))
                except (TypeError, ValueError):
                    pass
    L += ["", "Mean ± sd over seeds:", ""]
    for arm in ("rlda", "mada"):
        L.append(
            f"- **{arm.upper()}** Track B: π Fid {_msd(b_acc[arm].get('emp_fid') or [])}, "
            f"π Cov_test {_msd(b_acc[arm].get('emp_cov') or [])}, "
            f"π pert-Fid {_msd(b_acc[arm].get('pert_fid') or [])}, "
            f"Anchors Prec_D {_msd(b_acc[arm].get('anc_prec') or [])}, "
            f"Anchors Cov_D {_msd(b_acc[arm].get('anc_cov') or [])}, "
            f"π q/x {_msd(b_acc[arm].get('q_pi') or [])}, "
            f"Anchors q/x {_msd(b_acc[arm].get('q_anc') or [])}"
        )
    L += [
        "",
        "RLDA serving remains **0 q/x**. MADA serving is ~22 q/x on this K=5 set "
        "(not the ~3 q/x of the binary/ternary 11-set). Classical Anchors is ~9.7k q/x. "
        "π pert-Fid trails Anchors Prec_D — same limitation as the 11-set.",
        "",
    ]
    return L


def paper_local_appendix(main=None, inst=None, wrap: bool = True) -> List[str]:
    """Scale (seed-42 DNNs) + paper-lock Track B from paper_fiveseed."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    seed42 = paper_seed42_track_a()
    grouped = paper_track_b_grouped()
    L: List[str] = []
    if wrap:
        L += [
        BEGIN,
        "",
        "---",
        "",
        "# Appendix — supplemental experiments (paper)",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`.",
        "",
        ]
    else:
        L += ["---", "", "# Scale, classifier accuracy, and Track B", ""]
    L += [
        "Black-box accuracies are the **seed-42 paper-lock DNNs** (same files as "
        "Track A / baselines). **Track B** is scored on the locked policies in "
        "`runs/paper_fiveseed_overlap075/` (overlap w=0.75), not on the original-reward "
        "`low_budget_seed42` instance tables.",
        "",
        "Omitted from the paper: per-dataset Track A grids on the **original** reward, "
        "τ_C=0.05 re-eval (no Track A change at k=1), smoke repeats, covtype (OOM), "
        "and RF black-box runs.",
        "",
        "## Scale and black-box accuracy",
        "",
        "One classifier file per dataset, shared by every arm and every baseline.",
        "",
        "| dataset | train | val | test | total | train acc | val acc | test acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        res = seed42.get((ds, "rlda")) or seed42.get((ds, "mada"))
        if res is None and main:
            res = main.get((ds, "rlda")) or main.get((ds, "mada"))
        if res is None:
            continue
        ss = g(res, "extra", "split_sizes", default={}) or {}
        acc = res.get("classifier_accuracy") or {}
        total = sum(v for v in ss.values() if isinstance(v, int))
        tag = " *(paper scale)*" if ds in NEW_DATASETS else ""
        L.append(
            f"| `{ds}`{tag} | {ss.get('train', 0):,} | {ss.get('val', 0):,} | {ss.get('test', 0):,} | "
            f"{total:,} | {f(acc.get('train_accuracy'))} | {f(acc.get('val_accuracy'))} | "
            f"{f(acc.get('test_accuracy'))} |"
        )
    L += paper_eda_section()
    L += [
        "",
        "## Instance-level comparison vs classical Anchors (Track B)",
        "",
    ]
    L += paper_track_b_intro()
    for ds in DATASETS:
        for arm in ("rlda", "mada"):
            rows = grouped.get((ds, arm)) or []
            if not rows:
                continue
            mets = [_instance_metrics(d) for d in rows]
            n_scored = _mean([m["n_scored"] for m in mets])
            n_scored_s = f"{n_scored:.0f}" if n_scored is not None else "—"
            L.append(
                f"| `{ds}` | {arm.upper()} | {len(rows)} | {n_scored_s} | "
                f"{_msd_n([m['emp_fid'] for m in mets])} | "
                f"{_msd_n([m['emp_pur'] for m in mets])} | "
                f"{_msd_n([m['emp_cov'] for m in mets])} | "
                f"{_msd_n([m['pert_fid'] for m in mets])} | "
                f"{_msd_n([m['anc_prec'] for m in mets])} | "
                f"{_msd_n([m['anc_cov'] for m in mets])} | "
                f"{f(_mean([m['q_ref'] for m in mets]), 0, comma=True)} | "
                f"{_msd_n([m['q_pi'] for m in mets], 1)} | "
                f"{_msd_n([m['q_anc'] for m in mets], 1)} | "
                f"{_msd_n([m['s_pi'] for m in mets], 4)} | "
                f"{_msd_n([m['s_anc'] for m in mets], 4)} | "
                f"{_msd_n([m['speedup'] for m in mets], 1)}× |"
            )
    L += paper_track_b_takeaways(grouped)
    if wrap:
        L += [END, ""]
    return L


def paper_results_appendix(
    conflict: Dict[Any, Dict[str, Any]],
    sac: Dict[Any, Dict[str, Any]],
    baseline: Dict[Any, Dict[str, Any]],
    wrap: bool = True,
) -> List[str]:
    """Headline sweep + config search + seed-42 mechanism ablations only."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    bm = _mean_track_a(baseline, ["mada", "rlda"])
    cm = _mean_track_a(conflict, ["mada", "rlda"])
    sm = _mean_track_a(sac, ["mada", "rlda"])
    paper = paper_seed_means()
    paper_ddpg = count_track_a(PAPER_FIVESEED, "ddpg")
    paper_mada = count_track_a(PAPER_FIVESEED, "maddpg")
    paper_done = paper_ddpg + paper_mada
    w075 = mean_scoreboard_rows(
        load_scoreboard_file(os.path.join(OVERLAP_ROOTS["w075_cross"], "scoreboard.tsv"))
    )

    L: List[str] = []
    if wrap:
        L += [
        ABL_BEGIN,
        "",
        "---",
        "",
        "# Appendix — paper results and ablations",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`.",
        "",
        "**Headline statistics:** `runs/paper_fiveseed_overlap075/` — overlap "
        "w=**0.75**, cross-class ON, low matched budget, seeds 42–46, 11 datasets. "
        "**Eff** = Fid×Cov is the primary. **Cov_τ** = Cov·1[Fid ≥ 0.90] is a "
        "reported diagnostic. Do **not** headline conditional Fid alone, and do "
        "not claim fidelity or coverage against CART.",
        "",
        "Seed-42 ablations below decompose mechanism only (not multi-seeded). "
        "**Not in the paper:** full sweep inventory, τ_C=0.05 sensitivity, "
        "full-budget `conflict_align_sweep_dnn` (cancelled), per-dataset SAC/MASAC "
        "tables, and MASAC as a positive conflict result.",
        "",
        ]
    L += [
        "## 5-seed main comparison (`paper_fiveseed_overlap075`)",
        "",
        "MADA 24k frames/agent base; RLDA `mada_frames×3×n_classes` total steps. "
        f"**{paper_done}/110** Track A JSONs on disk. Seed 42 MADA reused from "
        "`mada_overlap075_seed42`; seed 42 RLDA from `low_budget_seed42_conflict_align`.",
        "",
        "| seed | arm | n | Fid | Cov | Cov_τ | Conf | Eff |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in ("42", "43", "44", "45", "46"):
        for arm in ("rlda", "mada"):
            m = paper.get(seed, {}).get(arm)
            if not m:
                L.append(f"| {seed} | {arm} | — | — | — | — | — | — |")
                continue
            n = int(m.get("n", 0))
            flag = "" if n == 11 else "*"
            L.append(
                f"| {seed} | {arm} | {n}{flag} | {f(m.get('fid'))} | {f(m.get('cov'))} | "
                f"{f(m.get('cov_tau'))} | {f(m.get('conf'))} | {f(m.get('eff'))} |"
            )
    L += ["", "Mean ± sd over seeds with 11/11 datasets:", ""]
    for arm in ("mada", "rlda"):
        seeds_complete = [
            paper[s][arm] for s in ("42", "43", "44", "45", "46")
            if s in paper and arm in paper[s] and paper[s][arm].get("n", 0) == 11
        ]
        if not seeds_complete:
            L.append(f"- **{arm.upper()}:** no complete seeds yet")
            continue
        L.append(
            f"- **{arm.upper()}** ({len(seeds_complete)} seeds): "
            f"Fid {_msd([x['fid'] for x in seeds_complete])}, "
            f"Cov {_msd([x['cov'] for x in seeds_complete])}, "
            f"Cov_τ {_msd([x.get('cov_tau') for x in seeds_complete if x.get('cov_tau') is not None])}, "
            f"Conf {_msd([x['conf'] for x in seeds_complete])}, "
            f"Eff {_msd([x['eff'] for x in seeds_complete])}"
        )
    if paper_done < 110:
        L += [
            "",
            f"Latest Track A log: `{paper_fiveseed_progress()}`. RL per-dataset "
            "rows below may have n<5 until 110/110; **baselines are 5/5**.",
            "",
        ]
    L += paper_wilcoxon_section()
    L += paper_fiveseed_detail_section()

    L += [
        "## Overlap-weight config search (seed 42, MADA only)",
        "",
        "Level inter-class overlap penalty; `shared_reward_weight=0.5` except no_cross arms. "
        "Justifies locked `inter_class_overlap_weight: 0.75` in `BenchMARL/conf/anchor.yaml`.",
        "",
        "| config | w_inter | cross-class | Fid | Cov | Conf | Eff |",
        "|---|---:|---|---:|---:|---:|---:|",
        f"| w=0.50 cross (conflict_align) | 0.50 | on | {f(cm['mada']['fid'])} | "
        f"{f(cm['mada']['cov'])} | {f(cm['mada']['conf'])} | "
        f"{f(cm['mada'].get('eff'))} |",
    ]
    for label, key, w, cross in [
        ("w=0.75 cross (**paper lock**)", "w075_cross", "0.75", "on"),
        ("w=1.0 cross", "w100_cross", "1.0", "on"),
        ("w=0.75 no_cross", "w075_nocross", "0.75", "off"),
        ("w=1.0 no_cross", "w100_nocross", "1.0", "off"),
    ]:
        rows = load_scoreboard_file(os.path.join(OVERLAP_ROOTS[key], "scoreboard.tsv"))
        m = mean_scoreboard_rows(rows)
        if not m:
            L.append(f"| {label} | {w} | {cross} | — | — | — | — |")
            continue
        L.append(
            f"| {label} | {w} | {cross} | {f(m['fid'])} | {f(m['cov'])} | "
            f"{f(m['conf'])} | {f(m['eff'])} |"
        )
    L += [
        "",
        "**Selection:** w=0.75 cross minimises mean conflict (0.030) while maximising "
        "Eff (0.719) among cross-class arms.",
        "",
        "## Reward mechanism ladder (seed 42, MADA means)",
        "",
        "Shows why the overlap penalty was redesigned; RLDA is unchanged across these rows.",
        "",
        "| stage | MADA Fid | MADA Cov | MADA Conf | MADA Eff |",
        "|---|---:|---:|---:|---:|",
        f"| original reward (`low_budget_seed42`) | {f(bm['mada']['fid'])} | "
        f"{f(bm['mada']['cov'])} | {f(bm['mada']['conf'])} | "
        f"{f(bm['mada'].get('eff'))} |",
        f"| conflict-align (level penalty, w=0.5) | {f(cm['mada']['fid'])} | "
        f"{f(cm['mada']['cov'])} | {f(cm['mada']['conf'])} | "
        f"{f(cm['mada'].get('eff'))} |",
    ]
    if w075:
        L.append(
            f"| **paper lock (w=0.75)** | {f(w075['fid'])} | {f(w075['cov'])} | "
            f"{f(w075['conf'])} | {f(w075['eff'])} |"
        )
    L += [
        "",
        "Do **not** claim MADA reduces conflict on the original reward (mean Conf "
        f"{f(bm['mada']['conf'])} vs RLDA {f(bm['rlda']['conf'])}).",
        "",
        "## Coordination ablation (`coord_ablation_seed42`, seed 42)",
        "",
        "Answers what multi-agent coordination buys: **within-class ensemble** "
        "(`agents_per_class=3`), not cross-class shaping.",
        "",
        "| arm | apc | cross-class | Fid | Cov | Conf | Eff |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    coord_sb = os.path.join(COORD_ROOT, "scoreboard.tsv")
    coord_meta = {
        "full_coord": (3, "on"),
        "no_cross_class": (3, "off"),
        "no_same_class": (1, "on"),
        "no_coord": (1, "off"),
    }
    coord_means = _scoreboard_arm_means(coord_sb, tuple(coord_meta))
    for arm in coord_meta:
        m = coord_means.get(arm)
        apc, cross = coord_meta[arm]
        if not m:
            L.append(f"| `{arm}` | {apc} | {cross} | — | — | — | — |")
            continue
        L.append(
            f"| `{arm}` | {apc} | {cross} | {f(m['fid'])} | {f(m['cov'])} | "
            f"{f(m['conf'])} | {f(m['eff'])} |"
        )
    L += [
        "",
        "## Local-penalty reward ablation (`reward_ablation_seed42`, seed 42)",
        "",
        "Overlap held at conflict-align settings; one local geometry term zeroed per arm.",
        "",
        "| arm | Fid | Cov | Conf | Eff |",
        "|---|---:|---:|---:|---:|",
    ]
    reward_means = _scoreboard_arm_means(
        os.path.join(REWARD_ROOT, "scoreboard.tsv"),
        ("full", "no_width", "no_drift", "no_anchor_drift", "no_local"),
    )
    for arm in ("full", "no_width", "no_drift", "no_anchor_drift", "no_local"):
        m = reward_means.get(arm)
        if not m:
            continue
        L.append(
            f"| `{arm}` | {f(m['fid'])} | {f(m['cov'])} | {f(m['conf'])} | {f(m['eff'])} |"
        )
    if not reward_means:
        L.append("| *(scoreboard missing)* | — | — | — | — |")

    L += [
        "",
        "## Algorithm robustness (seed 42, means only)",
        "",
        "Conflict-align env at w=0.5. SAC is a modest RLDA check; **MASAC fails** "
        "to replicate MADDPG conflict gains — cite as a limitation, not a headline.",
        "",
        "| arm | algo | Fid | Cov | Conf | Eff |",
        "|---|---|---:|---:|---:|---:|",
        f"| RLDA | DDPG | {f(cm['rlda']['fid'])} | {f(cm['rlda']['cov'])} | "
        f"{f(cm['rlda']['conf'])} | {f(cm['rlda'].get('eff'))} |",
        f"| RLDA | SAC | {f(sm['rlda']['fid'])} | {f(sm['rlda']['cov'])} | "
        f"{f(sm['rlda']['conf'])} | {f(sm['rlda'].get('eff'))} |",
        f"| MADA | MADDPG (w=0.5) | {f(cm['mada']['fid'])} | {f(cm['mada']['cov'])} | "
        f"{f(cm['mada']['conf'])} | {f(cm['mada'].get('eff'))} |",
        f"| MADA | MASAC | {f(sm['mada']['fid'])} | {f(sm['mada']['cov'])} | "
        f"{f(sm['mada']['conf'])} | {f(sm['mada'].get('eff'))} |",
        "",
    ]
    if wrap:
        L += [ABL_END, ""]
    return L


def paper_rules_document(
    mada: Dict[Any, Dict[str, Any]],
    rlda: Dict[Any, Dict[str, Any]],
    wrap: bool = False,
) -> List[str]:
    """Seed-42 paper-lock rules (MADA w=0.75, RLDA matched) for case studies."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L: List[str] = []
    if wrap:
        L += [BEGIN, "", "---", ""]
    L += [
        "# Extracted rules (paper case studies)",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`.",
        "",
        "Validation-selected unions in **original feature units**, scored on the test split. "
        "Seed 42, DNN. **MADA** is the paper-lock config (`inter_class_overlap_weight: 0.75`, "
        "`runs/mada_overlap075_seed42/`, identical to `paper_fiveseed_overlap075` seed 42). "
        "**RLDA** is the matched-budget DDPG run reused by the 5-seed sweep "
        "(`runs/low_budget_seed42_conflict_align/`). Fid is fidelity to the classifier; "
        "Pur is agreement with the label. **Cov on this page is class-conditional "
        "Cov_c** = \(P(x\\in B\\mid y=c)\) on the test split — not Track A Cov "
        "(any-class fire rate) and not classical Anchors Cov_D.",
        "",
        "These are the rules behind the seed-42 paper cells — not the obsolete "
        "full-budget cluster dump, and not a 5-seed rule dump.",
        "",
        "HELOC bounds of −9/−8/−7 are sentinel codes in the raw file, not a denormalisation bug.",
        "",
    ]
    for ds in DATASETS:
        L += [f"## {ds}", ""]
        L += _rules_block(mada.get((ds, "mada")), "MADA (MADDPG, w=0.75)")
        L += _rules_block(rlda.get((ds, "rlda")), "RLDA (DDPG)")

    # WyoDOT lives in its own run tree but is the 12th paper dataset, so its
    # rules belong on this page — the road-surface case study needs printable
    # predicates, not just aggregate Cov_c.
    wy = load_wyodot_track_a("dnn")
    wy42 = wy.get("42") or wy.get(42) or {}
    if wy42:
        L += [
            "## wyodot_kvdw_labeled",
            "",
            "Same lock (w=0.75, k=1, τ 0.90/0.10), housing-scale budget "
            "(MADA 48k frames/agent, RLDA 720k steps). Class ids are LabelEncoder "
            "order: 0 Dry, 1 Ice, 2 Slush, 3 Snow, 4 Wet.",
            "",
            "Units are metric, from the raw MesoWest columns: air temp −28.1 to "
            "27.8 °C, road temp −25.1 to 63.9 °C, dew point −32.6 to 14.6 °C, RH "
            "7.1–100 %, wind speed 0–24.9 (m s⁻¹ in MesoWest metric mode).",
            "",
            "**Worth a case-study paragraph:** the learned Ice and Snow rules both "
            "cut road temperature at approximately **0 °C** without ever being told "
            "that freezing matters — Ice at `road_temp ≤ 0.33` with `dew_point ≤ "
            "−17.9`, Snow at `road_temp ≤ 0.66` with a milder dew point. The "
            "policy recovers the physical threshold from the classifier's "
            "behaviour alone, which is exactly the kind of check a domain reader "
            "can perform on an explanation and a fidelity number cannot supply.",
            "",
        ]
        L += _rules_block(wy42.get("mada"), "MADA (MADDPG, w=0.75)")
        L += _rules_block(wy42.get("rlda"), "RLDA (DDPG)")
    if wrap:
        L += [END, ""]
    return L


def stub_appendix(begin: str, end: str, note: str) -> List[str]:
    return [begin, "", f"*{note}*", "", end, ""]


def _rules_block(r: Optional[Dict[str, Any]], label: str) -> List[str]:
    L: List[str] = []
    if r is None:
        return [f"**{label}** — no result", ""]
    L += [f"**{label}**", ""]
    for cls, blk in sorted((r.get("per_class") or {}).items()):
        rules = blk.get("selected_rules") or []
        u = blk.get("union") or {}
        L.append(
            f"- `{cls}` (k={blk.get('k')}, union Fid {f(u.get('fidelity'))}, "
            f"Pur {f(u.get('purity'))}, Cov {f(u.get('coverage'))}, "
            f"Conf {f(u.get('conflict_rate'))}, n={u.get('n_covered')})"
        )
        if not rules:
            L.append("  - *(no rule selected)*")
        for i, rule in enumerate(rules, 1):
            rm = rule.get("report_metrics") or {}
            L.append(
                f"  {i}. {rule.get('display_rule', '(no display string)')}"
                f"  — Fid {f(rm.get('fidelity'))}, Pur {f(rm.get('purity'))}, "
                f"Cov {f(rm.get('coverage'))}, n={rm.get('n_covered')}"
            )
    L.append("")
    return L


def ablation_results_appendix(
    conflict: Dict[Any, Dict[str, Any]],
    sac: Dict[Any, Dict[str, Any]],
    baseline: Dict[Any, Dict[str, Any]],
) -> List[str]:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    bm = _mean_track_a(baseline, ["mada", "rlda"])
    cm = _mean_track_a(conflict, ["mada", "rlda"])
    sm = _mean_track_a(sac, ["mada", "rlda"])
    L = [
        ABL_BEGIN,
        "",
        "---",
        "",
        "# Appendix — conflict-align and algorithm ablations (local, seed 42)",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`.",
        "",
        "Local DNN runs on this machine. **Not** folded into the cluster tables "
        "above unless noted. τ_P/τ_C = 0.90/0.10, k=1.",
        "",
    ]
    L += sweep_catalog_section(stamp)
    L += [
        "## Mean Track A — completed seed-42 sweeps (11 datasets)",
        "",
        "**Eff** = Fid×Cov (coverage-controlled rule-set accuracy vs f̂).",
        "",
        "| sweep | MADA algo | MADA Fid | MADA Cov | MADA Conf | MADA Eff | RLDA algo | RLDA Fid | RLDA Cov | RLDA Conf | RLDA Eff |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        f"| `low_budget_seed42` (baseline) | MADDPG | {f(bm['mada']['fid'])} | {f(bm['mada']['cov'])} | "
        f"{f(bm['mada']['conf'])} | {f(bm['mada'].get('eff'))} | DDPG | "
        f"{f(bm['rlda']['fid'])} | {f(bm['rlda']['cov'])} | {f(bm['rlda']['conf'])} | "
        f"{f(bm['rlda'].get('eff'))} |",
        f"| `conflict_align` (w=0.5) | MADDPG | {f(cm['mada']['fid'])} | {f(cm['mada']['cov'])} | "
        f"{f(cm['mada']['conf'])} | {f(cm['mada'].get('eff'))} | DDPG | "
        f"{f(cm['rlda']['fid'])} | {f(cm['rlda']['cov'])} | {f(cm['rlda']['conf'])} | "
        f"{f(cm['rlda'].get('eff'))} |",
        f"| `conflict_align_sac` | MASAC | {f(sm['mada']['fid'])} | {f(sm['mada']['cov'])} | "
        f"{f(sm['mada']['conf'])} | {f(sm['mada'].get('eff'))} | SAC | "
        f"{f(sm['rlda']['fid'])} | {f(sm['rlda']['cov'])} | {f(sm['rlda']['conf'])} | "
        f"{f(sm['rlda'].get('eff'))} |",
        "",
        "Conflict-align MADDPG/DDPG is the mechanism baseline at w=0.5. **Locked paper "
        "config:** overlap w=**0.75**, cross-class ON (`BenchMARL/conf/anchor.yaml`). "
        "SAC is a modest RLDA robustness check. **MASAC does not replicate the "
        "conflict-align win** at this budget.",
        "",
        "### MADA overlap-weight search (seed 42, low budget)",
        "",
        "Level inter-class overlap penalty; `shared_reward_weight=0.5` except no_cross arms.",
        "",
        "| config | w_inter | cross-class | Fid | Cov | Conf | Eff |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    overlap_labels = [
        ("w=0.50 cross (conflict_align)", "w100_cross", "0.50", "on"),
        ("w=0.75 cross (**paper lock**)", "w075_cross", "0.75", "on"),
        ("w=1.0 cross", "w100_cross", "1.0", "on"),
        ("w=0.75 no_cross", "w075_nocross", "0.75", "off"),
        ("w=1.0 no_cross", "w100_nocross", "1.0", "off"),
    ]
    # conflict_align w=0.5 row from cm
    L.append(
        f"| w=0.50 cross (conflict_align) | 0.50 | on | {f(cm['mada']['fid'])} | "
        f"{f(cm['mada']['cov'])} | {f(cm['mada']['conf'])} | "
        f"{f(cm['mada'].get('eff'))} |"
    )
    for label, key, w, cross in overlap_labels[1:]:
        rows = load_scoreboard_file(os.path.join(OVERLAP_ROOTS[key], "scoreboard.tsv"))
        m = mean_scoreboard_rows(rows)
        if not m:
            L.append(f"| {label} | {w} | {cross} | — | — | — | — |")
            continue
        L.append(
            f"| {label} | {w} | {cross} | {f(m['fid'])} | {f(m['cov'])} | "
            f"{f(m['conf'])} | {f(m['eff'])} |"
        )
    L += [
        "",
        "**Selection:** w=0.75 cross minimizes mean conflict (0.030) while maximising "
        "Eff (0.719) among cross-class arms. w=1.0 cross fails on wine/sick (Conf 0.50/0.54). "
        "w=0.75 no_cross regresses on heloc (Conf 0.376). w=1.0 no_cross has highest Eff "
        "(0.770) but Conf 0.108 — coverage-heavy, not cleaner rules.",
        "",
        "### Per-dataset RLDA: DDPG vs SAC (`conflict_align` vs `conflict_align_sac`)",
        "",
        "| dataset | DDPG Fid | DDPG Conf | SAC Fid | SAC Conf |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        dr = conflict.get((ds, "rlda"))
        sr = sac.get((ds, "rlda"))
        if dr is None and sr is None:
            continue
        dg, sg = (dr or {}).get("global_ruleset") or {}, (sr or {}).get("global_ruleset") or {}
        L.append(
            f"| `{ds}` | {f(dg.get('global_fidelity'))} | {f(dg.get('conflict_rate'))} | "
            f"{f(sg.get('global_fidelity'))} | {f(sg.get('conflict_rate'))} |"
        )
    L += [
        "",
        "### Per-dataset MADA: MADDPG vs MASAC",
        "",
        "| dataset | MADDPG Fid | MADDPG Conf | MASAC Fid | MASAC Conf |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        dr = conflict.get((ds, "mada"))
        sr = sac.get((ds, "mada"))
        if dr is None and sr is None:
            continue
        dg, sg = (dr or {}).get("global_ruleset") or {}, (sr or {}).get("global_ruleset") or {}
        L.append(
            f"| `{ds}` | {f(dg.get('global_fidelity'))} | {f(dg.get('conflict_rate'))} | "
            f"{f(sg.get('global_fidelity'))} | {f(sg.get('conflict_rate'))} |"
        )
    L += [
        "",
        "### MADA local-penalty reward ablation (`reward_ablation_seed42`)",
        "",
        "Conflict-align held fixed (w=0.5); one local term zeroed per arm. Means over 11 datasets:",
        "",
        "| arm | Fid | Cov | Conf | Eff |",
        "|---|---:|---:|---:|---:|",
    ]
    reward_sb = os.path.join(REWARD_ROOT, "scoreboard.tsv")
    if os.path.isfile(reward_sb):
        r_acc: Dict[str, Dict[str, List[float]]] = {}
        with open(reward_sb) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                a = row["arm"]
                if row["fid"] == "MISSING":
                    continue
                fid, cov, conf = float(row["fid"]), float(row["cov"]), float(row["conf"])
                r_acc.setdefault(a, {}).setdefault("fid", []).append(fid)
                r_acc[a].setdefault("cov", []).append(cov)
                r_acc[a].setdefault("conf", []).append(conf)
                r_acc[a].setdefault("eff", []).append(fid * cov)
        for arm in ("full", "no_width", "no_drift", "no_anchor_drift", "no_local"):
            if arm not in r_acc:
                continue
            m = r_acc[arm]
            L.append(
                f"| `{arm}` | {f(sum(m['fid'])/len(m['fid']))} | "
                f"{f(sum(m['cov'])/len(m['cov']))} | {f(sum(m['conf'])/len(m['conf']))} | "
                f"{f(sum(m['eff'])/len(m['eff']))} |"
            )
    else:
        L.append("| *(scoreboard missing)* | — | — | — | — |")
    L += [
        "",
        "### Coordination 2×2 (`coord_ablation_seed42`, MADA only)",
        "",
        "| arm | mean Fid | mean Cov | mean Conf | mean Eff |",
        "|---|---:|---:|---:|---:|",
    ]
    coord_sb = os.path.join(COORD_ROOT, "scoreboard.tsv")
    if os.path.isfile(coord_sb):
        arm_acc: Dict[str, Dict[str, List[float]]] = {}
        with open(coord_sb) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                a = row["arm"]
                fid, cov, conf = float(row["fid"]), float(row["cov"]), float(row["conf"])
                arm_acc.setdefault(a, {}).setdefault("fid", []).append(fid)
                arm_acc[a].setdefault("cov", []).append(cov)
                arm_acc[a].setdefault("conf", []).append(conf)
                arm_acc[a].setdefault("eff", []).append(fid * cov)
        for arm in ("full_coord", "no_cross_class", "no_same_class", "no_coord"):
            if arm not in arm_acc:
                continue
            m = arm_acc[arm]
            L.append(
                f"| `{arm}` | {f(sum(m['fid'])/len(m['fid']))} | "
                f"{f(sum(m['cov'])/len(m['cov']))} | {f(sum(m['conf'])/len(m['conf']))} | "
                f"{f(sum(m['eff'])/len(m['eff']))} |"
            )
    else:
        L.append("| *(scoreboard missing)* | — | — | — | — |")

    paper = paper_seed_means()
    paper_ddpg = count_track_a(PAPER_FIVESEED, "ddpg")
    paper_mada = count_track_a(PAPER_FIVESEED, "maddpg")
    L += [
        "",
        "### 5-seed paper sweep (`paper_fiveseed_overlap075`, complete)",
        "",
        "Seeds 42–46, 11 datasets, **overlap w=0.75**, cross-class ON, low matched budget "
        "(MADA 24k frames/agent base; RLDA `mada_frames×3×n_classes` total steps). "
        f"**{paper_ddpg + paper_mada}/110** Track A JSONs on disk. Seed 42 reused from "
        "`mada_overlap075_seed42` (MADA) and `low_budget_seed42_conflict_align` (RLDA).",
        "",
        "| seed | arm | n | Fid | Cov | Conf | Eff |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed in ("42", "43", "44", "45", "46"):
        for arm in ("rlda", "mada"):
            m = paper.get(seed, {}).get(arm)
            if not m:
                L.append(f"| {seed} | {arm} | — | — | — | — | — |")
                continue
            n = int(m.get("n", 0))
            flag = "" if n == 11 else "*"
            L.append(
                f"| {seed} | {arm} | {n}{flag} | {f(m.get('fid'))} | {f(m.get('cov'))} | "
                f"{f(m.get('conf'))} | {f(m.get('eff'))} |"
            )
    # aggregate complete seeds
    L += ["", "Mean ± sd over seeds with 11/11 datasets:", ""]
    for arm in ("mada", "rlda"):
        seeds_complete = [
            paper[s][arm] for s in ("42", "43", "44", "45", "46")
            if s in paper and arm in paper[s] and paper[s][arm].get("n", 0) == 11
        ]
        if not seeds_complete:
            L.append(f"- **{arm.upper()}:** no complete seeds yet")
            continue
        def _msd(key: str) -> str:
            vals = [x[key] for x in seeds_complete]
            mu = sum(vals) / len(vals)
            if len(vals) < 2:
                return f"{mu:.3f}"
            var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
            return f"{mu:.3f} ± {var ** 0.5:.3f}"
        L.append(
            f"- **{arm.upper()}** ({len(seeds_complete)} seeds): Fid {_msd('fid')}, "
            f"Cov {_msd('cov')}, Conf {_msd('conf')}, Eff {_msd('eff')}"
        )
    L += [
        "",
        "Wilcoxon lives in the paper RESULTS file (`python -m revision.paper_stats`). "
        "Do not Wilcoxon this lab appendix.",
        "",
        "Latest log line:",
        "",
        f"> `{paper_fiveseed_progress()}`",
        "",
        ABL_END,
        "",
    ]
    return L


def ablation_rules_appendix(
    conflict: Dict[Any, Dict[str, Any]],
    sac: Dict[Any, Dict[str, Any]],
) -> List[str]:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [
        ABL_BEGIN,
        "",
        "---",
        "",
        "# Appendix — rules from conflict-align ablations",
        "",
        f"Generated {stamp} by `python -m revision.make_local_docs --apply`.",
        "",
        "Validation-selected unions, test-split metrics. **Complete** seed-42 runs "
        "below; 5-seed paper sweep (`runs/paper_fiveseed_overlap075/`) is complete "
        "(110/110 Track A). Rules for seeds 43–46 are in that sweep's inference folders.",
        "",
        "## conflict_align — all 11 datasets (MADDPG / DDPG, seed 42)",
        "",
        "Primary mechanism rules behind the conflict-align numbers in the results appendix.",
        "",
    ]
    for ds in ALL_DATASETS:
        L += [f"### {ds}", ""]
        L += _rules_block(conflict.get((ds, "mada")), "MADA (MADDPG)")
        L += _rules_block(conflict.get((ds, "rlda")), "RLDA (DDPG)")
    L += [
        "## SAC / MASAC comparison (4 datasets, seed 42)",
        "",
        "Datasets where off-policy actors diverge most from DDPG/MADDPG at low budget.",
        "",
    ]
    for ds in ABLATION_RULE_DATASETS:
        L += [f"### {ds}", ""]
        L += _rules_block(sac.get((ds, "mada")), "MADA (MASAC)")
        L += _rules_block(sac.get((ds, "rlda")), "RLDA (SAC)")
    L += [ABL_END, ""]
    return L


def splice(path: str, block: List[str], begin: str = BEGIN, end: str = END) -> None:
    with open(path) as fh:
        text = fh.read()
    body = "\n".join(block)
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + body + tail
    else:
        text = text.rstrip("\n") + "\n\n" + body
    with open(path, "w") as fh:
        fh.write(text)
    print(f"wrote {path}")


def write_doc(path: str, lines: List[str]) -> None:
    with open(path, "w") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    print(f"wrote {path}")


def compose_paper_results(
    main_data: Dict[Any, Dict[str, Any]],
    inst: Dict[Any, Dict[str, Any]],
    conflict_data: Dict[Any, Dict[str, Any]],
    sac_data: Dict[Any, Dict[str, Any]],
) -> List[str]:
    return (
        paper_protocol_header()
        + paper_results_appendix(conflict_data, sac_data, main_data, wrap=False)
        + [""]
        + paper_local_appendix(main_data, inst, wrap=False)
        + paper_wyodot_section()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Rewrite docs/*.md in place.")
    ap.add_argument(
        "--lab",
        action="store_true",
        help="Include full lab notebook appendices (all sweeps, not just paper).",
    )
    args = ap.parse_args()

    main_data = load(MAIN)
    tauc05 = load(TAUC05)
    conflict_data = load(CONFLICT)
    sac_data = load(SAC_ROOT)
    inst = load_instances(MAIN)
    overlap_mada = load(OVERLAP_ROOTS["w075_cross"])

    results_path = os.path.join(REPO, "docs", "RESULTS_comparison.md")
    rules_path = os.path.join(REPO, "docs", "RULES.md")

    res = compose_paper_results(main_data, inst, conflict_data, sac_data)
    rul = paper_rules_document(overlap_mada, conflict_data)

    if args.apply:
        write_doc(results_path, res)
        write_doc(rules_path, rul)
        if args.lab:
            splice(results_path, results_appendix(main_data, tauc05, SMOKES, inst))
            splice(
                results_path,
                ablation_results_appendix(conflict_data, sac_data, main_data),
                ABL_BEGIN,
                ABL_END,
            )
            splice(rules_path, rules_appendix(main_data))
            splice(
                rules_path,
                ablation_rules_appendix(conflict_data, sac_data),
                ABL_BEGIN,
                ABL_END,
            )
        return

    print("\n".join(res))
    print("\n".join(rul))
    if args.lab:
        print("\n".join(results_appendix(main_data, tauc05, SMOKES, inst)))
        print("\n".join(ablation_results_appendix(conflict_data, sac_data, main_data)))
        print("\n".join(rules_appendix(main_data)))
        print("\n".join(ablation_rules_appendix(conflict_data, sac_data)))


if __name__ == "__main__":
    main()
