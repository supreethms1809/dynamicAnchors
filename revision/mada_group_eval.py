#!/usr/bin/env python3
"""Does MADA's agent group, taken as the class rule set, beat its single best rule?

MADA trains 3 agents per class on a shared class-union reward, but the paper's
rule sets pool every MADA box (15 instance-seeded + the class-level rollouts) and
keep the top k by per-rule score, so at k = 1 the group is never evaluated. Each
agent sees only its own box and terminates on it, so the per-agent class-level
rollouts at inference are the boxes a joint rollout would produce.

Variants, all scored by `revision.evaluate` (D_val selection with greedy
marginal gain and min_support, D_test report, D_val conflict tie-break):

  group    each agent's first class-level box (lowest rollout index): <= 3 rules
  cb_all   every class-level box of the class (all agents, all rollouts)
  pool_all every box MADA produced for the class, k = 20
  RLDA pool_all as the matched "use everything" reference

k = 1 and k = 3 / 5 top-k cells come from ../results/paper_final_valtb (main grid
and k-sweep). Rule files for seeds 42-43 only are on the Mac.

  python -m revision.mada_group_eval            # run
  python -m revision.mada_group_eval --report   # tables
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS  # noqa: E402
from utils.metrics import paired_wilcoxon, track_a_eff  # noqa: E402

PY = sys.executable
VALTB = REPO.parent / "results" / "paper_final_valtb"
OUT = REPO.parent / "results" / "mada_group_eval"
GRIDS = ("emp_tc0p10", "pert_tc0p10")
SEEDS = (42, 43)
ALGO = {"rlda": "ddpg", "mada": "maddpg"}
NM = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot"}


def cell(grid, ds, arm, seed, k=1):
    d = (VALTB / grid / "results" / ALGO[arm]) if k == 1 else (VALTB / "k_sweep" / f"k{k}")
    p = d / f"{ds}__{arm}__seed{seed}__tp0p90__tc0p10.json"
    return json.loads(p.read_text()) if p.is_file() else None


def class_based(cd):
    cb = cd.get("class_based_results") or {}
    return {ag: list(v.get("anchors") or []) for ag, v in cb.items() if isinstance(v, dict)}


def filtered(rules, keep):
    """Copy of a rules file whose classes hold only `keep(class_data)` anchors."""
    out = copy.deepcopy(rules)
    pcr = {}
    for key, cd in rules["per_class_results"].items():
        if key.endswith("_class_based"):
            continue
        pcr[key] = {"class": cd["class"], "anchors": keep(cd)}
    out["per_class_results"] = pcr
    return out


def group_first(cd):
    return [min(a, key=lambda x: x.get("rollout_idx", 0)) for a in class_based(cd).values() if a]


def cb_all(cd):
    return [x for a in class_based(cd).values() for x in a]


def jobs():
    todo = []
    for grid in GRIDS:
        for ds in DATASETS:
            for seed in SEEDS:
                for arm in ("mada", "rlda"):
                    c = cell(grid, ds, arm, seed)
                    if c is None:
                        continue
                    rf = Path(c["extra"]["rules_file"])
                    variants = {"pool_all": (None, 20)}
                    if arm == "mada":
                        variants.update({"group": (group_first, 3), "cb_all": (cb_all, 20)})
                    for name, (keep, k) in variants.items():
                        od = OUT / grid / name
                        dst = od / f"{ds}__{arm}__seed{seed}__tp0p90__tc0p10.json"
                        if dst.is_file():
                            continue
                        src = rf
                        if keep is not None:
                            # evaluate finds the classifier next to the rules file.
                            src = OUT / "rules" / grid / name / f"{ds}__{arm}__seed{seed}" / "extracted_rules.json"
                            src.parent.mkdir(parents=True, exist_ok=True)
                            src.write_text(json.dumps(filtered(json.loads(rf.read_text()), keep)))
                            clf = next(p for p in (rf.parent.parent / "training" / "classifier.pth",
                                                   rf.parent.parent / "classifier.pth",
                                                   rf.parent / "classifier.pth") if p.is_file())
                            link = src.parent / "classifier.pth"
                            if not link.exists():
                                link.symlink_to(clf)
                        todo.append([PY, "-m", "revision.evaluate", "--rules_file", str(src),
                                     "--dataset", ds, "--method", arm, "--seed", str(seed),
                                     "--tau_p", "0.90", "--tau_c", "0.10", "--k", str(k),
                                     "--coverage_basis", "predicted", "--out_dir", str(od)])
    return todo


def run(n_par):
    todo = jobs()
    print(f"{len(todo)} evaluate runs")
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "DYNANC_COVERAGE_BASIS": "predicted"}
    import os
    env = {**os.environ, **env}

    def one(cmd):
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        tag = " ".join(cmd[cmd.index("--dataset") + 1:cmd.index("--dataset") + 6:2]) + " " + cmd[-1].split("/")[-1]
        print(("ok   " if r.returncode == 0 else "FAIL ") + tag, flush=True)
        if r.returncode:
            print(r.stderr[-1500:])
        return r.returncode

    with ThreadPoolExecutor(n_par) as ex:
        fails = sum(1 for rc in ex.map(one, todo) if rc)
    print(f"{fails} failed")


def gm(j):
    g = j["global_ruleset"]
    f = g.get("global_fidelity")
    return {"fid": f if f is not None and f == f else np.nan, "cov": g.get("coverage") or 0.0,
            "eff": track_a_eff(g), "conf": g.get("conflict_rate") or 0.0,
            "rules": np.mean([len(b.get("selected_rules") or []) for b in j["per_class"].values()]) if j["per_class"] else 0.0}


def load_variant(grid, ds, arm, seed, v):
    if v.startswith("k"):
        return cell(grid, ds, arm, seed, int(v[1:]))
    p = OUT / grid / v / f"{ds}__{arm}__seed{seed}__tp0p90__tc0p10.json"
    return json.loads(p.read_text()) if p.is_file() else None


def report():
    cols = [("RLDA k1", "rlda", "k1"), ("MADA k1", "mada", "k1"),
            ("RLDA k3", "rlda", "k3"), ("MADA k3", "mada", "k3"),
            ("MADA group", "mada", "group"), ("MADA cb_all", "mada", "cb_all"),
            ("RLDA pool_all", "rlda", "pool_all"), ("MADA pool_all", "mada", "pool_all")]
    for grid in GRIDS:
        k3_ok = grid == "emp_tc0p10"   # the k-sweep exists for the main grid only
        use = [c for c in cols if k3_ok or c[2] != "k3"]
        per = {c[0]: {} for c in use}
        for lab, arm, v in use:
            for ds in DATASETS:
                ms = [gm(j) for s in SEEDS if (j := load_variant(grid, ds, arm, s, v))]
                if ms:
                    per[lab][ds] = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
        print(f"\n## {grid}, seeds {', '.join(map(str, SEEDS))}\n")
        print("Global rule set on D_test: Fid on decided rows, Cov = share of D_test rows decided, "
              "Eff = Fid × Cov, Conf = conflicted rows, rules = mean selected rules per class.\n")
        print("| | " + " | ".join(c[0] for c in use) + " |")
        print("|---|" + "---:|" * len(use))
        for key, name in (("eff", "Eff"), ("fid", "Fid"), ("cov", "Cov"), ("conf", "Conf"), ("rules", "rules/class")):
            print(f"| {name} (mean of 12) | " + " | ".join(
                f"{np.nanmean([per[c[0]][d][key] for d in DATASETS if d in per[c[0]]]):.3f}" for c in use) + " |")
        print("\nDataset-wise Eff\n")
        print("| dataset | " + " | ".join(c[0] for c in use) + " |")
        print("|---|" + "---:|" * len(use))
        for ds in DATASETS:
            print(f"| {NM.get(ds, ds)} | " + " | ".join(
                f"{per[c[0]][ds]['eff']:.3f}" if ds in per[c[0]] else "—" for c in use) + " |")
        print("\nPaired over 12 datasets (Eff, seed mean), Wilcoxon, unadjusted\n")
        pairs = [("MADA group", "MADA k1"), ("MADA group", "RLDA k1"), ("MADA cb_all", "MADA k1"),
                 ("MADA pool_all", "MADA k1"), ("MADA pool_all", "RLDA pool_all"), ("MADA k1", "RLDA k1")]
        if k3_ok:
            pairs += [("MADA group", "MADA k3"), ("MADA group", "RLDA k3"), ("MADA k3", "RLDA k3")]
        for a, b in pairs:
            ds_ok = [d for d in DATASETS if d in per[a] and d in per[b]]
            x = [per[a][d]["eff"] for d in ds_ok]
            y = [per[b][d]["eff"] for d in ds_ok]
            w = paired_wilcoxon(x, y)
            print(f"- {a} vs {b}: {sum(p > q for p, q in zip(x, y))}/{len(ds_ok)} higher, "
                  f"mean Δ {np.mean(np.subtract(x, y)):+.3f}, p = {w['pvalue']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--par", type=int, default=4)
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run(a.par)


if __name__ == "__main__":
    main()
