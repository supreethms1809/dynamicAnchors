#!/usr/bin/env python3
"""Re-score every RL cell with per-policy selection and the policy floor.

`revision.evaluate` pooled all of a class's MADA agents and kept the class's top-k,
so at k = 1 two of the three policies never contributed. It now selects each
policy's top-k from that policy's own pool and ORs the picks: a two-level OR for
MADA (within an agent, across agents), a one-level OR for RLDA's single policy.
A pick enters the OR only at D_val Fid >= the floor (0.60 by default, the same
for both methods); a class whose policies all miss it gets no rule.
Nothing is re-rolled: the stored rules files keep each agent's candidates apart.

The manifest lists every RLDA and MADA cell in ../results/paper_final_valtb (main
grids, k-sweep, ablations) with its rules file, k and tau_C. Each machine re-scores
the cells whose rules file it has (the Mac: seeds 42-43 and the WyoDOT Mac-classifier
ablations; spark: seeds 44-46) into the same layout under --out.

  python -m revision.rescore_per_policy --write_manifest     # Mac, once
  python -m revision.rescore_per_policy --apply              # either machine
  python -m revision.rescore_per_policy --check 20 --apply   # pooled, no-floor re-run of
                                                             # 20 cells; must equal the stored cells
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VALTB = REPO.parent / "results" / "paper_final_valtb"
MANIFEST = REPO / "revision" / "per_policy_manifest.json"
MAC_ROOT, SPARK_ROOT = "/Users/ssuresh/dynAnc_codeCleanup/dynamicAnchors", "/home/sureshm/dynamicAnchors"


def write_manifest() -> None:
    rows = []
    for p in sorted([*VALTB.rglob("*__mada__seed*__tp*__tc*.json"),
                     *VALTB.rglob("*__rlda__seed*__tp*__tc*.json")]):
        j = json.loads(p.read_text())
        rows.append({
            "rel": str(p.relative_to(VALTB)), "rules_file": j["extra"]["rules_file"],
            "method": j["method"],
            "dataset": j["dataset"], "seed": j["seed"], "tau_p": j["tau_p"], "tau_c": j["tau_c"],
            "k": j["extra"]["k"], "ranking_formula": j["extra"].get("ranking_formula"),
            "min_support": j.get("min_support"),
            "global_ruleset": j["global_ruleset"],
        })
    MANIFEST.write_text(json.dumps(rows, indent=0))
    print(f"{len(rows)} RL cells -> {MANIFEST}")


def local_rules(rf: str) -> str | None:
    for cand in (rf, rf.replace(SPARK_ROOT, MAC_ROOT), rf.replace(MAC_ROOT, SPARK_ROOT)):
        if os.path.isfile(cand):
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_manifest", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO.parent / "results" / "paper_final_perpolicy")
    ap.add_argument("--check", type=int, default=0,
                    help="instead: re-run N random local cells with --selection pooled and "
                         "compare with the stored global rule set")
    ap.add_argument("--policy_floor", default="0.60")
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.write_manifest:
        write_manifest()
        return 0
    rows = json.loads(MANIFEST.read_text())
    todo = [(r, lr) for r in rows if (lr := local_rules(r["rules_file"]))]
    print(f"{len(todo)} of {len(rows)} cells have their rules file here")
    selection, floor, out = "per_policy", a.policy_floor, a.out
    if a.check:
        random.seed(0)
        todo = random.sample(todo, min(a.check, len(todo)))
        selection, floor, out = "pooled", "none", a.out.parent / (a.out.name + "_pooled_check")
    todo = [(r, lr) for r, lr in todo if not (out / r["rel"]).is_file()]
    print(f"{len(todo)} to run ({selection}) -> {out}")
    if not a.apply:
        return 0
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "DYNANC_COVERAGE_BASIS": "predicted"}

    def one(item):
        r, lr = item
        od = out / Path(r["rel"]).parent
        cmd = [sys.executable, "-m", "revision.evaluate", "--rules_file", lr,
               "--dataset", r["dataset"], "--method", r["method"], "--seed", str(r["seed"]),
               "--tau_p", str(r["tau_p"]), "--tau_c", str(r["tau_c"]), "--k", str(r["k"]),
               "--coverage_basis", "predicted", "--selection", selection,
               "--policy_floor", floor, "--out_dir", str(od)]
        if r.get("ranking_formula"):
            cmd += ["--ranking_formula", r["ranking_formula"]]
        if r.get("min_support") is not None:
            cmd += ["--min_support", str(r["min_support"])]
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        if p.returncode:
            print(f"FAIL {r['rel']}\n{p.stderr[-800:]}", flush=True)
            return 1
        if a.check:
            got = json.loads((out / r["rel"]).read_text())["global_ruleset"]
            same = got == r["global_ruleset"]
            print(f"{'same' if same else 'DIFF'} {r['rel']}", flush=True)
            return 0 if same else 1
        print(f"ok   {r['rel']}", flush=True)
        return 0

    with ThreadPoolExecutor(a.par) as ex:
        bad = sum(ex.map(one, todo))
    print(f"done; {bad} failed or differed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
