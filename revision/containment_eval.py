"""Per-instance explanations with and without the containment fix, scored like-for-like.

For the test points in each stored instance file (`*__instances__seed*.json`),
roll out the policy for the predicted class twice: as stored (the saved box can
exclude x*, see `utils.quantile_mdp.contain_point`) and with
`enforce_instance_containment`. Score both boxes and the classical Anchors rule
for the same x* with ONE estimator, Anchors' D(z|A) over D_train (the Track B
port, `revision.dual_estimator_rescore.Scorer`), plus containment and D_test
coverage. Anchors' self-reported precision is kept for comparison: it is the
number `revision.evaluate_instances` reports, and the search stops as soon as
that estimate passes tau, so it is optimistic.

The RL policies must be on this machine (seeds 42-43 here; 44-46 live on spark).

    python -m revision.containment_eval --seeds 42 43
    # on spark, where the policies and instance files live:
    python -m revision.containment_eval --seeds 44 45 46 \
        --inst_dir runs/paper_final/emp_tc0p10/results --out ../results/containment_fix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import revision.evaluate_instances as EI  # noqa: E402
from revision.baselines import _apply_anchor_predicate  # noqa: E402
from revision.dual_estimator_rescore import Scorer  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402
from utils.inference_extract import persist_box_from_episode  # noqa: E402

INST = REPO.parent / "results" / "paper_final" / "emp_tc0p10" / "results"
OUT = REPO.parent / "results" / "containment_fix"
N_SAMPLES = 2000
TAU_P = 0.90


def score(sc: Scorer, lo, hi, cls: int, space: str, x, rng) -> Dict[str, Any]:
    d = sc.score_box(np.asarray(lo, float), np.asarray(hi, float), int(cls), N_SAMPLES, rng, 0.95, space)
    x = np.asarray(x, dtype=np.float32)
    lo32, hi32 = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    return {
        "contains_x": bool(np.all((x >= lo32) & (x <= hi32))),
        "cond_fid": d["fid_anchors"],
        "emp_fid": d["fid_emp"],
        "n_covered": d["n_covered"],
        "coverage": d["n_covered"] / len(sc.y_test),
        "n_active": d["n_active"],
    }


def run_cell(arm: str, ds: str, seed: int, max_points: int, inst_dir: Path = INST) -> Dict[str, Any]:
    algo = {"rlda": "ddpg", "mada": "maddpg"}[arm]
    J = json.loads((Path(inst_dir) / algo / f"{ds}__{arm}__instances__seed{seed}.json").read_text())
    exp_dir, clf = J["experiment_dir"], J["classifier_path"]
    if not Path(exp_dir).is_dir():
        raise FileNotFoundError(f"policies not on this machine: {exp_dir}")
    sc = Scorer(ds, seed, clf)
    L = sc.loader
    cfg = EI._env_yaml(arm, L, exp_dir, seed)
    env_data = L.get_anchor_env_data()
    if arm == "rlda":
        models, env_data = EI._load_rlda_models(exp_dir, L, cfg, "ddpg", "cpu")
    else:
        policies, agents, index, _ = EI._load_mada_policies(exp_dir, "cpu")
    y_hat = sc.y_hat_test["unit"]
    X_unit = np.asarray(L.X_test_unit, np.float32)
    X_orig = np.asarray(L.X_test, np.float32)
    anchors = {r["index"]: r for r in J["anchors"]["rows"] if r.get("ok")}
    # Open faces on features the rule does not use: starting from D_train's
    # min/max would drop test points outside the train range (iris, wine).
    open_face = np.full(X_orig.shape[1], np.finfo(np.float32).max, dtype=np.float32)
    rng = np.random.default_rng(seed + 101)
    rows: List[Dict[str, Any]] = []
    for idx in [r["index"] for r in J["pi"]["rows"]][:max_points]:
        x, c = X_unit[idx], int(y_hat[idx])
        row: Dict[str, Any] = {"index": int(idx), "y_hat": c}
        for tag, flag in (("pi", False), ("pi_contained", True)):
            conf = dict(cfg)
            conf["enforce_instance_containment"] = flag
            if arm == "rlda":
                ep = EI._rollout_rlda(models=models, loader=L, env_config=conf, env_data=env_data,
                                      x_star_unit=x, y_hat=c, device="cpu", seed=seed + idx)
            else:
                ep = EI._rollout_mada(policies=policies, agents=agents, index=index, loader=L,
                                      env_config=conf, env_data=env_data, x_star_unit=x,
                                      y_hat=c, device="cpu", seed=seed + idx)
            box = None if ep.get("error") else persist_box_from_episode(ep, conf, len(x))
            if box is None:
                row[tag] = None
                continue
            row[tag] = score(sc, box["lower_normalized"], box["upper_normalized"], c, "unit", x, rng)
        a = anchors.get(int(idx))
        if a is not None:
            lo, hi = -open_face.copy(), open_face.copy()
            for pred in (a.get("rule") or "").split(" and ") if a.get("rule") else []:
                _apply_anchor_predicate(pred, list(L.feature_names), lo, hi)
            row["anchors"] = {**score(sc, lo, hi, c, "original", X_orig[idx], rng),
                              "self_reported_precision": a.get("perturb_fid"),
                              "rule": a.get("rule")}
        rows.append(row)
    return {"arm": arm, "dataset": ds, "seed": seed, "n": len(rows), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--arms", nargs="+", default=["rlda", "mada"])
    ap.add_argument("--max_points", type=int, default=400)
    ap.add_argument("--inst_dir", type=Path, default=INST,
                    help="folder with {ddpg,maddpg}/*__instances__seed*.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for ds in args.datasets:
            for seed in args.seeds:
                path = out / f"{ds}__{arm}__seed{seed}.json"
                if path.is_file():
                    print(f"skip {path.name}")
                    continue
                try:
                    res = run_cell(arm, ds, seed, args.max_points, args.inst_dir)
                except FileNotFoundError as e:
                    print(f"missing {arm} {ds} {seed}: {e}")
                    continue
                path.write_text(json.dumps(res, indent=1))
                print(f"done {arm} {ds} {seed} n={res['n']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
