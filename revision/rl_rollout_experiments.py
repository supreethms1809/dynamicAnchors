"""Policy rollouts for the weakness battery and the local-to-global continuum.

Two jobs, both inference-only (the trained emp_tc0p10 policies, containment on):

  pool    Roll the class-c policy out from up to --n_start D_train rows that f_hat
          assigns to class c. Each rollout is the per-instance box for that row, so
          the pool is a set of candidate rules seeded across the data; selecting
          from it with different k / fidelity floors moves the method from local
          (one box for one x*) to global (enough boxes to cover the data). See
          `revision.local_global_continuum`.
  robust  For the containment_eval test points, roll out at x* and at a nearby x'
          (Gaussian noise, sd 0.02 in unit space on non-categorical features, same
          f_hat class) with the same rollout seed, so the two boxes differ only
          because the input moved. The tree side is scored from the saved x' in
          `revision.weakness_battery`.

The policies must be on this machine (seeds 42-43 here; 44-46 on spark).

    python -m revision.rl_rollout_experiments --job pool
    python -m revision.rl_rollout_experiments --job robust
    # seeds 44-46 on spark: revision/run_positioning_spark.sh
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import revision.evaluate_instances as EI  # noqa: E402
from revision.containment_eval import INST, OUT as CFIX  # noqa: E402
from revision.dual_estimator_rescore import Scorer  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402
from utils.inference_extract import persist_box_from_episode  # noqa: E402

OUT = REPO.parent / "results" / "local_global"
NOISE_SD = 0.02


class Roller:
    """One (arm, dataset, seed): loaded policies plus a rollout(x_unit, cls) -> box."""

    def __init__(self, arm: str, ds: str, seed: int, inst_dir: Path):
        algo = {"rlda": "ddpg", "mada": "maddpg"}[arm]
        J = json.loads((Path(inst_dir) / algo / f"{ds}__{arm}__instances__seed{seed}.json").read_text())
        self.exp_dir, self.clf = J["experiment_dir"], J["classifier_path"]
        if not Path(self.exp_dir).is_dir():
            raise FileNotFoundError(f"policies not on this machine: {self.exp_dir}")
        self.arm, self.seed = arm, seed
        self.sc = Scorer(ds, seed, self.clf)
        L = self.L = self.sc.loader
        self.cfg = dict(EI._env_yaml(arm, L, self.exp_dir, seed))
        self.cfg["enforce_instance_containment"] = True
        self.env_data = L.get_anchor_env_data()
        if arm == "rlda":
            self.models, self.env_data = EI._load_rlda_models(self.exp_dir, L, self.cfg, "ddpg", "cpu")
        else:
            self.policies, self.agents, self.index, _ = EI._load_mada_policies(self.exp_dir, "cpu")

    def box(self, x_unit: np.ndarray, cls: int, rseed: int):
        kw = dict(loader=self.L, env_config=self.cfg, env_data=self.env_data, x_star_unit=x_unit,
                  y_hat=int(cls), device="cpu", seed=int(rseed))
        if self.arm == "rlda":
            ep = EI._rollout_rlda(models=self.models, **kw)
        else:
            ep = EI._rollout_mada(policies=self.policies, agents=self.agents, index=self.index, **kw)
        b = None if ep.get("error") else persist_box_from_episode(ep, self.cfg, len(x_unit))
        if b is None:
            return None
        return [float(v) for v in b["lower_normalized"]], [float(v) for v in b["upper_normalized"]]


def job_pool(r: Roller, n_start: int) -> Dict[str, Any]:
    X = np.asarray(r.L.X_train_unit, np.float32)
    y_hat = r.sc.predict(X, "unit")
    rng = np.random.default_rng(r.seed + 303)
    out: List[Dict[str, Any]] = []
    t0 = time.time()
    for c in sorted(np.unique(y_hat).tolist()):
        idx = np.flatnonzero(y_hat == c)
        pick = rng.choice(idx, size=min(n_start, idx.size), replace=False)
        for i in pick:
            b = r.box(X[i], c, r.seed + int(i))
            if b is not None:
                out.append({"cls": int(c), "start": int(i), "lower": b[0], "upper": b[1]})
    return {"arm": r.arm, "seed": r.seed, "n_start_per_class": n_start, "boxes": out,
            "seconds": time.time() - t0, "classifier_path": r.clf}


def job_robust(r: Roller, ds: str, max_points: int, cfix_dir: Path = CFIX) -> Dict[str, Any]:
    pts = json.loads((Path(cfix_dir) / f"{ds}__{r.arm}__seed{r.seed}.json").read_text())["rows"][:max_points]
    X = np.asarray(r.L.X_test_unit, np.float32)
    cat = set(int(j) for j in (getattr(r.L, "categorical_indices", None) or []))
    cont = np.array([j not in cat for j in range(X.shape[1])])
    rng = np.random.default_rng(r.seed + 404)
    rows = []
    for p in pts:
        idx, c = int(p["index"]), int(p["y_hat"])
        x = X[idx]
        xp = None
        for _ in range(10):
            z = x.copy()
            z[cont] = np.clip(z[cont] + rng.normal(0, NOISE_SD, cont.sum()), 0, 1)
            if int(r.sc.predict(z[None, :], "unit")[0]) == c:
                xp = z
                break
        if xp is None:
            continue
        rs = r.seed + idx
        b0, b1 = r.box(x, c, rs), r.box(xp, c, rs)
        if b0 is None or b1 is None:
            continue
        rows.append({"index": idx, "y_hat": c, "x_prime_unit": xp.tolist(),
                     "box_x": b0, "box_x_prime": b1})
    return {"arm": r.arm, "seed": r.seed, "noise_sd": NOISE_SD, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["pool", "robust"], required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--arms", nargs="+", default=["rlda", "mada"])
    ap.add_argument("--n_start", type=int, default=150, help="pool: start rows per class")
    ap.add_argument("--max_points", type=int, default=200, help="robust: points per cell")
    ap.add_argument("--inst_dir", type=Path, default=INST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cfix_dir", type=Path, default=CFIX, help="robust: containment_eval output (the test points)")
    args = ap.parse_args()
    out = args.out / args.job
    out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for ds in args.datasets:
            for seed in args.seeds:
                path = out / f"{ds}__{arm}__seed{seed}.json"
                if path.is_file():
                    continue
                try:
                    r = Roller(arm, ds, seed, args.inst_dir)
                except FileNotFoundError as e:
                    print(f"missing {arm} {ds} {seed}: {e}", flush=True)
                    continue
                res = job_pool(r, args.n_start) if args.job == "pool" else job_robust(r, ds, args.max_points, args.cfix_dir)
                res["dataset"] = ds
                path.write_text(json.dumps(res))
                n = len(res.get("boxes") or res.get("rows") or [])
                print(f"done {args.job} {arm} {ds} {seed} n={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
