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
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import revision.evaluate_instances as EI  # noqa: E402
from revision.baselines import (  # noqa: E402
    _anchor_conditions_box, _anchor_rule_box, _try_import_anchor,
)
from revision.dual_estimator_rescore import Scorer, sample_anchors_conditional  # noqa: E402
from utils.metrics import active_feature_mask  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402
from utils.inference_extract import persist_box_from_episode  # noqa: E402

INST = REPO.parent / "results" / "paper_final" / "emp_tc0p10" / "results"
OUT = REPO.parent / "results" / "containment_fix"
N_SAMPLES = 2000
TAU_P = 0.90
MIN_SUPPORT = 10


def score(sc: Scorer, lo, hi, cls: int, space: str, x, rng, rng_all) -> Dict[str, Any]:
    d = sc.score_box(np.asarray(lo, float), np.asarray(hi, float), int(cls), N_SAMPLES, rng, 0.95, space)
    x = np.asarray(x, dtype=np.float32)
    lo32, hi32 = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    # cond_fid resamples only the faces narrower than 95% of the feature's span
    # (the ones a printed RL rule shows). That drops real conditions on skewed
    # features, e.g. Anchors' `x > q1` when q1 sits near the minimum. cond_fid_all
    # holds every face of the box; a face no D_train row violates changes nothing.
    z, _ = sample_anchors_conditional(sc.pools[space], lo32, hi32,
                                      np.ones(len(lo32), dtype=bool), N_SAMPLES, rng_all)
    return {
        "contains_x": bool(np.all((x >= lo32) & (x <= hi32))),
        "cond_fid": d["fid_anchors"],
        "cond_fid_all": float((sc.predict(z, space) == int(cls)).mean()),
        "lower": lo32.tolist(),
        "upper": hi32.tolist(),
        "emp_fid": d["fid_emp"],
        "n_covered": d["n_covered"],
        "coverage": d["n_covered"] / len(sc.y_test),
        "n_active": d["n_active"],
    }


def _patch_into_box(pool, z, lo, hi, active, rng):
    """Anchors' `sample_from_train` step for rows already drawn: replace each violated
    coordinate with a D_train value that satisfies that predicate."""
    z = z.copy()
    for j in np.flatnonzero(active):
        viol = (z[:, j] < lo[j]) | (z[:, j] > hi[j])
        nv = int(viol.sum())
        if not nv:
            continue
        opts = np.flatnonzero((pool[:, j] >= lo[j]) & (pool[:, j] <= hi[j]))
        z[viol, j] = (pool[rng.choice(opts, size=nv, replace=True), j] if opts.size
                      else rng.uniform(lo[j], hi[j], size=nv))
    return z


def score_union(sc: Scorer, boxes, cls: int, space: str, x, rng, rng_all) -> Dict[str, Any]:
    """Score an explanation made of several boxes (their OR) under D(z | union).

    Draw a D_train row; keep it if the union holds it, else patch it Anchors-style
    into one of the boxes, chosen with probability proportional to the box's D_train
    mass. With one box this is exactly the single-box D(z|A) of `score`.
    """
    B = [(np.asarray(lo, np.float32), np.asarray(hi, np.float32)) for lo, hi in boxes]
    pool, X_eval, yh = sc.pools[space], sc.evals[space], sc.y_hat_test[space]
    f_min, f_max = sc.feature_span[space]

    def in_union(Z):
        return np.logical_or.reduce([np.all((Z >= lo) & (Z <= hi), axis=1) for lo, hi in B])

    mass = np.array([np.all((pool >= lo) & (pool <= hi), axis=1).sum() for lo, hi in B], float)
    p_box = mass / mass.sum() if mass.sum() > 0 else np.full(len(B), 1.0 / len(B))

    def fid(active_of, r):
        z = pool[r.integers(0, len(pool), size=N_SAMPLES)].copy()
        out = np.flatnonzero(~in_union(z))
        if out.size:
            which = r.choice(len(B), size=out.size, p=p_box)
            for j, (lo, hi) in enumerate(B):
                rows = out[which == j]
                if rows.size:
                    z[rows] = _patch_into_box(pool, z[rows], lo, hi, active_of(lo, hi), r)
        return float((sc.predict(z, space) == int(cls)).mean())

    m = in_union(X_eval)
    n = int(m.sum())
    x = np.asarray(x, np.float32)
    act = [active_feature_mask(lo, hi, sparsity_width_ratio=0.95, feature_min=f_min, feature_max=f_max)
           for lo, hi in B]
    return {
        "contains_x": bool(in_union(x[None])[0]),
        "cond_fid": fid(lambda lo, hi: active_feature_mask(
            lo, hi, sparsity_width_ratio=0.95, feature_min=f_min, feature_max=f_max), rng),
        "cond_fid_all": fid(lambda lo, hi: np.ones(len(lo), bool), rng_all),
        "emp_fid": float((yh[m] == int(cls)).mean()) if n else float("nan"),
        "n_covered": n,
        "coverage": n / len(sc.y_test),
        "n_active": int(sum(a.sum() for a in act)),
        "n_boxes": len(B),
        "boxes": [{"lower": lo.tolist(), "upper": hi.tolist()} for lo, hi in B],
    }


def floor_check(lo, hi, X_val, y_hat_val, cls: int) -> Dict[str, Any]:
    """A box enters the OR at D_val Fid >= TAU_P; one holding < MIN_SUPPORT D_val
    rows cannot be judged and is kept."""
    m = np.all((X_val >= np.asarray(lo, np.float32)) & (X_val <= np.asarray(hi, np.float32)), axis=1)
    n = int(m.sum())
    f = float((y_hat_val[m] == cls).mean()) if n else float("nan")
    return {"n_val": n, "fid_val": f, "kept": bool(n < MIN_SUPPORT or f + 1e-12 >= TAU_P)}


def run_cell(arm: str, ds: str, seed: int, max_points: int, inst_dir: Path = INST,
             existing: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
        groups, _ = EI._load_mada_policy_groups(exp_dir, "cpu")
    y_hat = sc.y_hat_test["unit"]
    X_val_unit = np.asarray(L.X_val_unit, np.float32)
    y_hat_val = sc.predict(X_val_unit, "unit")
    rng_or, rng_or_all = np.random.default_rng(seed + 303), np.random.default_rng(seed + 404)
    old_rows = {r["index"]: r for r in (existing or {}).get("rows", [])}
    X_unit = np.asarray(L.X_test_unit, np.float32)
    X_orig = np.asarray(L.X_test, np.float32)
    anchors = {r["index"]: r for r in J["anchors"]["rows"] if r.get("ok")}
    # Open faces on features the rule does not use: starting from D_train's
    # min/max would drop test points outside the train range (iris, wine).
    open_face = np.full(X_orig.shape[1], np.finfo(np.float32).max, dtype=np.float32)
    rng = np.random.default_rng(seed + 101)
    rng_all = np.random.default_rng(seed + 202)
    # Only the discretizer's cut points are needed (no classifier calls).
    explainer = _try_import_anchor().AnchorTabularExplainer(
        [str(c) for c in range(L.n_classes)], list(L.feature_names), L.X_train)
    rows: List[Dict[str, Any]] = []

    def _pi_or(arm, row, x, c, idx) -> Dict[str, Any]:
        """The policies' explanation of x*: the OR of every policy's contained box
        that clears the floor (MADA: the class's agents; RLDA: its one policy)."""
        cand = []
        if arm == "rlda":
            pc = row.get("pi_contained")
            if pc:
                cand.append(("policy", pc["lower"], pc["upper"]))
        else:
            conf = dict(cfg)
            conf["enforce_instance_containment"] = True
            for ag, pol in groups.get(c, []):
                ep = EI._rollout_mada(policies={c: pol}, agents={c: ag}, index=index, loader=L,
                                      env_config=conf, env_data=env_data, x_star_unit=x,
                                      y_hat=c, device="cpu", seed=seed + idx)
                box = None if ep.get("error") else persist_box_from_episode(ep, conf, len(x))
                if box is not None:
                    cand.append((ag, box["lower_normalized"], box["upper_normalized"]))
        checks = [{"agent": ag, **floor_check(lo, hi, X_val_unit, y_hat_val, c)} for ag, lo, hi in cand]
        kept = [(lo, hi) for (ag, lo, hi), ch in zip(cand, checks) if ch["kept"]]
        if not kept:
            return {"pi_or": None, "pi_or_abstain": True, "pi_or_policies": checks}
        return {"pi_or": score_union(sc, kept, c, "unit", x, rng_or, rng_or_all),
                "pi_or_abstain": False, "pi_or_policies": checks}

    for idx in [r["index"] for r in J["pi"]["rows"]][:max_points]:
        x, c = X_unit[idx], int(y_hat[idx])
        if idx in old_rows:
            # Earlier cell: keep its pi / pi_contained / anchors, add only pi_or.
            row = dict(old_rows[idx])
            row.update(_pi_or(arm, row, x, c, idx))
            rows.append(row)
            continue
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
            row[tag] = score(sc, box["lower_normalized"], box["upper_normalized"], c, "unit", x, rng, rng_all)
        a = anchors.get(int(idx))
        if a is not None and not (a.get("conditions") or a.get("rule")):
            # The empty anchor is the class prior, not a rule; the report drops
            # this x* from every arm.
            row["anchors"] = None
            row["anchors_empty"] = True
        elif a is not None:
            # Exact bins: the stored conditions, or the printed rule (edges
            # rounded to 2 decimals) matched back to the discretizer cut points.
            lo, hi = -open_face.copy(), open_face.copy()
            amb = 0
            if a.get("conditions"):
                _anchor_conditions_box(explainer, [tuple(t) for t in a["conditions"]], lo, hi)
            else:
                amb = _anchor_rule_box(a["rule"], list(L.feature_names), explainer, X_orig[idx], lo, hi)
            row["anchors"] = {**score(sc, lo, hi, c, "original", X_orig[idx], rng, rng_all),
                              "self_reported_precision": a.get("perturb_fid"),
                              "rule": a.get("rule"), "ambiguous_edges": amb}
        row.update(_pi_or(arm, row, x, c, idx))
        rows.append(row)
    return {"arm": arm, "dataset": ds, "seed": seed, "n": len(rows),
            "precision_estimator": cfg.get("precision_estimator"),
            "anchor_box": "exact discretizer bins",
            # pi_or: the OR of every policy's contained box at D_val Fid >= TAU_P
            # (boxes on < MIN_SUPPORT D_val rows are kept), scored under D(z|union).
            "pi_or": {"floor": TAU_P, "min_support": MIN_SUPPORT,
                      "policies": "all agents of the class" if arm == "mada" else "the class policy"},
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--arms", nargs="+", default=["rlda", "mada"])
    ap.add_argument("--max_points", type=int, default=400)
    ap.add_argument("--inst_dir", type=Path, default=INST,
                    help="folder with {ddpg,maddpg}/*__instances__seed*.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--add_or", action="store_true",
                    help="for cells already in --out, keep pi / pi_contained / anchors and add "
                         "only the floored OR explanation (pi_or)")
    ap.add_argument("--conf_dir", type=Path, default=None,
                    help="env YAMLs the policies were trained with (default: <inst_dir>/../conf)")
    args = ap.parse_args()
    # Roll out under the grid's own env YAMLs (the pert grid sets precision_estimator:
    # conditional); the checkout's defaults are the emp grid's. The training
    # pipeline passed them the same way, through these two variables.
    conf = args.conf_dir or args.inst_dir.parent / "conf"
    if (conf / "anchor.yaml").is_file():
        os.environ["ANCHOR_CONFIG"] = str((conf / "anchor.yaml").resolve())
        os.environ["ANCHOR_SINGLE_CONFIG"] = str((conf / "anchor_single.yaml").resolve())
        print(f"env YAMLs from {conf}")
    else:
        print(f"no {conf}/anchor.yaml, using the checkout's env YAMLs")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for ds in args.datasets:
            for seed in args.seeds:
                path = out / f"{ds}__{arm}__seed{seed}.json"
                existing = None
                if path.is_file():
                    existing = json.loads(path.read_text())
                    if not args.add_or or all("pi_or" in r for r in existing["rows"]):
                        print(f"skip {path.name}")
                        continue
                try:
                    res = run_cell(arm, ds, seed, args.max_points, args.inst_dir, existing)
                except FileNotFoundError as e:
                    print(f"missing {arm} {ds} {seed}: {e}")
                    continue
                path.write_text(json.dumps(res, indent=1))
                print(f"done {arm} {ds} {seed} n={res['n']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
