"""Instance explanations from a plain global surrogate, on the containment_eval points.

The explanation of x* is the leaf of a DecisionTreeClassifier (fit to
f_hat(D_train)) that x* falls in, read as a box with open faces. It is scored
exactly like the RL boxes and the Anchors rules in `containment_eval`: Anchors'
D(z|A) over D_train with target y_hat(x*), 2,000 draws, plus containment,
D_test coverage and active features. When the tree's label for x* differs from
y_hat(x*), the leaf is still scored for y_hat(x*): it is a wrong explanation.

Serving cost: one f_hat query per input (for y_hat); the fit costs one query
per D_train row, once.

    python -m revision.tree_instance_eval
    # on spark, after containment_eval has written its seed 44-46 files:
    python -m revision.tree_instance_eval --seeds 44 45 46 \
        --inst_dir runs/paper_final/emp_tc0p10/results --cfix_dir ../results/containment_fix
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.cart_global_surrogate_pert import leaf_boxes  # noqa: E402
from revision.containment_eval import INST, OUT as CFIX, score  # noqa: E402
from revision.dual_estimator_rescore import Scorer  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402

TUNE_DEPTHS = tuple(range(1, 16)) + (None,)


def run_cell(ds: str, seed: int, inst_dir: Path = INST, cfix_dir: Path = CFIX) -> Dict[str, Any]:
    from sklearn.tree import DecisionTreeClassifier

    J = json.loads((Path(inst_dir) / "ddpg" / f"{ds}__rlda__instances__seed{seed}.json").read_text())
    pts = json.loads((Path(cfix_dir) / f"{ds}__rlda__seed{seed}.json").read_text())["rows"]
    sc = Scorer(ds, seed, J["classifier_path"])
    L = sc.loader
    X_tr = sc.pools["original"]
    t0 = time.time()
    y_tr = sc.predict(X_tr, "original")
    Xv = np.asarray(L.X_val, np.float32)
    y_v = sc.predict(Xv, "original")

    def fit(d):
        return DecisionTreeClassifier(max_depth=d, random_state=seed).fit(X_tr, y_tr)

    val = {d: float((fit(d).predict(Xv) == y_v).mean()) for d in TUNE_DEPTHS}
    d_star = next(d for d in TUNE_DEPTHS if val[d] == max(val.values()))
    trees = {"depth 3": fit(3), "depth 5": fit(5), "val-tuned depth": fit(d_star)}
    fit_s = time.time() - t0
    Xt = sc.evals["original"]
    y_hat = sc.y_hat_test["original"]
    rng = np.random.default_rng(seed + 202)
    out: Dict[str, Any] = {"dataset": ds, "seed": seed, "val_tuned_depth": d_star,
                           "fit_seconds": fit_s, "queries_fit": int(len(X_tr)), "rows": []}
    boxes = {name: leaf_boxes(tr, Xt.shape[1]) for name, tr in trees.items()}
    cache: Dict[Any, Dict[str, Any]] = {}
    serve = {name: 0.0 for name in trees}
    for p in pts:
        idx = int(p["index"])
        # Same target as the RL and Anchors scores: y_hat on the (clipped) unit
        # inputs. It can differ from y_hat on original units for test rows
        # outside the train range.
        c = int(p["y_hat"])
        row: Dict[str, Any] = {"index": idx, "y_hat": c, "y_hat_original_units": int(y_hat[idx])}
        x = Xt[idx:idx + 1]
        for name, tr in trees.items():
            t1 = time.time()
            leaf = int(tr.apply(x)[0])
            label = int(tr.predict(x)[0])
            serve[name] += time.time() - t1
            lo, up, _ = boxes[name][leaf]
            key = (name, leaf, c)
            if key not in cache:
                cache[key] = score(sc, lo, up, c, "original", Xt[idx], rng)
            s = dict(cache[key])
            s["contains_x"] = bool(np.all((Xt[idx] >= lo) & (Xt[idx] <= up)))
            row[name] = {**s, "tree_label_matches_y_hat": label == c, "leaf": leaf}
        out["rows"].append(row)
    out["serve_seconds_per_input"] = {k: v / max(len(pts), 1) for k, v in serve.items()}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--inst_dir", type=Path, default=INST,
                    help="folder with ddpg/*__rlda__instances__seed*.json (for classifier_path)")
    ap.add_argument("--cfix_dir", type=Path, default=CFIX, help="containment_eval output (the test points)")
    args = ap.parse_args()
    out = args.cfix_dir / "tree"
    out.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for seed in args.seeds:
            path = out / f"{ds}__seed{seed}.json"
            if path.is_file():
                continue
            res = run_cell(ds, seed, args.inst_dir, args.cfix_dir)
            path.write_text(json.dumps(res, indent=1))
            m = np.mean([r["depth 3"]["cond_fid"] for r in res["rows"]])
            print(f"{ds} s{seed}: n={len(res['rows'])} depth-3 condFid {m:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
