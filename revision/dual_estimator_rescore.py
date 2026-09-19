"""
Diagnostic: re-score already-selected rules under three fidelity estimators.

This is a DIAGNOSTIC pass, deliberately kept out of the paper artifact
directories. It does not retrain, re-search, or re-select anything: it reads the
rules that `revision.evaluate` / `revision.baselines` already selected on D_val
and reported on D_test, and asks what their fidelity would be under a different
estimand.

Three estimators, same boxes, same black box:

  fid_emp      P(f_hat(x) = c | x in B), x drawn from held-out D_test rows.
               This is what the paper reports. On-manifold, finite support.

  fid_anchors  P(f_hat(z) = c), z ~ D(z|B) using a faithful port of Ribeiro's
               `anchor_tabular.AnchorTabularExplainer.sample_from_train`:
               draw real TRAIN rows uniformly; for each row that VIOLATES a
               predicate, overwrite that one coordinate with a value drawn from
               the marginal of train rows satisfying the predicate. Rows already
               inside the box are returned verbatim. This is the estimand the
               Anchors precision guarantee is stated over.

  fid_crn      P(f_hat(z) = c), z ~ the repo's own `qmdp.crn_perturb`, which
               resamples EVERY drawn row's constrained coordinates uniformly
               over the observed knots inside the box. Strictly harsher than
               Ribeiro's sampler (it perturbs satisfying rows too), so it is a
               conservative lower bound, NOT a reproduction of Anchors.

The gap fid_emp - fid_anchors is the quantity of interest: it is large exactly
when the box has wide, unconstrained dimensions whose corners carry no data, so
that recombining coordinates lands off the data manifold.

Usage:
    python -m revision.dual_estimator_rescore \\
        --results_dir runs/paper_fiveseed_overlap075/results \\
        --out_dir revision/diagnostics/dual_estimator
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))

from utils import quantile_mdp as qmdp  # noqa: E402
from utils.dataset_factory import make_tabular_loader  # noqa: E402
from utils.metrics import active_feature_mask, box_mask, wilson_interval  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("revision.dual_estimator_rescore")

SCHEMA = "dual_estimator_rescore_v1"

# `_emit` in revision/baselines.py takes box_space as a call argument and does not
# persist it, so the space each method's stored bounds live in is recorded here.
# revision/evaluate.py does persist it as extra.bounds_space for the RL arms.
BOX_SPACE_BY_METHOD = {
    "mada": "unit",
    "rlda": "unit",
    "random_search": "unit",
    "cart": "original",
    "sp_anchors": "original",
    "greedy_anchors": "original",
    "anchors": "original",
}


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_anchors_conditional(
    X_pool_unit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port of Anchors' `sample_from_train`, in unit space.

    Ribeiro patches ONLY the coordinates that violate a predicate, and draws the
    replacement from the marginal of train rows satisfying that predicate. Rows
    already inside the box come back untouched, so D(z|B) is a mixture of real
    in-box rows and coordinate-wise hybrids. Returns (z, was_modified).
    """
    n_pool = X_pool_unit.shape[0]
    idx = rng.integers(0, n_pool, size=n_samples)
    z = X_pool_unit[idx].copy()
    modified = np.zeros(n_samples, dtype=bool)
    for j in np.flatnonzero(active):
        lo, up = float(lower[j]), float(upper[j])
        viol = (z[:, j] < lo) | (z[:, j] > up)
        n_viol = int(viol.sum())
        if n_viol == 0:
            continue
        opts = np.flatnonzero((X_pool_unit[:, j] >= lo) & (X_pool_unit[:, j] <= up))
        if opts.size == 0:
            # Ribeiro's fallback when no train row satisfies the predicate.
            repl = rng.uniform(lo, up, size=n_viol)
        else:
            repl = X_pool_unit[rng.choice(opts, size=n_viol, replace=True), j]
        z[viol, j] = repl
        modified |= viol
    return z.astype(np.float32), modified


def sample_crn(
    X_pool_unit: np.ndarray,
    v_all: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The repo's own D(z|A), via utils.quantile_mdp.crn_perturb.

    Unlike Ribeiro's, this resamples every drawn row's constrained coordinates,
    including rows that already satisfy the box.
    """
    idx = rng.integers(0, X_pool_unit.shape[0], size=n_samples).astype(np.int64)
    U = rng.random((n_samples, X_pool_unit.shape[1]))
    return qmdp.crn_perturb(
        X_pool_unit, idx, U, lower, upper, active, v_all,
        x_star_unit=None, class_mode_values=None,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class Scorer:
    """Holds one (dataset, seed) loader + classifier and scores boxes against it."""

    def __init__(self, dataset: str, seed: int, classifier_path: str):
        import torch  # noqa: F401

        self.dataset = dataset
        self.seed = int(seed)
        loader = make_tabular_loader(dataset, random_state=int(seed))
        loader.load_dataset()
        loader.preprocess_data()
        loader.classifier = loader.load_classifier(filepath=classifier_path, device="cpu")
        self.loader = loader
        self.classifier_path = classifier_path

        self.X_min = np.asarray(loader.X_min, dtype=np.float64)
        self.X_range = np.asarray(loader.X_range, dtype=np.float64)
        self.scaler_mean = np.asarray(loader.scaler.mean_, dtype=np.float64)
        self.scaler_scale = np.asarray(loader.scaler.scale_, dtype=np.float64)
        self.feature_names = list(loader.feature_names)

        # Every box is scored in the space it was BUILT in. The RL arms and
        # random search work in the unit cube; CART and the Anchors family work
        # in original feature units. Converting between them is lossy here:
        # X_*_unit is CLIPPED to [0, 1], so a test row beyond the train range
        # collapses onto the cube face and can fall inside a converted box it was
        # outside of in original units. Scoring natively keeps fid_emp identical
        # to the published artifact, which is the cross-check this pass relies on.
        self.pools = {
            "unit": np.asarray(loader.X_train_unit, dtype=np.float32),
            "original": np.asarray(loader.X_train, dtype=np.float32),
        }
        self.evals = {
            "unit": np.asarray(loader.X_test_unit, dtype=np.float32),
            "original": np.asarray(loader.X_test, dtype=np.float32),
        }
        self.y_test = np.asarray(loader.y_test)
        # Sorted all-train values per feature: the D(z|A) knot pool (all classes,
        # matching AnchorTabularExplainer(train_data=X_train)).
        self.knots = {
            sp: np.stack([np.sort(P[:, j]) for j in range(P.shape[1])], axis=0).astype(np.float64)
            for sp, P in self.pools.items()
        }
        # Original-space compactness compares against the train span, matching
        # `_emit` in revision/baselines.py; unit space uses the [0, 1] default.
        self.feature_span = {
            "unit": (None, None),
            "original": (
                np.min(self.pools["original"], axis=0).astype(np.float64),
                np.max(self.pools["original"], axis=0).astype(np.float64),
            ),
        }
        self.y_hat_test = {sp: self.predict(self.evals[sp], sp) for sp in self.pools}

    # -- coordinate spaces --------------------------------------------------
    def unit_to_std(self, X_unit: np.ndarray) -> np.ndarray:
        return np.asarray(X_unit, dtype=np.float64) * self.X_range + self.X_min

    def to_std(self, X: np.ndarray, space: str) -> np.ndarray:
        """Both spaces map onto the StandardScaler space the classifier expects."""
        if space == "unit":
            return self.unit_to_std(X)
        return (np.asarray(X, dtype=np.float64) - self.scaler_mean) / self.scaler_scale

    def predict(self, X: np.ndarray, space: str) -> np.ndarray:
        import torch
        from utils.networks import predict_proba_torch

        clf = self.loader.classifier
        if hasattr(clf, "eval"):
            clf.eval()
        X_std = self.to_std(X, space).astype(np.float32)
        out = np.empty(X_std.shape[0], dtype=np.int64)
        step = 20000
        with torch.no_grad():
            for s in range(0, X_std.shape[0], step):
                chunk = torch.from_numpy(X_std[s:s + step])
                probs = predict_proba_torch(clf, chunk).cpu().numpy()
                out[s:s + step] = probs.argmax(axis=1)
        return out

    # -- one box ------------------------------------------------------------
    def score_box(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        target_class: int,
        n_samples: int,
        rng: np.random.Generator,
        sparsity_width_ratio: float,
        space: str,
    ) -> Dict[str, Any]:
        lower = np.asarray(lower, dtype=np.float32).reshape(-1)
        upper = np.asarray(upper, dtype=np.float32).reshape(-1)
        pool = self.pools[space]
        X_eval = self.evals[space]
        y_hat = self.y_hat_test[space]
        f_min, f_max = self.feature_span[space]
        active = active_feature_mask(
            lower, upper, sparsity_width_ratio=sparsity_width_ratio,
            feature_min=f_min, feature_max=f_max,
        )

        # 1. Empirical: real held-out rows inside the box.
        mask = box_mask(X_eval, lower, upper)
        n_cov = int(mask.sum())
        n_agree = int((y_hat[mask] == int(target_class)).sum()) if n_cov else 0
        fid_emp = (n_agree / n_cov) if n_cov else float("nan")
        emp_lo, emp_hi = wilson_interval(n_agree, n_cov)

        # 2. Anchors' D(z|B).
        z_anc, modified = sample_anchors_conditional(
            pool, lower, upper, active, n_samples, rng
        )
        pred_anc = self.predict(z_anc, space)
        fid_anchors = float((pred_anc == int(target_class)).mean())
        frac_synth = float(modified.mean())
        # Fidelity split by whether the sample was a real row or a hybrid.
        fid_anchors_real = (
            float((pred_anc[~modified] == int(target_class)).mean())
            if (~modified).any() else float("nan")
        )
        fid_anchors_hybrid = (
            float((pred_anc[modified] == int(target_class)).mean())
            if modified.any() else float("nan")
        )

        # 3. The repo's harsher CRN sampler.
        z_crn = sample_crn(
            pool, self.knots[space], lower, upper, active, n_samples, rng
        )
        fid_crn = float((self.predict(z_crn, space) == int(target_class)).mean())

        # Widths are reported as a fraction of each feature's full span so the
        # number means the same thing in both spaces.
        span = (np.ones_like(lower, dtype=np.float64) if f_min is None
                else np.maximum(np.asarray(f_max) - np.asarray(f_min), 1e-12))
        widths = np.clip((upper - lower) / span, 0.0, 1.0)
        return {
            "target_class": int(target_class),
            "n_active": int(active.sum()),
            "active_features": [self.feature_names[j] for j in np.flatnonzero(active)],
            "n_features": int(lower.shape[0]),
            "mean_active_width_frac": (
                float(widths[active].mean()) if active.any() else float("nan")
            ),
            "max_active_width_frac": (
                float(widths[active].max()) if active.any() else float("nan")
            ),
            "fid_emp": fid_emp,
            "n_covered": n_cov,
            "fid_emp_ci": [emp_lo, emp_hi],
            "fid_anchors": fid_anchors,
            "fid_anchors_real_rows": fid_anchors_real,
            "fid_anchors_hybrid_rows": fid_anchors_hybrid,
            "frac_synthetic": frac_synth,
            "fid_crn": fid_crn,
            "gap_anchors": (
                float("nan") if not np.isfinite(fid_emp) else float(fid_emp - fid_anchors)
            ),
            "gap_crn": (
                float("nan") if not np.isfinite(fid_emp) else float(fid_emp - fid_crn)
            ),
        }


# ---------------------------------------------------------------------------
# Artifact walking
# ---------------------------------------------------------------------------

def _classifier_for(result: Dict[str, Any], result_path: Path) -> Optional[str]:
    """Shared per-(dataset, seed) classifier; all methods in a run share one file."""
    extra = result.get("extra") or {}
    cand: List[str] = []
    if extra.get("classifier_path"):
        cand.append(str(extra["classifier_path"]))
    rules_file = extra.get("rules_file")
    if rules_file:
        exp_dir = os.path.dirname(os.path.dirname(str(rules_file)))
        cand.append(os.path.join(exp_dir, "training", "classifier.pth"))
        cand.append(os.path.join(exp_dir, "classifier.pth"))
    # runs/<sweep>/results/<method>/x.json -> runs/<sweep>/classifiers/<ds>_seed<n>.pth
    sweep = result_path.parent.parent.parent
    cand.append(str(sweep / "classifiers" / f"{result['dataset']}_seed{result['seed']}.pth"))
    return next((c for c in cand if os.path.exists(c)), None)


def rescore_result(
    result_path: Path,
    scorer: Scorer,
    n_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    result = json.loads(result_path.read_text())
    method = str(result["method"])
    extra = result.get("extra") or {}
    space = extra.get("bounds_space") or BOX_SPACE_BY_METHOD.get(method)
    if space is None:
        raise ValueError(f"Unknown bounds space for method {method!r} ({result_path})")
    sparsity = float(extra.get("sparsity_width_ratio", 0.95))

    per_class_out: Dict[str, Any] = {}
    for cls_key, block in (result.get("per_class") or {}).items():
        rules = block.get("selected_rules") or []
        target = None
        for src in (block.get("best") or {}, {}):
            target = (src.get("fidelity") or {}).get("target_class")
            if target is not None:
                break
        if target is None:
            target = int(str(cls_key).rsplit("_", 1)[-1])

        scored: List[Dict[str, Any]] = []
        for r in rules:
            lo, up = r.get("lower_bounds"), r.get("upper_bounds")
            if lo is None or up is None:
                continue
            lo = np.asarray(lo, dtype=np.float64)
            up = np.asarray(up, dtype=np.float64)
            if space == "unit":
                # Unit boxes are already cube-bounded; clip guards float drift only.
                lo, up = np.clip(lo, 0.0, 1.0), np.clip(up, 0.0, 1.0)
            row = scorer.score_box(
                lo, up, int(target), n_samples, rng, sparsity, space
            )
            row["rule_id"] = r.get("rule_id")
            row["display_rule"] = r.get("display_rule")
            reported = (r.get("report_metrics") or {}).get("fidelity")
            if reported is None:
                reported = (r.get("metrics") or {}).get("fidelity")
            row["fid_emp_reported"] = reported
            # Cross-check against the published artifact. box_mask casts bounds to
            # float32, so a row sitting exactly on a box face can land on either
            # side after the bounds round-trip through JSON; allow two rows of
            # slack and carry the raw delta so any real drift is still visible.
            if reported is None or not np.isfinite(row["fid_emp"]) or not row["n_covered"]:
                row["emp_abs_delta"] = None
                row["emp_matches_reported"] = None
            else:
                delta = abs(float(reported) - row["fid_emp"])
                row["emp_abs_delta"] = float(delta)
                row["emp_matches_reported"] = bool(delta <= 2.0 / row["n_covered"] + 1e-9)
            scored.append(row)

        if not scored:
            continue
        per_class_out[str(cls_key)] = {
            "target_class": int(target),
            "n_selected": len(scored),
            "rules": scored,
            # Rank-1 of the selected set, i.e. the `best` column in the paper.
            "best": scored[0],
        }

    return {
        "schema": SCHEMA,
        "dataset": result["dataset"],
        "method": method,
        "seed": int(result["seed"]),
        "tau_p": result.get("tau_p"),
        "tau_c": result.get("tau_c"),
        "source_result": str(result_path.resolve()),
        "classifier_path": scorer.classifier_path,
        "bounds_space": space,
        "n_samples": int(n_samples),
        "sampler_pool": "D_train, all classes (matches AnchorTabularExplainer(train_data=X_train))",
        "empirical_split": "D_test",
        "scoring_space": (
            "Boxes are scored in the space they were built in; unit and original "
            "boxes are never converted, because X_*_unit is clipped to [0,1] and "
            "the round trip moves test rows across box faces."
        ),
        "notes": (
            "Diagnostic only. Boxes are NOT re-selected; these are the rules the "
            "source artifact already selected on D_val. fid_emp recomputes the "
            "reported number as a cross-check (emp_matches_reported). "
            "fid_anchors is a faithful port of Ribeiro's sample_from_train. "
            "fid_crn is the repo's crn_perturb, which also perturbs rows that "
            "already satisfy the box, so it is a conservative lower bound and "
            "must not be labelled 'Anchors precision'."
        ),
        "per_class": per_class_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_dir", nargs="+",
        default=[
            str(REPO / "runs" / "paper_fiveseed_overlap075" / "results"),
            str(REPO / "runs" / "wyodot_fiveseed_overlap075" / "dnn" / "results"),
        ],
    )
    ap.add_argument("--out_dir", default=str(REPO / "revision" / "diagnostics" / "dual_estimator"))
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the samplers")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for d in args.results_dir:
        p = Path(d)
        if not p.is_dir():
            logger.warning("Not a directory, skipping: %s", p)
            continue
        paths.extend(sorted(p.rglob("*.json")))

    # Group by (dataset, seed) so each loader + classifier is built once.
    groups: Dict[Tuple[str, int], List[Path]] = defaultdict(list)
    skipped: List[str] = []
    for p in paths:
        try:
            r = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{p}: unreadable ({exc})")
            continue
        if "per_class" not in r or "dataset" not in r or "method" not in r:
            skipped.append(f"{p}: not a result artifact")
            continue
        # Instance-level artifacts come from revision.evaluate_instances and carry
        # per-instance boxes, not the class-level rules this diagnostic is about.
        if "__instances__" in p.name:
            skipped.append(f"{p}: instance-level artifact")
            continue
        if args.datasets and r["dataset"] not in args.datasets:
            continue
        if args.methods and r["method"] not in args.methods:
            continue
        groups[(r["dataset"], int(r["seed"]))].append(p)

    keys = sorted(groups)
    if args.limit:
        keys = keys[: args.limit]

    written: List[str] = []
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    for gi, key in enumerate(keys, 1):
        dataset, seed = key
        members = groups[key]
        clf = None
        for m in members:
            clf = _classifier_for(json.loads(m.read_text()), m)
            if clf:
                break
        if clf is None:
            failures.append(f"{dataset} seed{seed}: no classifier.pth found")
            continue
        logger.info("[%s/%s] %s seed=%s (%s artifacts)", gi, len(keys), dataset, seed, len(members))
        try:
            scorer = Scorer(dataset, seed, clf)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{dataset} seed{seed}: loader/classifier failed ({exc})")
            continue

        for m in members:
            rng = np.random.default_rng(args.seed)  # same draws for every method
            try:
                res = rescore_result(m, scorer, args.n_samples, rng)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{m}: {exc}")
                continue
            name = f"{res['dataset']}__{res['method']}__seed{res['seed']}.json"
            dest = out_dir / name
            dest.write_text(json.dumps(res, indent=2))
            written.append(str(dest))
            for cls_key, blk in res["per_class"].items():
                for r in blk["rules"]:
                    rows.append({
                        "dataset": res["dataset"],
                        "method": res["method"],
                        "seed": res["seed"],
                        "class": blk["target_class"],
                        "rule_id": r.get("rule_id"),
                        "n_features": r["n_features"],
                        "n_active": r["n_active"],
                        "mean_active_width_frac": r["mean_active_width_frac"],
                        "n_covered": r["n_covered"],
                        "fid_emp": r["fid_emp"],
                        "fid_anchors": r["fid_anchors"],
                        "fid_crn": r["fid_crn"],
                        "gap_anchors": r["gap_anchors"],
                        "gap_crn": r["gap_crn"],
                        "frac_synthetic": r["frac_synthetic"],
                        "fid_anchors_real_rows": r["fid_anchors_real_rows"],
                        "fid_anchors_hybrid_rows": r["fid_anchors_hybrid_rows"],
                        "emp_matches_reported": r["emp_matches_reported"],
                        "emp_abs_delta": r["emp_abs_delta"],
                    })

    csv_path = out_dir / "rescore_rules.csv"
    if rows:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    (out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": SCHEMA + "_manifest",
        "results_dirs": [str(Path(d).resolve()) for d in args.results_dir],
        "n_samples": args.n_samples,
        "sampler_rng_seed": args.seed,
        "n_artifacts_written": len(written),
        "n_rules_scored": len(rows),
        "skipped": skipped,
        "failures": failures,
    }, indent=2))

    logger.info("Wrote %s artifacts, %s scored rules -> %s", len(written), len(rows), out_dir)
    if failures:
        logger.error("%s failure(s); see run_manifest.json", len(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
