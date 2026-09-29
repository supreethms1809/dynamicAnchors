"""Rebuild stored rule boxes on D_val and D_test, without re-running any method.

Every result JSON (RL arms and baselines) stores each selected rule's bounds and
its row counts on the selection split (`selection_metrics`, D_val) and the
reporting split (`report_metrics`, D_test). This module rebuilds the per-rule
masks on both splits and checks them against those counts, so a re-score from
the stored boxes is verified row for row before anything is reported.

Why the ulp repair: bounds are stored as float32 that was rounded on the machine
that trained the policy. Recomputing X_unit on another machine can land a coded
feature value one or two float32 ulps on the other side of a box face. On
folktables seed 45 every California-born row (POBP = 6, ~40% of the data) sat
1 ulp below the stored lower bound, which moved RLDA coverage from 0.655 to
0.488. A rule is therefore widened by up to `MAX_ULP` float32 ulps per face,
and only when that makes BOTH stored counts match; a flat tolerance (1e-6)
breaks cells that were exact (housing, breast_cancer).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.abstention_paper_final import classifier_path  # noqa: E402
from revision.cov_tau_class_gated import (  # noqa: E402
    ORIGINAL_UNIT_METHODS, _loader_for, _predictions,
)
from utils.eval_harness import evaluate_ruleset_as_classifier  # noqa: E402

MAX_ULP = 8
# Near 0 a float32 ulp is ~1e-45, far below the ~1e-8 absolute error of
# recomputing (x - min) / range, so faces are also tried at small absolute
# widths. Real gaps between distinct feature values are >= 1e-3 in unit space.
ABS_EPS = (1e-8, 3e-8, 1e-7, 3e-7)
RESULTS = REPO.parent / "results" / "paper_final"
TREES = ("emp_tc0p10", "emp_tc0p20", "pert_tc0p10", "pert_tc0p20")
METHODS = ("rlda", "mada", "cart", "random_search", "sp_anchors", "greedy_anchors")


@dataclass
class Split:
    X_unit: np.ndarray
    X_orig: np.ndarray
    y: np.ndarray
    y_hat: np.ndarray

    def X(self, method: str) -> np.ndarray:
        return self.X_orig if method in ORIGINAL_UNIT_METHODS else self.X_unit


@dataclass
class SeedData:
    dataset: str
    seed: int
    loader: Any
    val: Split
    test: Split
    classifier: Path


def load_seed_data(dataset: str, seed: int, clf: Optional[Path] = None) -> SeedData:
    clf = clf or classifier_path(dataset, seed)
    if clf is None:
        raise FileNotFoundError(f"no classifier for {dataset} seed {seed}")
    loader = _loader_for(dataset, seed)
    loader.classifier = loader.load_classifier(filepath=str(clf), device="cpu")

    def split(unit, orig, scaled, y):
        return Split(
            X_unit=np.asarray(unit, dtype=np.float32),
            X_orig=np.asarray(orig, dtype=np.float32),
            y=np.asarray(y),
            y_hat=_predictions(loader, scaled),
        )

    return SeedData(
        dataset, seed, loader,
        val=split(loader.X_val_unit, loader.X_val, loader.X_val_scaled, loader.y_val),
        test=split(loader.X_test_unit, loader.X_test, loader.X_test_scaled, loader.y_test),
        classifier=Path(clf),
    )


def cell_classifier(cell: Dict[str, Any]) -> Optional[Path]:
    """The classifier a cell was scored with, when it is on this machine.

    RL cells point at their rules file; the leg's classifier sits beside its
    output/ folder (the wyodot ablations for seeds 44-46 were trained on the Mac
    with their own classifier). Baselines record classifier_path directly.
    """
    extra = cell.get("extra") or {}
    rf = str(extra.get("rules_file") or "")
    if "/output/" in rf:
        p = Path(rf.split("/output/")[0]) / "classifiers" / f"{cell['dataset']}_seed{cell['seed']}.pth"
        if p.is_file():
            return p
    cp = extra.get("classifier_path")
    return Path(cp) if cp and Path(cp).is_file() else None


def with_classifier(sd: SeedData, clf: Optional[Path]) -> SeedData:
    """Same splits, f_hat from another checkpoint (no dataset reload)."""
    if clf is None or clf.resolve() == sd.classifier.resolve():
        return sd
    loader = sd.loader
    loader.classifier = loader.load_classifier(filepath=str(clf), device="cpu")

    def split(s: Split, scaled) -> Split:
        return Split(s.X_unit, s.X_orig, s.y, _predictions(loader, scaled))

    return SeedData(
        sd.dataset, sd.seed, loader,
        val=split(sd.val, loader.X_val_scaled),
        test=split(sd.test, loader.X_test_scaled),
        classifier=Path(clf),
    )


def result_path(tree: str, method: str, dataset: str, seed: int,
                root: Path = RESULTS, tc: Optional[str] = None) -> Path:
    """Cell path inside a paper_final-style tree ({emp,pert}_tc0pXX)."""
    est, tc_tree = tree.split("_")
    tc = tc or tc_tree
    name = f"{dataset}__{method}__seed{seed}__tp0p90__{tc}.json"
    if method in ("rlda", "mada"):
        algo = {"rlda": "ddpg", "mada": "maddpg"}[method]
        return root / tree / "results" / algo / name
    return root / f"baselines_{est}" / name


def _widen(b, k: int, direction: float) -> np.ndarray:
    b = np.asarray(b, dtype=np.float32)
    for _ in range(k):
        b = np.nextafter(b, np.float32(direction * np.inf))
    return b


def _box(X: np.ndarray, lo: np.ndarray, up: np.ndarray) -> np.ndarray:
    return np.all((X >= lo) & (X <= up), axis=1)


@dataclass
class ClassRules:
    cls: int
    val_masks: List[np.ndarray]
    test_masks: List[np.ndarray]
    stored_test_union_fid: float
    # the float32 box each mask was taken with (after the face repair)
    boxes: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def val_union(self) -> np.ndarray:
        return np.logical_or.reduce(self.val_masks)

    @property
    def test_union(self) -> np.ndarray:
        return np.logical_or.reduce(self.test_masks)


@dataclass
class Rebuilt:
    classes: Dict[int, ClassRules]
    n_rules: int = 0
    n_widened: int = 0
    failures: List[str] = field(default_factory=list)


def rebuild(cell: Dict[str, Any], sd: SeedData) -> Rebuilt:
    method = cell["method"]
    Xv, Xt = sd.val.X(method), sd.test.X(method)
    out = Rebuilt(classes={})
    for key, block in (cell.get("per_class") or {}).items():
        if not isinstance(block, dict):
            continue
        union = block.get("union") or {}
        cls = union.get("target_class")
        if cls is None:
            cls = int(str(key).split("_")[-1])
        vms, tms, bxs = [], [], []
        for rule in block.get("selected_rules") or []:
            lo0, up0 = rule.get("lower_bounds"), rule.get("upper_bounds")
            if lo0 is None or up0 is None:
                continue
            want_v = (rule.get("selection_metrics") or {}).get("n_covered")
            want_t = (rule.get("report_metrics") or {}).get("n_covered")
            out.n_rules += 1
            lo_f = np.asarray(lo0, dtype=np.float32)
            up_f = np.asarray(up0, dtype=np.float32)
            widths = [(_widen(lo0, k, -1), _widen(up0, k, 1)) for k in range(MAX_ULP + 1)]
            widths += [(lo_f - np.float32(e), up_f + np.float32(e)) for e in ABS_EPS]
            for i, (lo, up) in enumerate(widths):
                vm, tm = _box(Xv, lo, up), _box(Xt, lo, up)
                if (want_v is None or vm.sum() == want_v) and (want_t is None or tm.sum() == want_t):
                    break
            else:
                i = 0
                vm, tm = _box(Xv, lo_f, up_f), _box(Xt, lo_f, up_f)
                out.failures.append(
                    f"{key}:{rule.get('rule_id', '?')} val {int(vm.sum())}/{want_v} "
                    f"test {int(tm.sum())}/{want_t}"
                )
            out.n_widened += int(i > 0)
            vms.append(vm)
            tms.append(tm)
            bxs.append(widths[i])
        if not vms:
            continue
        f = union.get("fidelity")
        out.classes[int(cls)] = ClassRules(
            int(cls), vms, tms,
            float(f) if f is not None and np.isfinite(f) else -np.inf,
            bxs,
        )
    return out


def union_fid(mask: np.ndarray, y_hat: np.ndarray, cls: int) -> float:
    """Class-union fidelity on a split (the union Fid `evaluate` stores); -inf when empty."""
    return float((y_hat[mask] == cls).mean()) if mask.any() else -np.inf


def tiebreak_fids(rb: Rebuilt, sd: SeedData, tiebreak: str) -> Dict[int, float]:
    """Class-union Fid used to settle conflicts: 'val' (D_val, leak-free),
    'test' (rebuilt D_test), or 'stored' (the D_test value in the JSON)."""
    if tiebreak == "stored":
        return {c: cr.stored_test_union_fid for c, cr in rb.classes.items()}
    if tiebreak == "val":
        return {c: union_fid(cr.val_union, sd.val.y_hat, c) for c, cr in rb.classes.items()}
    if tiebreak == "test":
        return {c: union_fid(cr.test_union, sd.test.y_hat, c) for c, cr in rb.classes.items()}
    raise ValueError(tiebreak)


def global_result(rb: Rebuilt, sd: SeedData, which: str, tiebreak: str):
    """Score the rule set as a classifier on the `which` split ('val' / 'test')."""
    split = sd.val if which == "val" else sd.test
    masks = {c: (cr.val_union if which == "val" else cr.test_union) for c, cr in rb.classes.items()}
    return evaluate_ruleset_as_classifier(masks, tiebreak_fids(rb, sd, tiebreak), split.y, split.y_hat)


def summary_dict(res) -> Dict[str, float]:
    d = res.to_dict()
    return {k: d[k] for k in (
        "n_eval", "n_decided", "n_conflict", "n_fid_agree", "n_pur_agree",
        "global_fidelity", "global_purity", "coverage", "conflict_rate", "effectiveness",
    )}


def load_cell(path: Path) -> Optional[Dict[str, Any]]:
    return json.loads(path.read_text()) if path.is_file() else None
