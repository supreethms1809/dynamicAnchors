import argparse
import os
import sys
import numpy as np

# Support running as a script or as a module
if __package__ is None or __package__ == "":
    # Add project root to sys.path: .../xai
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _pkg_root = os.path.dirname(os.path.dirname(_this_dir))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from model_agnostic.dynamic_anchors.dyn_anchor import train_dynamic_anchors
    from model_agnostic.dynamic_anchors.static_anchor import run_static_anchors
else:
    from .dyn_anchor import train_dynamic_anchors
    from .static_anchor import run_static_anchors


def parse_nullable_int(v: str | None) -> int | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nan", ""):  # accept common None spellings
        return None
    return int(v)


def parse_nullable_float(v: str | None) -> float | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("none", "null", "nan", ""):
        return None
    return float(v)


def summarize_dynamic_per_class(dynamic_out: dict, reduce: str = "last", last_k: int = 1) -> dict:
    per_class_hist = dynamic_out.get("per_class_precision_coverage_history", {})
    summary = {}
    for cls, series in per_class_hist.items():
        if not series:
            continue
        if reduce == "last":
            vals = [series[-1]]
        elif reduce == "last_k" and last_k > 1:
            vals = series[-last_k:]
        else:  # mean across all episodes (legacy)
            vals = series
        # Use hard precision for fair comparison with static anchors' thresholded precision
        prec = float(np.mean([d.get("hard_precision", 0.0) for d in vals]))
        cov = float(np.mean([d.get("coverage", 0.0) for d in vals]))
        summary[int(cls)] = {"avg_precision": prec, "avg_coverage": cov, "episodes": len(vals)}
    return summary

def summarize_dynamic_greedy(dynamic_out: dict) -> dict:
    greedy = dynamic_out.get("final_greedy", {})
    summary = {}
    for cls, d in greedy.items():
        total_anchors = d.get("total_anchors", d.get("num_rollouts", 1))
        num_instances = d.get("num_instances", 0)
        # Total anchors = number of instances (one anchor per instance)
        summary[int(cls)] = {
            "avg_precision": float(d.get("hard_precision", d.get("precision", 0.0))),
            "avg_coverage": float(d.get("coverage", 0.0)),
            "episodes": total_anchors,  # Total anchors = number of test instances
            "num_instances": num_instances,
        }
    return summary


def summarize_static_per_class(static_out: dict) -> dict:
    per_class = static_out.get("per_class_results", {})
    summary = {}
    for cls, series in per_class.items():
        if not series:
            continue
        avg_prec = float(np.mean([d.get("precision", 0.0) for d in series]))
        avg_cov = float(np.mean([d.get("coverage", 0.0) for d in series]))
        summary[int(cls)] = {"avg_precision": avg_prec, "avg_coverage": avg_cov, "n_instances": len(series)}
    return summary

def summarize_static_best(static_out: dict) -> dict:
    """Get best static anchor per class (single evaluation, for apples-to-apples with greedy dynamic)."""
    per_class = static_out.get("per_class_results", {})
    summary = {}
    for cls, series in per_class.items():
        if not series:
            continue
        # Best = highest precision, then highest coverage
        best = sorted(series, key=lambda d: (d.get('precision', 0.0), d.get('coverage', 0.0)), reverse=True)[0]
        summary[int(cls)] = {
            "avg_precision": float(best.get("precision", 0.0)),
            "avg_coverage": float(best.get("coverage", 0.0)),
            "n_instances": 1,
        }
    return summary


def run_compare(
    dataset: str = "covtype",
    seed: int = 42,
    device: str = "auto",
    precision_target: float | None = None,
    coverage_target: float | None = None,
    # Dynamic optional overrides (None => auto presets in dyn_anchor)
    episodes: int | None = None,
    steps: int | None = None,
    classifier_epochs: int | None = None,
    # Static optional overrides (None => auto presets in static_anchor)
    threshold: float | None = None,
    per_class: int | None = None,
    static_classifier_epochs: int | None = None,
    # Dynamic discretization options
    dynamic_discretize: bool = True,
    disc_perc: list[int] | None = None,
    show_plots: bool = True,
    num_greedy_rollouts: int = 20,  # Number of test instances per class (one rollout per instance, like static)
    num_test_instances_per_class: int | None = None,  # Override number of test instances (None = use num_greedy_rollouts)
    max_features_in_rule: int = 5,  # Maximum number of features to show in dynamic anchor rules
    max_anchor_size: int | None = None,  # Maximum number of features in static anchors (None = no limit, uses default greedy behavior)
):
    # Align static threshold to dynamic precision_target if threshold not provided
    if threshold is None and precision_target is not None:
        threshold = float(precision_target)

    # Run static baseline
    static_out = run_static_anchors(
        dataset=dataset,
        seed=seed,
        device_preference=device,
        anchor_threshold=threshold,
        num_instances_per_class=per_class,
        classifier_epochs=static_classifier_epochs,
        disc_perc=None,
        coverage_target=coverage_target,
        max_anchor_size=max_anchor_size,
    )

    # Get number of instances used by static (for matching dynamic)
    static_results = static_out.get("per_class_results", {})
    num_instances_used = None
    if static_results:
        # Find max instances across all classes
        max_instances = max(len(series) for series in static_results.values() if series)
        if max_instances > 0:
            num_instances_used = max_instances
    
    # Use static's number of instances if not specified, otherwise use provided value
    dynamic_num_instances = num_test_instances_per_class if num_test_instances_per_class is not None else num_instances_used

    # Run dynamic method
    dyn_out = train_dynamic_anchors(
        dataset=dataset,
        seed=seed,
        device_preference=device,
        precision_target=precision_target,
        coverage_target=coverage_target,
        use_discretization=dynamic_discretize,
        disc_perc=disc_perc,
        bin_edges=static_out.get("bin_edges") if dynamic_discretize else None,
        episodes=episodes,
        steps_per_episode=steps,
        classifier_epochs_per_round=classifier_epochs,
        target_classes=None,
        entropy_coef=None,
        value_coef=None,
        reg_lambda_inside_anchor=None,
        use_perturbation=None,
        perturbation_mode=None,
        n_perturb=None,
        initial_window=None,
        show_plots=show_plots,
        num_greedy_rollouts=num_greedy_rollouts,
        num_test_instances_per_class=dynamic_num_instances,
        max_features_in_rule=max_features_in_rule,
    )

    class_names = static_out.get("class_names", [])

    # Prefer greedy evaluation if available; fallback to last-5 episodes
    dyn_summary = summarize_dynamic_greedy(dyn_out)
    if dyn_summary:
        # Check if greedy uses multiple anchors (multiple instances × rollouts) or single
        greedy_data = dyn_out.get("final_greedy", {})
        total_anchors = any(d.get("total_anchors", d.get("num_rollouts", 1)) > 1 for d in greedy_data.values())
        if total_anchors:
            # If using multiple anchors (instances × rollouts), compare with static average for apples-to-apples
            static_summary = summarize_static_per_class(static_out)
        else:
            # If using single anchor, compare with best static (single evaluation) for apples-to-apples
            static_summary = summarize_static_best(static_out)
    else:
        # If using episode average, compare with static average
        dyn_summary = summarize_dynamic_per_class(dyn_out, reduce="last_k", last_k=5)
        static_summary = summarize_static_per_class(static_out)

    # Print final bounding boxes/anchors before comparison
    final_greedy = dyn_out.get("final_greedy", {})
    if final_greedy:
        for cls, fg in final_greedy.items():
            cls_i = int(cls)
            cls_name = class_names[cls_i] if cls_i < len(class_names) else str(cls_i)
            rule = fg.get("rule", "")
            lower = fg.get("lower")
            upper = fg.get("upper")
            total_anchors = fg.get('total_anchors', fg.get('num_rollouts', 1))
            num_instances = fg.get('num_instances', 0)
            if total_anchors > 1:
                rollout_str = f" (avg over {total_anchors} instances)"
            else:
                rollout_str = ""
            # Format rule as list like static anchors
            if rule and rule != "any values (no tightened features)":
                rule_list = [r.strip() for r in rule.split(" and ") if r.strip()]
            else:
                rule_list = []
            
            if rule_list:
                print(f"[dyn final cls={cls_i}] {cls_name} | anchor={rule_list} | hard_precision={fg.get('hard_precision', fg.get('precision', 0.0)):.3f} | coverage={fg.get('coverage', 0.0):.3f}{rollout_str}")
            else:
                print(f"[dyn final cls={cls_i}] {cls_name} | anchor=[] | hard_precision={fg.get('hard_precision', fg.get('precision', 0.0)):.3f} | coverage={fg.get('coverage', 0.0):.3f}{rollout_str}")

    # For static, pick a representative final anchor per class (best precision then coverage)
    static_results = static_out.get("per_class_results", {})
    for cls, series in static_results.items():
        if not series:
            continue
        best = sorted(series, key=lambda d: (d.get('precision', 0.0), d.get('coverage', 0.0)), reverse=True)[0]
        cls_i = int(cls)
        cls_name = class_names[cls_i] if cls_i < len(class_names) else str(cls_i)
        names = best.get('anchor', [])
        print(f"[static final cls={cls_i}] {cls_name} | anchor={names} | precision={best.get('precision', 0.0):.3f} | coverage={best.get('coverage', 0.0):.3f}")

    # Main comparison (greedy dynamic vs best static, or average dynamic vs average static)
    print("\n=== Main Comparison (best/best or avg/avg) ===")
    all_classes = sorted(set(dyn_summary.keys()) | set(static_summary.keys()))
    for cls in all_classes:
        ds = dyn_summary.get(cls, {"avg_precision": 0.0, "avg_coverage": 0.0, "episodes": 0})
        ss = static_summary.get(cls, {"avg_precision": 0.0, "avg_coverage": 0.0, "n_instances": 0})
        cls_name = class_names[cls] if cls < len(class_names) else str(cls)
        dyn_episodes = ds.get("episodes", 0)
        dyn_type = "avg" if dyn_episodes > 1 else "single"
        static_instances = ss.get("n_instances", 0)
        static_type = "avg" if static_instances > 1 else "best"
        dyn_note = f" (n={dyn_episodes})" if dyn_episodes > 1 else ""
        static_note = f" (n={static_instances})" if static_instances > 1 else ""
        print(
            f"[compare cls={cls}] {cls_name} | "
            f"dyn ({dyn_type}{dyn_note}): precision={ds['avg_precision']:.3f}, coverage={ds['avg_coverage']:.3f} | "
            f"static ({static_type}{static_note}): precision={ss['avg_precision']:.3f}, coverage={ss['avg_coverage']:.3f}"
        )

    return {"dynamic": dyn_summary, "static": static_summary, "class_names": class_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dynamic vs static anchors comparison")
    parser.add_argument("--dataset", type=str, default="covtype", choices=["breast_cancer", "synthetic", "covtype"], help="Dataset to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device preference")
    parser.add_argument("--precision_target", type=parse_nullable_float, default=None, help="Unified precision target for both methods (None=auto)")
    parser.add_argument("--coverage_target", type=parse_nullable_float, default=None, help="Unified coverage target for both methods (None=auto)")

    # Dynamic overrides
    parser.add_argument("--episodes", type=parse_nullable_int, default=None, help="Dynamic: episodes (None=auto)")
    parser.add_argument("--steps", type=parse_nullable_int, default=None, help="Dynamic: steps per episode (None=auto)")
    parser.add_argument("--classifier_epochs", type=parse_nullable_int, default=None, help="Dynamic: classifier epochs per round (None=auto)")

    # Static overrides
    parser.add_argument("--threshold", type=parse_nullable_float, default=None, help="Static: anchor precision threshold (None=auto)")
    parser.add_argument("--per_class", type=parse_nullable_int, default=None, help="Static: instances per class (None=auto)")
    parser.add_argument("--static_classifier_epochs", type=parse_nullable_int, default=None, help="Static: classifier epochs (None=auto)")
    # Dynamic discretization (enabled by default); provide a flag to disable
    parser.add_argument("--no_dynamic_discretize", dest="dynamic_discretize", action="store_false", help="Disable discretized dynamic anchors")
    parser.add_argument("--disc_perc", type=parse_nullable_int, nargs='*', default=None, help="Percentiles for dynamic discretization (None=auto)")
    parser.set_defaults(dynamic_discretize=True)
    parser.add_argument("--show_plots", action="store_true", default=True, help="Enable visualization plots (default: True)")
    parser.add_argument("--no-plots", dest="show_plots", action="store_false", help="Disable visualization plots")
    parser.add_argument("--num_greedy_rollouts", type=int, default=20, help="Dynamic: number of test instances per class (one anchor per instance, default: 20)")
    parser.add_argument("--num_test_instances", type=parse_nullable_int, default=None, help="Dynamic: override number of test instances per class (None = use num_greedy_rollouts)")
    parser.add_argument("--max_features_in_rule", type=int, default=5, help="Dynamic: maximum number of features to show in anchor rules (default: 5, use 0 for all features)")
    parser.add_argument("--max_anchor_size", type=parse_nullable_int, default=None, help="Static: maximum number of features in anchor explanations (None = no limit, default: None)")

    args = parser.parse_args()

    run_compare(
        dataset=args.dataset,
        seed=args.seed,
        device=args.device,
        precision_target=args.precision_target,
        coverage_target=args.coverage_target,
        episodes=args.episodes,
        steps=args.steps,
        classifier_epochs=args.classifier_epochs,
        threshold=args.threshold,
        per_class=args.per_class,
        static_classifier_epochs=args.static_classifier_epochs,
        dynamic_discretize=args.dynamic_discretize,
        disc_perc=args.disc_perc,
        show_plots=args.show_plots,
        num_greedy_rollouts=args.num_greedy_rollouts,
        num_test_instances_per_class=args.num_test_instances,
        max_features_in_rule=args.max_features_in_rule,
        max_anchor_size=args.max_anchor_size,
    )
