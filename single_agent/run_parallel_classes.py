"""
Launcher: runs single_agent/driver.py with target classes sharded across N parallel
subprocesses, all writing into one shared experiment folder.

Combine with driver.py's --n_envs flag for two-level parallelism:
    --parallel_classes K --n_envs M  =>  K class workers, each running M vec envs

Examples:
    # M4 Pro: 4 class shards x 2 envs each = 8 cores
    python single_agent/run_parallel_classes.py \\
        --dataset wyodot_testbed --algorithm sac --n_classes 4 \\
        --parallel_classes 4 --n_envs 2 --total_timesteps 1500000

    # 5800X (8C/16T): 4 class shards x 3 envs each = 12 cores
    python single_agent/run_parallel_classes.py \\
        --dataset wyodot_testbed --algorithm sac --n_classes 4 \\
        --parallel_classes 4 --n_envs 3 --total_timesteps 1500000

Notes:
    - All shards share one experiment folder (passed via --experiment_folder_override).
    - One classifier.pth is fit (or loaded) BEFORE shards spawn. Every class worker
      gets --classifier_path pointing at that file. Do not retrain the black box
      per class: independent DNN fits are not bit-identical, and shards race on
      training/classifier.pth, so inference can score a different ŷ than training.
    - Per-shard stdout/stderr is captured into a log file inside the shared folder.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def peel_classifier_path(extras):
    """Remove --classifier_path/--classifier-path from extras; return (path, rest)."""
    extras = list(extras or [])
    if extras and extras[0] == "--":
        extras = extras[1:]
    path = None
    rest = []
    i = 0
    while i < len(extras):
        tok = extras[i]
        if tok in ("--classifier_path", "--classifier-path") and i + 1 < len(extras):
            path = extras[i + 1]
            i += 2
            continue
        rest.append(tok)
        i += 1
    return path, rest


def _ensure_shared_classifier(args, shared_folder, extras_path):
    """Return a classifier.pth that every shard will load (never retrain)."""
    dest = os.path.join(shared_folder, "classifier.pth")
    parent_dest = os.path.join(args.output_dir, "training", "classifier.pth")
    os.makedirs(os.path.dirname(parent_dest), exist_ok=True)

    src = extras_path
    if not src or not os.path.exists(src):
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from revision.fit_shared_classifier import fit_or_load
        epochs = args.classifier_epochs if args.classifier_epochs is not None else 500
        print(f"[launcher] fitting ONE classifier for {args.dataset} -> {dest}", flush=True)
        fit_or_load(
            args.dataset, Path(dest), seed=args.seed, device=args.device,
            classifier_type=args.classifier_type, epochs=epochs,
        )
        src = dest
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy2(src, dest)
    if os.path.abspath(src) != os.path.abspath(parent_dest):
        shutil.copy2(src, parent_dest)
    print(f"[launcher] shared classifier: {dest}", flush=True)
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="Parallel-class launcher for single_agent/driver.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", default="sac", choices=["ddpg", "sac"])
    parser.add_argument(
        "--parallel_classes", type=int, default=2,
        help="Number of parallel class-worker processes."
    )
    cls_group = parser.add_mutually_exclusive_group(required=True)
    cls_group.add_argument(
        "--target_classes", type=int, nargs="+",
        help="Explicit list of class IDs to train."
    )
    cls_group.add_argument(
        "--n_classes", type=int,
        help="Convenience: train classes [0..N)."
    )

    # Forwarded to driver.py
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max_cycles", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--classifier_type", default="dnn",
        choices=["dnn", "random_forest", "gradient_boosting"]
    )
    parser.add_argument("--classifier_epochs", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--classifier_path", default=None,
        help="Pre-trained classifier.pth shared by every class shard. "
             "If omitted, one classifier is fit into the experiment folder first.",
    )
    parser.add_argument(
        "--eval_on_test_data", action="store_true",
        help="Forward --eval_on_test_data to each shard."
    )
    parser.add_argument(
        "--extra_args", nargs=argparse.REMAINDER, default=[],
        help="Any other driver.py args (place after `--`)."
    )

    args = parser.parse_args()

    target_classes = args.target_classes if args.target_classes is not None \
        else list(range(args.n_classes))

    if args.output_dir is None:
        args.output_dir = f"./output/single_agent_sb3_{args.dataset}_{args.algorithm}/"

    # One shared experiment folder for all shards. Must match the layout the
    # trainer would generate so downstream tools (inference, etc.) still find it.
    timestamp = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    shared_folder = os.path.join(
        args.output_dir, "training",
        f"{args.algorithm}_single_agent_sb3_{timestamp}",
    )
    os.makedirs(shared_folder, exist_ok=True)

    K = max(1, args.parallel_classes)
    # Round-robin shard so class counts are balanced across workers.
    shards = [target_classes[i::K] for i in range(K)]
    shards = [s for s in shards if s]

    extras_path, extras = peel_classifier_path(args.extra_args)
    clf_src = args.classifier_path or extras_path
    clf_path = _ensure_shared_classifier(args, shared_folder, clf_src)

    print(f"[launcher] dataset={args.dataset} algorithm={args.algorithm}")
    print(f"[launcher] shared experiment folder: {shared_folder}")
    print(f"[launcher] classes: {target_classes}")
    print(f"[launcher] workers: {len(shards)} (each: n_envs={args.n_envs})")
    print(f"[launcher] shards: {shards}")
    print(f"[launcher] classifier_path (all shards load, none refit): {clf_path}")

    driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "driver.py")

    procs = []
    for shard_idx, shard in enumerate(shards):
        cmd = [
            sys.executable, driver_path,
            "--dataset", args.dataset,
            "--algorithm", args.algorithm,
            "--seed", str(args.seed),
            "--n_envs", str(args.n_envs),
            "--device", args.device,
            "--target_classes", *[str(c) for c in shard],
            "--output_dir", args.output_dir,
            "--experiment_folder_override", shared_folder,
            "--classifier_type", args.classifier_type,
            "--classifier_path", clf_path,
        ]
        if args.total_timesteps is not None:
            cmd += ["--total_timesteps", str(args.total_timesteps)]
        if args.max_cycles is not None:
            cmd += ["--max_cycles", str(args.max_cycles)]
        if args.learning_rate is not None:
            cmd += ["--learning_rate", str(args.learning_rate)]
        if args.classifier_epochs is not None:
            cmd += ["--classifier_epochs", str(args.classifier_epochs)]
        if args.eval_on_test_data:
            cmd += ["--eval_on_test_data"]
        if extras:
            cmd += extras

        log_name = f"shard_{shard_idx}_classes_{'_'.join(map(str, shard))}.log"
        log_path = os.path.join(shared_folder, log_name)
        log_f = open(log_path, "w")
        print(f"[launcher] shard {shard_idx}: classes={shard} log={log_path}")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((p, log_f, shard))

    failures = []
    for p, log_f, shard in procs:
        ret = p.wait()
        log_f.close()
        status = "OK" if ret == 0 else f"FAILED (exit {ret})"
        print(f"[launcher] shard classes={shard}: {status}")
        if ret != 0:
            failures.append(shard)

    if failures:
        sys.exit(f"[launcher] {len(failures)} shard(s) failed: {failures}")
    print(f"[launcher] all shards complete -> {shared_folder}")


if __name__ == "__main__":
    main()
