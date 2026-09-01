#!/usr/bin/env python3
"""Fit one black-box classifier per (dataset, seed) and save classifier.pth.

Class-parallel RL shards and DDPG vs SAC must load this file. Per-shard
retraining is not bit-identical (DataLoader shuffle + last-writer race on
training/classifier.pth), so the policy can be trained against a different
ŷ than inference/baselines use.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.dataset_factory import make_tabular_loader  # noqa: E402


def classifier_patience(dataset: str) -> int:
    """Must match single_agent/driver.py."""
    name = dataset.lower()
    if name.startswith("folktables_") or name.startswith("uci_"):
        return 200
    return {
        "housing": 200,
        "breast_cancer": 100,
        "wine": 100,
        "iris": 50,
    }.get(name, 50)


def fit_classifier_to_path(
    dataset: str,
    dest: Path,
    seed: int = 42,
    device: str = "cpu",
    classifier_type: str = "dnn",
    epochs: int = 500,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loader = make_tabular_loader(dataset, random_state=seed, test_size=0.2)
    loader.load_dataset()
    loader.preprocess_data()
    clf = loader.create_classifier(
        classifier_type=classifier_type, use_batch_norm=True, device=device,
    )
    trained, test_acc, _ = loader.train_classifier(
        clf,
        epochs=epochs,
        batch_size=256,
        lr=1e-3,
        patience=classifier_patience(dataset),
        weight_decay=1e-4,
        use_lr_scheduler=True,
        device=device,
    )
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    loader.save_classifier(trained, str(tmp))
    os.replace(tmp, dest)
    meta = dest.with_suffix(".acc.txt")
    meta.write_text(f"dataset={dataset} seed={seed} test_acc={test_acc:.6f}\n")
    return dest


def fit_or_load(
    dataset: str,
    dest: Path,
    seed: int = 42,
    device: str = "cpu",
    classifier_type: str = "dnn",
    epochs: int = 500,
) -> Path:
    """One writer; other processes block on the lock and then load dest."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    lock_path = dest.with_name(dest.name + ".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        return fit_classifier_to_path(
            dataset, dest, seed=seed, device=device,
            classifier_type=classifier_type, epochs=epochs,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Fit one shared classifier.pth")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--classifier_type", default="dnn")
    p.add_argument("--epochs", type=int, default=500)
    args = p.parse_args()
    path = fit_or_load(
        args.dataset, Path(args.out), seed=args.seed, device=args.device,
        classifier_type=args.classifier_type, epochs=args.epochs,
    )
    print(f"classifier ready: {path}", flush=True)


if __name__ == "__main__":
    main()
