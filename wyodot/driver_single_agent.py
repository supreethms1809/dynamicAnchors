"""
WyoDOT Single-Agent Dynamic Anchors Training Pipeline (Stable-Baselines3)

Usage:
python driver_single_agent.py --dataset wyodot_kvdw_labeled --algorithm ddpg --seed 42

Datasets: wyodot_kvdw_labeled, wyodot_testbed
Algorithms: ddpg, sac
Classifier: random_forest (default, with paper params), dnn, gradient_boosting
"""

import sys
import os
# Add project root, single_agent, and BenchMARL to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "single_agent"))

from wyodot_dataset_loader import WyoDOTDatasetLoader
from anchor_trainer_sb3 import AnchorTrainerSB3
import argparse
import random
import numpy as np
import torch
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    available_algorithms = list(AnchorTrainerSB3.ALGORITHM_MAP.keys())
    if not available_algorithms:
        raise RuntimeError("No algorithms available. Make sure Stable-Baselines3 is properly installed.")

    parser = argparse.ArgumentParser(description="WyoDOT Single-Agent Anchor Training Pipeline (SB3)")

    parser.add_argument(
        "--dataset", type=str, default="wyodot_kvdw_labeled",
        choices=list(WyoDOTDatasetLoader.DATASETS.keys()),
        help="WyoDOT dataset to use"
    )
    parser.add_argument(
        "--algorithm", type=str,
        default="ddpg" if "ddpg" in available_algorithms else available_algorithms[0],
        choices=available_algorithms,
        help=f"RL algorithm. Available: {', '.join(available_algorithms)}"
    )
    parser.add_argument(
        "--classifier_type", type=str, default="random_forest",
        choices=["dnn", "random_forest", "gradient_boosting"],
        help="Type of classifier (default: random_forest with paper params)"
    )
    parser.add_argument("--classifier_epochs", type=int, default=500, help="Classifier training epochs")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_eda", action="store_true", help="Skip EDA analysis")
    parser.add_argument("--skip_classifier", action="store_true", help="Skip classifier training")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"],
        help="Device to use"
    )
    parser.add_argument("--eval_on_test_data", action="store_true", help="Evaluate on test data")
    parser.add_argument("--max_cycles", type=int, default=None, help="Max cycles per episode")
    parser.add_argument(
        "--target_classes", type=int, nargs="+", default=None,
        help="Target classes to generate anchors for (default: all)"
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None,
        help="Path to checkpoint to load (skips training)"
    )
    parser.add_argument("--total_timesteps", type=int, default=320_000, help="Total training timesteps")
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Learning rate (default: dataset-specific)"
    )

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir is None:
        args.output_dir = f"./output/single_agent_sb3_{args.dataset}_{args.algorithm}/"

    print("=" * 80)
    if args.load_checkpoint:
        print("WYODOT SINGLE-AGENT EVALUATION MODE (Loading from checkpoint)")
    else:
        print("WYODOT SINGLE-AGENT ANCHOR TRAINING PIPELINE (SB3)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Classifier: {args.classifier_type}")
    print(f"Output directory: {args.output_dir}")
    if args.load_checkpoint:
        print(f"Checkpoint: {args.load_checkpoint}")
    print("=" * 80)

    # Load dataset
    dataset_loader = WyoDOTDatasetLoader(
        dataset_name=args.dataset,
        test_size=0.2,
        random_state=args.seed
    )
    dataset_loader.load_dataset()

    if not args.skip_eda:
        dataset_loader.perform_eda_analysis(
            output_dir=f"{args.output_dir}eda/",
            use_ydata_profiling=True
        )

    dataset_loader.preprocess_data()

    # Train classifier
    if args.load_checkpoint:
        logger.info("\nSkipping classifier training (will load from checkpoint)")
    elif not args.skip_classifier:
        classifier = dataset_loader.create_classifier(
            classifier_type=args.classifier_type,
            device=args.device
        )

        trained_classifier, test_acc, history = dataset_loader.train_classifier(
            classifier,
            epochs=args.classifier_epochs,
            batch_size=256,
            lr=1e-3,
            patience=50,
            weight_decay=1e-4,
            use_lr_scheduler=True,
            device=args.device
        )

        logger.info(f"\nClassifier trained with test accuracy: {test_acc:.4f}")

        classifier_output_dir = f"{args.output_dir}training/"
        os.makedirs(classifier_output_dir, exist_ok=True)
        classifier_path = os.path.join(classifier_output_dir, "classifier.pth")
        dataset_loader.save_classifier(trained_classifier, classifier_path)
        logger.info(f"Classifier saved to: {classifier_path}")
    else:
        if dataset_loader.classifier is None:
            raise ValueError("Classifier not found. Train a classifier first or remove --skip_classifier.")
        logger.info("\nUsing existing classifier from dataset_loader")

    # Determine learning rate and network architecture based on dataset size
    n_train_samples = len(dataset_loader.y_train)

    if args.learning_rate is None:
        if n_train_samples > 10000:
            learning_rate = 3e-4
        else:
            learning_rate = 5e-4
    else:
        learning_rate = args.learning_rate

    if n_train_samples > 10000:
        policy_net_arch = [512, 512, 256]
    elif n_train_samples > 5000:
        policy_net_arch = [256, 256, 256]
    else:
        policy_net_arch = [256, 256]

    logger.info(f"  Training samples: {n_train_samples}, LR: {learning_rate}, Net arch: {policy_net_arch}")

    experiment_config = {
        "total_timesteps": args.total_timesteps,
        "eval_freq": 6000,
        "n_eval_episodes": 20,
        "checkpoint_freq": 64_000,
        "log_interval": 10,
        "tensorboard_log": True,
    }

    algorithm_config = {
        "learning_rate": learning_rate,
        "buffer_size": 1_000_000,
        "learning_starts": 1000,
        "batch_size": 1024,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "action_noise_sigma": 0.1,
        "policy_kwargs": {
            "net_arch": policy_net_arch
        },
    }

    if args.algorithm == "sac":
        algorithm_config.update({
            "ent_coef": "auto",
            "target_update_interval": 1,
            "target_entropy": "auto",
        })

    trainer = AnchorTrainerSB3(
        dataset_loader=dataset_loader,
        algorithm=args.algorithm,
        experiment_config=experiment_config,
        algorithm_config=algorithm_config,
        output_dir=f"{args.output_dir}training/",
        seed=args.seed
    )

    if args.load_checkpoint:
        logger.info(f"\n{'='*80}")
        logger.info("RELOADING EXPERIMENT (Skipping training)")
        logger.info(f"{'='*80}")

        experiment_dir = args.load_checkpoint if os.path.isdir(args.load_checkpoint) else os.path.dirname(args.load_checkpoint)
        classifier_path = os.path.join(experiment_dir, "classifier.pth")
        if os.path.exists(classifier_path):
            classifier = dataset_loader.load_classifier(
                filepath=classifier_path,
                classifier_type=args.classifier_type,
                device=args.device
            )
            dataset_loader.classifier = classifier
            logger.info("Classifier loaded from checkpoint")
        elif dataset_loader.classifier is None:
            raise ValueError("Classifier not found in checkpoint or dataset_loader.")

        trainer.reload_experiment(
            experiment_dir=experiment_dir,
            env_config=None,
            target_classes=args.target_classes,
            max_cycles=args.max_cycles,
            device=args.device,
            eval_on_test_data=args.eval_on_test_data
        )
        checkpoint_path = experiment_dir
    else:
        trainer.setup_experiment(
            env_config=None,
            target_classes=args.target_classes,
            max_cycles=args.max_cycles,
            device=args.device,
            eval_on_test_data=args.eval_on_test_data
        )

        trainer.train()

        checkpoint_path = trainer.get_checkpoint_path()
        logger.info(f"\nSB3 checkpoint: {checkpoint_path}")

        # Copy classifier to checkpoint directory
        if hasattr(dataset_loader, 'classifier') and dataset_loader.classifier is not None:
            classifier_source = os.path.join(f"{args.output_dir}training/", "classifier.pth")
            classifier_dest = os.path.join(checkpoint_path, "classifier.pth")
            if os.path.exists(classifier_source):
                shutil.copy2(classifier_source, classifier_dest)

    # Evaluate
    eval_results = trainer.evaluate(n_episodes=10)
    logger.info(f"\nEvaluation complete")
    if "overall" in eval_results:
        overall = eval_results["overall"]
        logger.info(f"  Overall mean reward: {overall['mean_reward']:.4f} +/- {overall['std_reward']:.4f}")
        logger.info(f"  Evaluated {overall['n_classes']} classes")
    else:
        logger.info(f"  Mean reward: {eval_results.get('mean_reward', 0):.4f}")

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE!" if not args.load_checkpoint else "EVALUATION COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Experiment folder: {checkpoint_path}")


if __name__ == "__main__":
    main()
