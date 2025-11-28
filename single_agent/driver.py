"""
Single-Agent Dynamic Anchors Training Pipeline using Stable-Baselines3

This script is used to train the single-agent dynamic anchors model using
Stable-Baselines3 (DDPG or SAC algorithms).

Usage:
python driver.py --dataset <dataset_name> --algorithm <algorithm_name> --seed <seed>

Example:
python driver.py --dataset breast_cancer --algorithm ddpg --seed 42

Training pipeline supports the following datasets and algorithms:
Dataset: breast_cancer, wine, iris, synthetic, moons, circles, covtype, housing
Algorithm: ddpg, sac
Classifier: dnn, random_forest, gradient_boosting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add single_agent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BenchMARL.tabular_datasets import TabularDatasetLoader
from anchor_trainer_sb3 import AnchorTrainerSB3
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    available_algorithms = list(AnchorTrainerSB3.ALGORITHM_MAP.keys())
    
    if not available_algorithms:
        raise RuntimeError(
            "No algorithms available. Make sure Stable-Baselines3 is properly installed."
        )
    
    parser = argparse.ArgumentParser(description="Single-Agent Anchor Training Pipeline (SB3)")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "wine", "iris", "synthetic", "moons", "circles", "covtype", "housing"],
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ddpg" if "ddpg" in available_algorithms else available_algorithms[0],
        choices=available_algorithms,
        help=f"RL algorithm to use. Available: {', '.join(available_algorithms)}"
    )
    
    parser.add_argument(
        "--classifier_type",
        type=str,
        default="dnn",
        choices=["dnn", "random_forest", "gradient_boosting"],
        help="Type of classifier to train"
    )
    
    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=100,
        help="Number of epochs to train classifier"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./output/single_agent_sb3_{dataset}_{algorithm}/)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--skip_eda",
        action="store_true",
        help="Skip EDA analysis"
    )
    
    parser.add_argument(
        "--skip_classifier",
        action="store_true",
        help="Skip classifier training (use existing classifier)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to use"
    )
    
    parser.add_argument(
        "--eval_on_test_data",
        action="store_true",
        help="Evaluate on test data instead of training data (for final evaluation)"
    )
    
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=1000,
        help="Maximum cycles per episode"
    )
    
    parser.add_argument(
        "--target_classes",
        type=int,
        nargs="+",
        default=None,
        help="Target classes to generate anchors for (default: all classes)"
    )
    
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to load (skips training, only runs evaluation). "
             "Can be experiment folder or specific checkpoint file."
    )
    
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=72_000,
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the algorithm"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"./output/single_agent_sb3_{args.dataset}_{args.algorithm}/"
    
    print("="*80)
    if args.load_checkpoint:
        print("ANCHOR EVALUATION MODE (Loading from checkpoint)")
    else:
        print("SINGLE-AGENT ANCHOR TRAINING PIPELINE (Stable-Baselines3)")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Output directory: {args.output_dir}")
    if args.load_checkpoint:
        print(f"Checkpoint: {args.load_checkpoint}")
    print("="*80)
    
    # Load dataset
    dataset_loader = TabularDatasetLoader(
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
            dropout_rate=0.3,
            use_batch_norm=True,
            device=args.device
        )
        
        trained_classifier, test_acc, history = dataset_loader.train_classifier(
            classifier,
            epochs=args.classifier_epochs,
            batch_size=256,
            lr=1e-3,
            patience=10,
            device=args.device
        )
        
        logger.info(f"\nClassifier trained with test accuracy: {test_acc:.4f}")
        
        # Save classifier
        classifier_output_dir = f"{args.output_dir}training/"
        os.makedirs(classifier_output_dir, exist_ok=True)
        classifier_path = os.path.join(classifier_output_dir, "classifier.pth")
        dataset_loader.save_classifier(trained_classifier, classifier_path)
        logger.info(f"Classifier saved to: {classifier_path}")
    else:
        if dataset_loader.classifier is None:
            raise ValueError(
                "Classifier not found. Either train a classifier first "
                "or remove --skip_classifier flag."
            )
        logger.info("\nUsing existing classifier from dataset_loader")
    
    # Create trainer
    experiment_config = {
        "total_timesteps": args.total_timesteps,
        "eval_freq": 48_000,
        "n_eval_episodes": 4,
        "checkpoint_freq": 48_000,
        "log_interval": 10,
        "tensorboard_log": True,
    }
    
    algorithm_config = {
        "learning_rate": args.learning_rate,
        "buffer_size": 1_000_000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "action_noise_sigma": 0.1,
        "policy_kwargs": {
            "net_arch": [256, 256]
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
        # Load existing experiment instead of training
        logger.info(f"\n{'='*80}")
        logger.info("RELOADING EXPERIMENT (Skipping training)")
        logger.info(f"{'='*80}")
        logger.info(f"Loading experiment from: {args.load_checkpoint}")
        
        # Load classifier from checkpoint directory if available
        experiment_dir = args.load_checkpoint if os.path.isdir(args.load_checkpoint) else os.path.dirname(args.load_checkpoint)
        classifier_path = os.path.join(experiment_dir, "classifier.pth")
        if os.path.exists(classifier_path):
            logger.info(f"Loading classifier from: {classifier_path}")
            classifier = dataset_loader.load_classifier(
                filepath=classifier_path,
                classifier_type=args.classifier_type,
                device=args.device
            )
            dataset_loader.classifier = classifier
            logger.info("Classifier loaded successfully")
        else:
            logger.warning(f"Warning: Classifier not found at {classifier_path}")
            if dataset_loader.classifier is None:
                raise ValueError(
                    "Classifier not found in experiment directory and not available in dataset_loader. "
                    "Please train a classifier first or ensure it's saved in the experiment directory."
                )
            logger.info("Using existing classifier from dataset_loader")
        
        # Reload the entire experiment (set up environments and load models)
        trainer.reload_experiment(
            experiment_dir=experiment_dir,
            env_config=None,
            target_classes=args.target_classes,
            max_cycles=args.max_cycles,
            device=args.device,
            eval_on_test_data=args.eval_on_test_data
        )
        
        # Use the experiment directory as the checkpoint path
        checkpoint_path = experiment_dir
    else:
        # Normal training flow - setup experiment first
        trainer.setup_experiment(
            env_config=None,
            target_classes=args.target_classes,
            max_cycles=args.max_cycles,
            device=args.device,
            eval_on_test_data=args.eval_on_test_data
        )
        
        trainer.train()
        
        # Get checkpoint path
        checkpoint_path = trainer.get_checkpoint_path()
        logger.info(f"\nSB3 checkpoint location: {checkpoint_path}")
        logger.info(f"  Use this path to load checkpoints later for evaluation or continued training")
        
        # Copy classifier to checkpoint directory for easy access during inference
        if hasattr(dataset_loader, 'classifier') and dataset_loader.classifier is not None:
            import shutil
            classifier_source = os.path.join(f"{args.output_dir}training/", "classifier.pth")
            classifier_dest = os.path.join(checkpoint_path, "classifier.pth")
            if os.path.exists(classifier_source):
                shutil.copy2(classifier_source, classifier_dest)
                logger.info(f"  Classifier copied to checkpoint directory: {classifier_dest}")
    
    # Run evaluation
    eval_results = trainer.evaluate(n_episodes=10)
    logger.info(f"\nEvaluation complete")
    if "overall" in eval_results:
        overall = eval_results["overall"]
        logger.info(f"  Overall mean reward: {overall['mean_reward']:.4f} +/- {overall['std_reward']:.4f}")
        logger.info(f"  Evaluated {overall['n_classes']} classes")
    else:
        # Fallback for old format
        logger.info(f"  Mean reward: {eval_results.get('mean_reward', 0):.4f} +/- {eval_results.get('std_reward', 0):.4f}")
    
    logger.info(f"\n{'='*80}")
    if args.load_checkpoint:
        logger.info("EVALUATION COMPLETE!")
    else:
        logger.info("TRAINING COMPLETE!")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nExperiment folder: {checkpoint_path}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

