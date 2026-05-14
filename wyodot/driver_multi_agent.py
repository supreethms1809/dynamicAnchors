"""
WyoDOT Multi-Agent Dynamic Anchors Training Pipeline

Usage:
python driver_multi_agent.py --dataset wyodot_kvdw_labeled --algorithm maddpg --seed 42

Datasets: wyodot_kvdw_labeled, wyodot_testbed
Algorithms: maddpg, masac
Classifier: random_forest (default, with paper params), dnn, gradient_boosting
"""

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import sys
# Add project root and BenchMARL to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "BenchMARL"))

from wyodot_dataset_loader import WyoDOTDatasetLoader
from anchor_trainer import AnchorTrainer
import argparse
import random
import numpy as np
import torch
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    available_algorithms = list(AnchorTrainer.ALGORITHM_MAP.keys())
    if not available_algorithms:
        raise RuntimeError("No algorithms available. Make sure BenchMARL is properly installed.")

    parser = argparse.ArgumentParser(description="WyoDOT Multi-Agent Anchor Training Pipeline")

    parser.add_argument(
        "--dataset", type=str, default="wyodot_kvdw_labeled",
        choices=list(WyoDOTDatasetLoader.DATASETS.keys()),
        help="WyoDOT dataset to use"
    )
    parser.add_argument(
        "--algorithm", type=str,
        default="maddpg" if "maddpg" in available_algorithms else available_algorithms[0],
        choices=available_algorithms,
        help=f"MARL algorithm. Available: {', '.join(available_algorithms)}"
    )
    parser.add_argument(
        "--algorithm_config", type=str, default=None,
        help="Path to algorithm config YAML (default: BenchMARL/conf/{algorithm}.yaml)"
    )
    parser.add_argument(
        "--experiment_config", type=str,
        default=os.path.join(PROJECT_ROOT, "BenchMARL", "conf", "base_experiment.yaml"),
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--mlp_config", type=str,
        default=os.path.join(PROJECT_ROOT, "BenchMARL", "conf", "mlp.yaml"),
        help="Path to MLP model config YAML"
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
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--eval_on_test_data", nargs='?', const=True, default=None,
        help="Evaluate on test data instead of training data"
    )
    parser.add_argument("--max_cycles", type=int, default=None, help="Max cycles per episode")
    parser.add_argument("--max_n_frames", type=int, default=None, help="Total training frames")
    parser.add_argument(
        "--target_classes", type=int, nargs="+", default=None,
        help="Target classes to generate anchors for (default: all)"
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None,
        help="Path to checkpoint to load (skips training)"
    )

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir is None:
        args.output_dir = f"./output/{args.dataset}_{args.algorithm}/"

    print("=" * 80)
    if args.load_checkpoint:
        print("WYODOT ANCHOR EVALUATION MODE (Loading from checkpoint)")
    else:
        print("WYODOT MULTI-AGENT ANCHOR TRAINING PIPELINE")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
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

    # Set up trainer
    # Resolve algorithm config path relative to BenchMARL/conf/
    algorithm_config = args.algorithm_config
    if algorithm_config is None:
        algorithm_config = os.path.join(PROJECT_ROOT, "BenchMARL", "conf", f"{args.algorithm}.yaml")

    trainer = AnchorTrainer(
        dataset_loader=dataset_loader,
        algorithm=args.algorithm,
        algorithm_config_path=algorithm_config,
        experiment_config_path=args.experiment_config,
        mlp_config_path=args.mlp_config,
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

        trainer.reload_experiment(experiment_dir)
        checkpoint_path = experiment_dir
    else:
        trainer.setup_experiment(
            env_config=None,
            target_classes=args.target_classes,
            max_cycles=args.max_cycles,
            max_n_frames=args.max_n_frames,
            device=args.device,
            eval_on_test_data=args.eval_on_test_data
        )

        trainer.train()

        checkpoint_path = trainer.get_checkpoint_path()
        logger.info(f"\nBenchMARL checkpoint: {checkpoint_path}")

        # Copy classifier to checkpoint directory
        if hasattr(dataset_loader, 'classifier') and dataset_loader.classifier is not None:
            classifier_source = os.path.join(f"{args.output_dir}training/", "classifier.pth")
            classifier_dest = os.path.join(checkpoint_path, "classifier.pth")
            if os.path.exists(classifier_source):
                shutil.copy2(classifier_source, classifier_dest)

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"\nEvaluation complete. Total frames: {eval_results['total_frames']}")

    # Save evaluation anchor data
    evaluation_anchor_data = eval_results.get("evaluation_anchor_data", [])
    if evaluation_anchor_data:
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        experiment_dir = str(trainer.experiment.folder_name)
        json_path = os.path.join(experiment_dir, "evaluation_anchor_data.json")
        with open(json_path, 'w') as f:
            json.dump({
                "n_episodes": len(evaluation_anchor_data),
                "episodes": convert_to_serializable(evaluation_anchor_data),
                "metadata": {
                    "total_frames": eval_results.get("total_frames", 0),
                    "n_iters_performed": eval_results.get("n_iters_performed", 0),
                    "experiment_folder": eval_results.get("experiment_folder", ""),
                }
            }, f, indent=2)
        logger.info(f"Evaluation anchor data saved to: {json_path}")

    # Extract individual models
    logger.info(f"\n{'='*80}")
    logger.info("EXTRACTING INDIVIDUAL MODELS")
    logger.info("=" * 80)

    best_model_path = os.path.join(checkpoint_path, "best_model", "best_checkpoint.pt")
    if os.path.exists(best_model_path):
        try:
            trainer.reload_experiment(best_model_path)
            logger.info("Reloaded from best model checkpoint")
        except Exception as e:
            logger.warning(f"Could not reload best model: {e}")

    try:
        trainer.extract_and_save_individual_models(save_policies=True, save_critics=False)
        models_dir = os.path.join(str(trainer.experiment.folder_name), "individual_models")
        logger.info(f"Individual models saved to: {models_dir}")
    except Exception as e:
        logger.warning(f"Could not extract individual models: {e}")

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING COMPLETE!" if not args.load_checkpoint else "EVALUATION COMPLETE!")
    logger.info(f"{'='*80}")

    if not args.load_checkpoint:
        logger.info(f"\nTo extract rules, run:")
        logger.info(f"  cd {os.path.join(PROJECT_ROOT, 'BenchMARL')}")
        logger.info(f"  python inference.py --experiment_dir {checkpoint_path} --dataset {args.dataset}")


if __name__ == "__main__":
    main()
