from tabular_datasets import TabularDatasetLoader
from anchor_trainer import AnchorTrainer
import argparse
import os


def main():
    available_algorithms = list(AnchorTrainer.ALGORITHM_MAP.keys())
    
    if not available_algorithms:
        raise RuntimeError(
            "No algorithms available. Make sure BenchMARL is properly installed."
        )
    
    parser = argparse.ArgumentParser(description="Anchor Training Pipeline")
    
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
        default="maddpg" if "maddpg" in available_algorithms else available_algorithms[0],
        choices=available_algorithms,
        help=f"MARL algorithm to use. Available: {', '.join(available_algorithms)}"
    )
    
    parser.add_argument(
        "--algorithm_config",
        type=str,
        default=None,
        help="Path to algorithm config YAML (default: conf/{algorithm}.yaml)"
    )
    
    parser.add_argument(
        "--experiment_config",
        type=str,
        default="conf/base_experiment.yaml",
        help="Path to experiment config YAML"
    )
    
    parser.add_argument(
        "--mlp_config",
        type=str,
        default="conf/mlp.yaml",
        help="Path to MLP model config YAML"
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
        help="Output directory (default: ./output/{dataset}_{algorithm}/)"
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
        choices=["cpu", "cuda", "mps"],
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
        default=100,
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
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"./output/{args.dataset}_{args.algorithm}/"
    
    print("="*80)
    if args.load_checkpoint:
        print("ANCHOR EVALUATION MODE (Loading from checkpoint)")
    else:
        print("ANCHOR TRAINING PIPELINE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Output directory: {args.output_dir}")
    if args.load_checkpoint:
        print(f"Checkpoint: {args.load_checkpoint}")
    print("="*80)
    
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
    
    # Skip classifier training if loading from checkpoint (will load from checkpoint instead)
    if args.load_checkpoint:
        # When loading from checkpoint, we'll load the classifier from the checkpoint directory
        # Skip training here - it will be loaded later
        print("\nSkipping classifier training (will load from checkpoint)")
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
        
        print(f"\nClassifier trained with test accuracy: {test_acc:.4f}")
        
        # Save classifier for later use in inference
        # We'll save it in the output directory (will be moved to checkpoint folder after training)
        classifier_output_dir = f"{args.output_dir}training/"
        os.makedirs(classifier_output_dir, exist_ok=True)
        classifier_path = os.path.join(classifier_output_dir, "classifier.pth")
        dataset_loader.save_classifier(trained_classifier, classifier_path)
        print(f"Classifier saved to: {classifier_path}")
    else:
        if dataset_loader.classifier is None:
            raise ValueError(
                "Classifier not found. Either train a classifier first "
                "or remove --skip_classifier flag."
            )
        print("\nUsing existing classifier from dataset_loader")
    
    trainer = AnchorTrainer(
        dataset_loader=dataset_loader,
        algorithm=args.algorithm,
        algorithm_config_path=args.algorithm_config,
        experiment_config_path=args.experiment_config,
        mlp_config_path=args.mlp_config,
        output_dir=f"{args.output_dir}training/",
        seed=args.seed
    )
    
    if args.load_checkpoint:
        # Load existing experiment instead of training
        print(f"\n{'='*80}")
        print("RELOADING EXPERIMENT (Skipping training)")
        print(f"{'='*80}")
        print(f"Loading experiment from: {args.load_checkpoint}")
        
        # Load classifier from checkpoint directory if available
        experiment_dir = args.load_checkpoint if os.path.isdir(args.load_checkpoint) else os.path.dirname(args.load_checkpoint)
        classifier_path = os.path.join(experiment_dir, "classifier.pth")
        if os.path.exists(classifier_path):
            print(f"Loading classifier from: {classifier_path}")
            classifier = dataset_loader.load_classifier(
                filepath=classifier_path,
                classifier_type=args.classifier_type,
                device=args.device
            )
            dataset_loader.classifier = classifier
            print("Classifier loaded successfully")
        else:
            print(f"Warning: Classifier not found at {classifier_path}")
            if dataset_loader.classifier is None:
                raise ValueError(
                    "Classifier not found in experiment directory and not available in dataset_loader. "
                    "Please train a classifier first or ensure it's saved in the experiment directory."
                )
            print("Using existing classifier from dataset_loader")
        
        # Reload the entire experiment using BenchMARL's official method
        # Can pass either directory or checkpoint file - reload_experiment will find the checkpoint
        trainer.reload_experiment(experiment_dir)
        
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
        
        # Get BenchMARL checkpoint path (BenchMARL saves checkpoints automatically)
        checkpoint_path = trainer.get_checkpoint_path()
        print(f"\nBenchMARL checkpoint location: {checkpoint_path}")
        print(f"  Use this path to load checkpoints later for evaluation or continued training")
        
        # Copy classifier to checkpoint directory for easy access during inference
        if hasattr(dataset_loader, 'classifier') and dataset_loader.classifier is not None:
            import shutil
            classifier_source = os.path.join(f"{args.output_dir}training/", "classifier.pth")
            classifier_dest = os.path.join(checkpoint_path, "classifier.pth")
            if os.path.exists(classifier_source):
                shutil.copy2(classifier_source, classifier_dest)
                print(f"  Classifier copied to checkpoint directory: {classifier_dest}")
    
    # Run BenchMARL evaluation (for metrics only)
    eval_results = trainer.evaluate()
    print(f"\nEvaluation complete")
    print(f"  Total frames: {eval_results['total_frames']}")
    
    # Save evaluation anchor data to JSON file
    evaluation_anchor_data = eval_results.get("evaluation_anchor_data", [])
    if evaluation_anchor_data:
        import json
        import numpy as np
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable types to JSON-compatible types."""
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
            else:
                return obj
        
        # Convert data to JSON-serializable format
        serializable_data = convert_to_serializable(evaluation_anchor_data)
        
        # Save to experiment directory
        experiment_dir = str(trainer.experiment.folder_name)
        json_path = os.path.join(experiment_dir, "evaluation_anchor_data.json")
        
        with open(json_path, 'w') as f:
            json.dump({
                "n_episodes": len(evaluation_anchor_data),
                "episodes": serializable_data,
                "metadata": {
                    "total_frames": eval_results.get("total_frames", 0),
                    "n_iters_performed": eval_results.get("n_iters_performed", 0),
                    "experiment_folder": eval_results.get("experiment_folder", ""),
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Evaluation anchor data saved to: {json_path}")
        print(f"  Total episodes: {len(evaluation_anchor_data)}")
        
        # Print summary of collected data
        if len(evaluation_anchor_data) > 0:
            first_episode = evaluation_anchor_data[0]
            print(f"\n  Data structure:")
            print(f"    Episodes: {len(evaluation_anchor_data)}")
            if isinstance(first_episode, dict):
                print(f"    Groups per episode: {list(first_episode.keys())}")
                for group, group_data in first_episode.items():
                    if isinstance(group_data, dict):
                        print(f"      Group '{group}' contains: {list(group_data.keys())}")
                        if "precision" in group_data:
                            print(f"        Precision: {group_data.get('precision', 'N/A')}")
                        if "coverage" in group_data:
                            print(f"        Coverage: {group_data.get('coverage', 'N/A')}")
                        if "final_observation" in group_data:
                            obs_len = len(group_data.get("final_observation", []))
                            print(f"        Final observation length: {obs_len}")
    else:
        print(f"\n⚠ No evaluation anchor data collected")
    
    # Extract and save individual models for easier standalone inference
    print(f"\n{'='*80}")
    print("EXTRACTING INDIVIDUAL MODELS FOR STANDALONE INFERENCE")
    print("="*80)
    try:
        saved_models = trainer.extract_and_save_individual_models(
            save_policies=True,
            save_critics=False  # Set to True if you need critic models
        )
        # Models are saved in the experiment's run log directory
        models_dir = os.path.join(str(trainer.experiment.folder_name), "individual_models")
        print(f"\n✓ Individual models extracted successfully!")
        print(f"  Models saved to: {models_dir}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not extract individual models: {e}")
        print("  You can still use the full BenchMARL checkpoint for inference")
    
    # Note: Rule extraction should be done separately using inference.py
    print(f"\n{'='*80}")
    if args.load_checkpoint:
        print("EVALUATION COMPLETE!")
    else:
        print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    if not args.load_checkpoint:
        print(f"\nTo extract rules, run:")
        print(f"  python inference.py --experiment_dir {checkpoint_path} --dataset {args.dataset}")
        print(f"\nOr to re-run evaluation only, use:")
        print(f"  python driver.py --load_checkpoint {checkpoint_path} --dataset {args.dataset} --algorithm {args.algorithm}")
        print(f"\nThis will:")
        print(f"  1. Load the trained model from checkpoint")
        print(f"  2. Run evaluation")
        print(f"  3. Extract and save individual models")
    else:
        print(f"\nTo extract rules, run:")
        print(f"  python inference.py --experiment_dir {checkpoint_path} --dataset {args.dataset}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

