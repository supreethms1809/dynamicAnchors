"""
Simple unit tests for single-agent pipeline.

Tests training, inference, test_extracted_rules, and plotting with minimal parameters.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Disable wandb for tests
os.environ["WANDB_MODE"] = "disabled"
os.environ["DISABLE_WANDB"] = "1"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "single_agent"))

from BenchMARL.tabular_datasets import TabularDatasetLoader
from single_agent.anchor_trainer_sb3 import AnchorTrainerSB3
from single_agent.test_extracted_rules_single import test_rules_from_json
from single_agent.summarize_and_plot_rules_single import summarize_rules_from_json, generate_summary_report
from single_agent.single_agent_inference import extract_rules_single_agent


def test_training():
    """Test single-agent training with minimal parameters."""
    print("\n" + "="*80)
    print("TEST: Single-Agent Training")
    print("="*80)
    
    # Use small dataset for quick testing
    dataset_name = "iris"
    seed = 42
    algorithm = "ddpg"
    
    # Create temporary output directory (persistent until manually cleaned)
    temp_dir = tempfile.mkdtemp(prefix="test_single_agent_")
    output_dir = os.path.join(temp_dir, "single_agent_test")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load dataset
        loader = TabularDatasetLoader(dataset_name=dataset_name, random_state=seed)
        loader.load_dataset()
        loader.preprocess_data()
        
        # Create and train classifier (required before training)
        classifier = loader.create_classifier(classifier_type="dnn", device="cpu")
        loader.train_classifier(
            classifier=classifier,
            device="cpu",
            epochs=10,  # Minimal epochs for testing
            patience=5
        )
        
        # Create trainer
        trainer = AnchorTrainerSB3(
            dataset_loader=loader,
            algorithm=algorithm,
            experiment_config={
                "total_timesteps": 200,  # Minimal timesteps for testing
                "tensorboard_log": False,  # Disable tensorboard for tests
            },
            algorithm_config={
                "learning_rate": 1e-3,
                "buffer_size": 10000,
                "learning_starts": 100,
                "batch_size": 32,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": (1, "step"),
                "gradient_steps": 1,
                "action_noise_sigma": 0.1,  # Required for DDPG
                "policy_kwargs": {
                    "net_arch": [64, 64]  # Smaller network for faster tests
                },
            },
            output_dir=output_dir  # Pass output_dir to trainer constructor
        )
        
        # Setup experiment (creates experiment_folder automatically)
        # Use all classes from the dataset (None = all classes)
        trainer.setup_experiment(
            target_classes=None,  # Use all classes from dataset
            device="cpu"
        )
        
        # Train
        trainer.train()
        
        # Save classifier to experiment folder (required for inference)
        if loader.classifier is not None:
            classifier_path = os.path.join(trainer.experiment_folder, "classifier.pth")
            loader.save_classifier(loader.classifier, classifier_path)
            print(f"Classifier saved to: {classifier_path}")
        
        # Verify outputs
        assert trainer.experiment_folder is not None, "Experiment folder should be created"
        assert os.path.exists(trainer.experiment_folder), "Experiment folder should exist"
        assert len(trainer.models) > 0, "Models should be created"
        
        print("✓ Training test passed!")
        return trainer.experiment_folder  # Return the actual experiment folder created by setup_experiment
    except Exception:
        # Clean up on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def test_inference(experiment_dir):
    """Test single-agent inference."""
    print("\n" + "="*80)
    print("TEST: Single-Agent Inference")
    print("="*80)
    
    dataset_name = "iris"
    seed = 42
    
    # Create output directory
    inference_dir = os.path.join(experiment_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    
    # Run inference
    results = extract_rules_single_agent(
        experiment_dir=experiment_dir,
        dataset_name=dataset_name,
        n_instances_per_class=2,  # Minimal instances for testing
        steps_per_episode=10,  # Minimal steps for testing
        seed=seed,
        device="cpu",
        output_dir=inference_dir
    )
    
    # Verify output
    assert results is not None, "Results should not be None"
    assert "per_class_results" in results, "Results should have per_class_results"
    assert len(results["per_class_results"]) > 0, "Should have at least one class"
    
    # Check rules file exists
    rules_file = os.path.join(inference_dir, "extracted_rules_single_agent.json")
    assert os.path.exists(rules_file), f"Rules file should exist: {rules_file}"
    
    print("✓ Inference test passed!")
    return rules_file


def test_test_extracted_rules(rules_file):
    """Test test_extracted_rules."""
    print("\n" + "="*80)
    print("TEST: Single-Agent Test Extracted Rules")
    print("="*80)
    
    dataset_name = "iris"
    seed = 42
    
    # Run tests
    test_results = test_rules_from_json(
        rules_file=rules_file,
        dataset_name=dataset_name,
        use_test_data=True,
        seed=seed
    )
    
    # Verify output structure
    assert test_results is not None, "Test results should not be None"
    assert "dataset" in test_results, "Should have dataset field"
    assert "per_class_results" in test_results, "Should have per_class_results"
    
    print("✓ Test extracted rules passed!")
    return test_results


def test_plotting(rules_file):
    """Test plotting and summarization."""
    print("\n" + "="*80)
    print("TEST: Single-Agent Plotting")
    print("="*80)
    
    dataset_name = "iris"
    
    # Load rules
    import json
    with open(rules_file, 'r') as f:
        rules_data = json.load(f)
    
    # Summarize
    summary = summarize_rules_from_json(rules_data)
    
    # Verify summary structure
    assert "n_classes" in summary, "Summary should have n_classes"
    assert "per_class_summary" in summary, "Summary should have per_class_summary"
    assert "overall_stats" in summary, "Summary should have overall_stats"
    
    # Generate report (text only, no plots)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        report_file = f.name
    
    try:
        generate_summary_report(summary, None, report_file)
        assert os.path.exists(report_file), "Report file should exist"
        print("✓ Plotting test passed!")
    finally:
        if os.path.exists(report_file):
            os.remove(report_file)


def run_all_tests():
    """Run all single-agent tests."""
    print("\n" + "="*80)
    print("RUNNING SINGLE-AGENT TESTS")
    print("="*80)
    
    experiment_dir = None
    temp_dirs_to_cleanup = []
    
    try:
        # Test 1: Training
        experiment_dir = test_training()
        if experiment_dir:
            temp_dirs_to_cleanup.append(os.path.dirname(experiment_dir))
        
        # Test 2: Inference
        rules_file = test_inference(experiment_dir) if experiment_dir else None
        
        # Test 3: Test extracted rules
        test_results = test_test_extracted_rules(rules_file) if rules_file else None
        
        # Test 4: Plotting
        test_plotting(rules_file) if rules_file else None
        
        print("\n" + "="*80)
        print("ALL SINGLE-AGENT TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup temporary directories
        for temp_dir in temp_dirs_to_cleanup:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    run_all_tests()
