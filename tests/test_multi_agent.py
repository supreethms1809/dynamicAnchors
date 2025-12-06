"""
Simple unit tests for multi-agent pipeline.

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
sys.path.insert(0, str(project_root / "BenchMARL"))

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

from BenchMARL.tabular_datasets import TabularDatasetLoader
from BenchMARL.anchor_trainer import AnchorTrainer
from BenchMARL.inference import extract_rules_from_policies
from BenchMARL.test_extracted_rules import test_rules_from_json
from BenchMARL.summarize_and_plot_rules import summarize_rules_from_json, generate_summary_report


def test_training():
    """Test multi-agent training with minimal parameters."""
    print("\n" + "="*80)
    print("TEST: Multi-Agent Training")
    print("="*80)
    
    # Use small dataset for quick testing
    dataset_name = "iris"
    seed = 42
    algorithm = "maddpg"
    
    # Create temporary output directory (persistent until manually cleaned)
    temp_dir = tempfile.mkdtemp(prefix="test_multi_agent_")
    output_dir = os.path.join(temp_dir, "multi_agent_test")
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
        
        # Create trainer with test config files
        test_conf_dir = Path(__file__).parent / "conf"
        
        trainer = AnchorTrainer(
            dataset_loader=loader,
            algorithm=algorithm,
            algorithm_config_path=str(test_conf_dir / "maddpg.yaml"),
            experiment_config_path=str(test_conf_dir / "base_experiment.yaml"),
            mlp_config_path=str(test_conf_dir / "mlp.yaml"),
            anchor_config_path=str(test_conf_dir / "anchor.yaml"),
            output_dir=output_dir,
            seed=seed
        )
        
        # Setup experiment
        trainer.setup_experiment(
            target_classes=[0, 1],  # Use first two classes for faster training
            max_cycles=50,  # Minimal cycles for testing
            device="cpu",
            eval_on_test_data=False  # Skip evaluation for faster tests
        )
        
        # Train
        trainer.train()
        
        # Extract and save individual models (required for inference)
        trainer.extract_and_save_individual_models(
            save_policies=True,
            save_critics=False
        )
        
        # Save classifier to checkpoint path (required for inference)
        if loader.classifier is not None:
            checkpoint_path = trainer.get_checkpoint_path()
            classifier_path = os.path.join(checkpoint_path, "classifier.pth")
            loader.save_classifier(loader.classifier, classifier_path)
            print(f"Classifier saved to: {classifier_path}")
        
        # Verify outputs
        assert trainer.experiment is not None, "Experiment should be created"
        checkpoint_path = trainer.get_checkpoint_path()
        assert checkpoint_path is not None, "Checkpoint path should exist"
        
        # Verify individual models directory exists
        models_dir = os.path.join(checkpoint_path, "individual_models")
        assert os.path.exists(models_dir), f"Individual models directory should exist: {models_dir}"
        
        print("✓ Training test passed!")
        return checkpoint_path
    except Exception:
        # Clean up on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def test_inference(experiment_dir):
    """Test multi-agent inference."""
    print("\n" + "="*80)
    print("TEST: Multi-Agent Inference")
    print("="*80)
    
    dataset_name = "iris"
    seed = 42
    
    # Create output directory
    inference_dir = os.path.join(experiment_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    
    # Get test config directory
    test_conf_dir = Path(__file__).parent / "conf"
    
    # Run inference
    results = extract_rules_from_policies(
        experiment_dir=experiment_dir,
        dataset_name=dataset_name,
        mlp_config_path=str(test_conf_dir / "mlp.yaml"),
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
    rules_file = os.path.join(inference_dir, "extracted_rules.json")
    assert os.path.exists(rules_file), f"Rules file should exist: {rules_file}"
    
    print("✓ Inference test passed!")
    return rules_file


def test_test_extracted_rules(rules_file):
    """Test test_extracted_rules."""
    print("\n" + "="*80)
    print("TEST: Multi-Agent Test Extracted Rules")
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
    print("TEST: Multi-Agent Plotting")
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
    """Run all multi-agent tests."""
    print("\n" + "="*80)
    print("RUNNING MULTI-AGENT TESTS")
    print("="*80)
    
    experiment_dir = None
    temp_dirs_to_cleanup = []
    
    try:
        # Test 1: Training
        experiment_dir = test_training()
        if experiment_dir:
            # Try to find the temp directory from experiment_dir path
            parts = Path(experiment_dir).parts
            for i, part in enumerate(parts):
                if part.startswith("test_multi_agent_"):
                    temp_dirs_to_cleanup.append(str(Path(*parts[:i+1])))
                    break
        
        # Test 2: Inference
        rules_file = test_inference(experiment_dir) if experiment_dir else None
        
        # Test 3: Test extracted rules
        test_results = test_test_extracted_rules(rules_file) if rules_file else None
        
        # Test 4: Plotting
        test_plotting(rules_file) if rules_file else None
        
        print("\n" + "="*80)
        print("ALL MULTI-AGENT TESTS PASSED!")
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
