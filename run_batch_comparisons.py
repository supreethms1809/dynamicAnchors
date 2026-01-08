#!/usr/bin/env python3
"""
Run multiple comparisons using run_comparison_pipeline.py for a set of
datasets and algorithms, without any command line options.

Edit the DATASETS and ALGORITHMS lists below to control what gets run.
"""

import subprocess
import sys
from pathlib import Path


# Editable list of datasets to run.
# Make sure each dataset is supported by run_comparison_pipeline.py.
DATASETS = [
    "iris",
    "breast_cancer",
    "wine",
    "housing",
    "covtype",
    "circles",
    "uci_credit",
    "folktables_income_CA_2018",
    # Add/remove datasets here as needed
]

# Editable list of multi-agent algorithms to run.
# These are passed directly to run_comparison_pipeline.py.
ALGORITHMS = [
    "maddpg",
    # Add/remove algorithms here as needed
]


def main() -> None:
    project_root = Path(__file__).resolve().parent
    pipeline_script = project_root / "run_comparison_pipeline.py"

    if not pipeline_script.exists():
        raise FileNotFoundError(f"run_comparison_pipeline.py not found at {pipeline_script}")

    for dataset in DATASETS:
        for algo in ALGORITHMS:
            cmd = [
                sys.executable,
                str(pipeline_script),
                "--dataset",
                dataset,
                "--algorithm",
                algo,
                "--force_retrain",
            ]
            print("\n" + "=" * 80)
            print(f"Running comparison pipeline for dataset='{dataset}', algorithm='{algo}'")
            print("Command:", " ".join(cmd))
            print("=" * 80)
            subprocess.run(cmd, check=True, cwd=str(project_root))

if __name__ == "__main__":
    main()


