"""
Baseline explainability methods for comparison with dynamic anchors.
"""

import sys

# Avoid importing establish_baseline in __init__.py when it's being run as a module
# to prevent RuntimeWarning: module found in sys.modules before execution
# When running "python -m baseline.establish_baseline", Python imports the package
# before executing the module. We detect this by checking sys.argv.
_skip_import = False

# Check all possible positions of '-m' flag in sys.argv
try:
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-m' and i + 1 < len(sys.argv):
            module_name = sys.argv[i + 1]
            if module_name == 'baseline.establish_baseline':
                _skip_import = True
                break
except (IndexError, AttributeError, TypeError):
    pass

# Only import if not running as module
if not _skip_import:
    from .establish_baseline import (
        load_dataset,
        train_classifier,
        run_lime,
        run_static_anchors,
        run_shap,
        run_feature_importance,
        main,
    )

__all__ = [
    "load_dataset",
    "train_classifier",
    "run_lime",
    "run_static_anchors",
    "run_shap",
    "run_feature_importance",
    "main",
]

