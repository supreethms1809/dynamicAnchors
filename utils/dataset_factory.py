"""Dataset loader factory.

WyoDOT lives in `wyodot.WyoDOTDatasetLoader` (CSV + paper RF), not in
`TabularDatasetLoader.DATASETS`. Revision baselines / evaluate used to only
construct the BenchMARL loader, so `--dataset wyodot_kvdw_labeled` failed.
"""
from __future__ import annotations


def make_tabular_loader(dataset_name: str, random_state: int = 42, **kwargs):
    """Return the loader class instance for `dataset_name` (not yet loaded)."""
    if str(dataset_name).startswith("wyodot"):
        from wyodot.wyodot_dataset_loader import WyoDOTDatasetLoader
        return WyoDOTDatasetLoader(
            dataset_name=dataset_name, random_state=random_state, **kwargs
        )
    from BenchMARL.tabular_datasets import TabularDatasetLoader
    return TabularDatasetLoader(
        dataset_name=dataset_name, random_state=random_state, **kwargs
    )
