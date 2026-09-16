"""The marginal-call unit must separate table builds from genuine new calls.

Regression guard for the accounting that replaced the closed-form query counts:
re-reading a split must be free, and only rows that were never tabulated may be
charged as marginal.
"""
import numpy as np
import torch

from utils.networks import SimpleClassifier
from utils.query_meter import QueryMeter


def _model(d=4, k=3):
    m = SimpleClassifier(input_dim=d, num_classes=k)
    m.eval()
    return m


def test_registered_split_is_a_one_time_build_then_free():
    X = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    m = _model()
    with QueryMeter([X]) as meter, torch.no_grad():
        for _ in range(5):
            m(torch.from_numpy(X))
    assert meter.table_rows_unique == 50      # charged once
    assert meter.table_reads == 4             # the other four are free
    assert meter.marginal_rows == 0


def test_untabulated_rows_are_marginal():
    X = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    Z = np.random.default_rng(1).normal(size=(37, 4)).astype(np.float32)
    m = _model()
    with QueryMeter([X]) as meter, torch.no_grad():
        m(torch.from_numpy(X))
        m(torch.from_numpy(Z))
    assert meter.table_rows_unique == 50
    assert meter.marginal_rows == 37
    assert meter.marginal_calls == 1


def test_sub_batch_of_a_tabulated_split_is_a_read_not_a_query():
    """Rows inside a candidate box are already tabulated; scoring them is free.

    Matching whole matrices only was what inflated MADA from ~0 to 467
    "marginal" rows per explanation on housing.
    """
    X = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    m = _model()
    with QueryMeter([X]) as meter, torch.no_grad():
        m(torch.from_numpy(X[7:19]))       # a filtered subset of the split
    assert meter.marginal_rows == 0
    assert meter.table_read_rows == 12


def test_mixed_batch_charges_only_the_untabulated_rows():
    X = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    Z = np.random.default_rng(1).normal(size=(3, 4)).astype(np.float32)
    mixed = np.vstack([X[:5], Z]).astype(np.float32)
    m = _model()
    with QueryMeter([X]) as meter, torch.no_grad():
        m(torch.from_numpy(mixed))
    assert meter.marginal_rows == 3        # only the synthetic rows
    assert meter.table_read_rows == 5


def test_meter_restores_forward_on_exit():
    orig = SimpleClassifier.forward
    with QueryMeter([]):
        assert SimpleClassifier.forward is not orig
    assert SimpleClassifier.forward is orig
