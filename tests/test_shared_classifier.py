import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "single_agent"))

from run_parallel_classes import peel_classifier_path  # noqa: E402
from revision.fit_shared_classifier import classifier_patience  # noqa: E402


def test_peel_classifier_path_strips_flag():
    path, rest = peel_classifier_path(
        ["--", "--skip_eda", "--classifier_path", "/tmp/c.pth", "--foo"]
    )
    assert path == "/tmp/c.pth"
    assert rest == ["--skip_eda", "--foo"]


def test_peel_classifier_path_hyphen_alias():
    path, rest = peel_classifier_path(["--classifier-path", "/x.pth"])
    assert path == "/x.pth"
    assert rest == []


def test_classifier_patience_matches_driver_large_and_small():
    assert classifier_patience("iris") == 50
    assert classifier_patience("wine") == 100
    assert classifier_patience("housing") == 200
    assert classifier_patience("uci_adult") == 200
    assert classifier_patience("uci_credit") == 200
    assert classifier_patience("folktables_income_CA_2018") == 200
    assert classifier_patience("covtype") == 50
