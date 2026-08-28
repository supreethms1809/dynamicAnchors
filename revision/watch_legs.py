"""Watch eval JSON files and print a leg summary when they appear/update.

Used for the in-flight sweep (the running orchestrator was started before
print-on-eval was added). Safe to leave running; exits when all expected
legs have been printed or after --timeout-s.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from print_leg import summarize  # noqa: E402

RESULTS = REPO / "revision" / "results"
OUT = REPO / "revision" / "logs" / "leg_summaries.log"

LEGS = [
    ("iris", "rlda", 42),
    ("iris", "mada", 42),
    ("wine", "rlda", 42),
    ("wine", "mada", 42),
    ("breast_cancer", "rlda", 42),
    ("breast_cancer", "mada", 42),
    ("uci_credit", "rlda", 42),
    ("uci_credit", "mada", 42),
    ("folktables_income_CA_2018", "rlda", 42),
    ("folktables_income_CA_2018", "mada", 42),
]


def _json_path(dataset: str, method: str, seed: int) -> Path:
    return RESULTS / f"{dataset}__{method}__seed{seed}__tp0p90__tc0p20.json"


def _rules_path(dataset: str, method: str, seed: int) -> Path | None:
    if method == "rlda":
        root = REPO / "output" / f"{dataset}_rlda_ddpg_seed{seed}" / "training"
        glob = "**/extracted_rules_single_agent.json"
    else:
        root = REPO / "output" / f"{dataset}_mada_maddpg_seed{seed}" / "training"
        glob = "**/extracted_rules.json"
    if not root.is_dir():
        return None
    cands = list(root.glob(glob))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _emit(text: str) -> None:
    print(text, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as fh:
        fh.write(text + "\n\n")


def main() -> None:
    start = time.time()
    timeout = 36 * 3600
    seen_mtime: dict[tuple, float] = {}
    printed: set[tuple] = set()
    _emit(f"[watch_legs] waiting for {len(LEGS)} legs (stale JSONs ignored until rewritten)")
    while time.time() - start < timeout:
        for ds, method, seed in LEGS:
            key = (ds, method, seed)
            p = _json_path(ds, method, seed)
            if not p.exists():
                continue
            mt = p.stat().st_mtime
            if mt <= start and key in printed:
                continue
            if seen_mtime.get(key) == mt:
                continue
            if mt < start:
                # stale JSON from a previous run; wait until it is rewritten
                continue
            seen_mtime[key] = mt
            printed.add(key)
            rules = _rules_path(ds, method, seed)
            _emit(summarize(ds, method, seed, str(rules) if rules else None))
        if len(printed) == len(LEGS):
            _emit("[watch_legs] all remaining legs printed")
            return
        time.sleep(20)
    _emit("[watch_legs] timeout")


if __name__ == "__main__":
    main()
