#!/usr/bin/env python3
"""Generate per-sweep-point copies of the two anchor config files.

The precision/coverage targets are read from several places across training,
inference and rule testing, all of which resolve the config path relative to
their own script. Rather than plumb overrides through every reader, each sweep
point gets a complete copy of both config files with only the two target values
changed; the sweep driver copies the pair over the live configs before running.

Only lines that are actual `precision_target:` / `coverage_target:` keys are
rewritten. Both names also appear inside explanatory comments in anchor.yaml
(e.g. "floor = min(precision_target * 0.8, ...)"), which must be left alone, so
matching is anchored to an indented `key: <number>` form. Everything else in the
file — comments, ordering, unrelated values — is copied byte for byte.

Usage:
    python sweep_configs/generate_sweep_configs.py
    python sweep_configs/generate_sweep_configs.py --precision 0.90 \
        --coverages 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40
"""

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MA_CONFIG = REPO / "BenchMARL" / "conf" / "anchor.yaml"
SA_CONFIG = REPO / "single_agent" / "conf" / "anchor_single.yaml"

# An indented key whose value is a bare number. Comment lines start with '#'
# after optional whitespace and therefore never match.
KEY_RE = r"^(?P<indent>[ \t]+)(?P<key>{key}):(?P<space>[ \t]*)(?P<value>-?\d+(?:\.\d+)?)(?P<rest>.*)$"


def set_key(text: str, key: str, value: float, path: Path) -> str:
    """Replace the single `key: <number>` line in `text`. Raises if not exactly one."""
    pattern = re.compile(KEY_RE.format(key=re.escape(key)), re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise SystemExit(
            f"{path}: expected exactly one '{key}:' key line, found {len(matches)}"
            + (f" (lines {[text[:m.start()].count(chr(10)) + 1 for m in matches]})" if matches else "")
        )

    def repl(m: re.Match) -> str:
        # Keep any trailing inline comment so the file stays self-documenting.
        return f"{m.group('indent')}{m.group('key')}:{m.group('space')}{value:g}{m.group('rest')}"

    return pattern.sub(repl, text, count=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", type=float, default=0.90,
                    help="Fixed precision_target for every sweep point (default: 0.90)")
    ap.add_argument("--coverages", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                    help="coverage_target values, one sweep point each")
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent),
                    help="Where to write the per-point config directories")
    args = ap.parse_args()

    ma_src = MA_CONFIG.read_text()
    sa_src = SA_CONFIG.read_text()
    # Resolve so a relative --out_dir (e.g. "sweep_configs/foo") still prints and
    # compares cleanly against REPO below.
    out_root = Path(args.out_dir).resolve()

    print(f"Source configs:\n  {MA_CONFIG}\n  {SA_CONFIG}")
    print(f"precision_target = {args.precision:g} for all points\n")

    for cov in args.coverages:
        point = out_root / f"c{cov:.2f}"
        point.mkdir(parents=True, exist_ok=True)

        ma = set_key(ma_src, "precision_target", args.precision, MA_CONFIG)
        ma = set_key(ma, "coverage_target", cov, MA_CONFIG)
        sa = set_key(sa_src, "precision_target", args.precision, SA_CONFIG)
        sa = set_key(sa, "coverage_target", cov, SA_CONFIG)

        (point / "anchor.yaml").write_text(ma)
        (point / "anchor_single.yaml").write_text(sa)
        shown = point.relative_to(REPO) if point.is_relative_to(REPO) else point
        print(f"  {shown}/  ->  Pthr={args.precision:g}  Cthr={cov:g}")

    print(f"\n{len(args.coverages)} sweep points written under {out_root}")
    print("The sweep driver copies each pair over:")
    print(f"  {MA_CONFIG.relative_to(REPO)}")
    print(f"  {SA_CONFIG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
