#!/usr/bin/env python3
"""Pilot: train RLDA + MADA on Anchors-style perturbed Fid (precision_estimator: conditional).

12 datasets x {RLDA ddpg, MADA maddpg} x seed 42, at the paper_fiveseed_overlap075
budget and lock config, differing ONLY in precision_estimator. Coverage stays on
real rows. Seed-42 classifiers are copied from the paper-lock trees so the black
box is identical to the comparison runs.

Scheduling: 4 slots = 2 "big" lane + 2 "small" lane, so the long datasets never
occupy every slot. When one lane's queue is empty its slot takes work from the
other lane.

  python revision/run_perturb_fid_seed42.py
  python revision/run_perturb_fid_seed42.py --smoke     # iris only, tiny budget

Ablation mode (defaults reproduce the headline config exactly):
  --override KEY=VALUE   edit one env_config key in both YAMLs (repeatable)
  --mada_algo masac / --rlda_algo sac, --classifier_type random_forest,
  --mada_frames_mult 3 (apc=1 arms: matched total agent-steps),
  --datasets ..., --coverage_basis predicted, --with_baselines
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "revision"))
from run_mada_pipeline import DATASET_CONFIGS as MADA_CFG  # noqa: E402

PY = "/opt/anaconda3/envs/marl/bin/python"
SEED = 42  # set from --seed in main(); the paper lock has 42-46
# Main checkout holding the paper-lock runs (this file may live in a worktree).
MAIN_RUNS = Path(os.environ.get("PAPER_RUNS_DIR", REPO.parents[2] / "runs"
                                if REPO.parent.name == "worktrees" else REPO / "runs"))

# Lanes sized from the seed-43 paper sweep wall times (RLDA+MADA minutes).
BIG = ["housing", "folktables_income_CA_2018", "heloc", "uci_adult",
       "wyodot_kvdw_labeled", "mammography"]
SMALL = ["uci_credit", "sick", "synthetic", "iris", "breast_cancer", "wine"]
LANE_SLOTS = {"big": 2, "small": 2}

MADA_BASE = 24_000
MADA_SCALE = MADA_BASE / 360_000
EVAL_INTERVAL = 4800
TAU_P, TAU_C, K = 0.90, 0.10, 1


def _tc_tag(tau_c: float) -> str:
    return f"{float(tau_c):.2f}".replace(".", "p")

BASE_ENV = os.environ.copy()
BASE_ENV.update({
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "WANDB_MODE": "offline", "WANDB_SILENT": "true", "DISABLE_WANDB": "1",
    "PYTHONUNBUFFERED": "1",
})


def mada_frames(dataset: str) -> int:
    raw = int(MADA_CFG[dataset]["ma_frames"] * MADA_SCALE)
    return max(EVAL_INTERVAL, (raw // EVAL_INTERVAL) * EVAL_INTERVAL)


def rlda_timesteps(dataset: str) -> int:
    return mada_frames(dataset) * 3 * max(int(MADA_CFG[dataset]["n_classes"]), 1)


def paper_classifier(dataset: str) -> Path:
    if dataset.startswith("wyodot"):
        return MAIN_RUNS / "wyodot_fiveseed_overlap075" / "dnn" / "classifiers" / f"{dataset}_seed{SEED}.pth"
    return MAIN_RUNS / "paper_fiveseed_overlap075" / "classifiers" / f"{dataset}_seed{SEED}.pth"


def write_estimator_yaml(src: Path, dst: Path, estimator: str, overrides=None,
                         require_overrides: bool = True) -> None:
    text = src.read_text()
    new, n = re.subn(r"(?m)^(\s*precision_estimator:\s*)\S+", r"\g<1>" + estimator, text)
    if n != 1:
        raise SystemExit(f"expected one precision_estimator key in {src}, found {n}")
    for key, val in (overrides or {}).items():
        new, n = re.subn(rf"(?m)^([ \t]*{re.escape(key)}:[ \t]*)[^#\n]*?([ \t]*(#.*)?)$",
                         lambda m: m.group(1) + val + (("  " + m.group(3)) if m.group(3) else ""), new)
        if n > 1 or (n == 0 and require_overrides):
            raise SystemExit(f"override {key}: expected one key in {src}, found {n}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new)


class Sweep:
    def __init__(self, root: Path, datasets_big, datasets_small, smoke: bool,
                 arms=("rlda", "mada"), estimator: str = "conditional", overrides=None,
                 mada_algo: str = "maddpg", rlda_algo: str = "ddpg", classifier_type: str = "dnn",
                 mada_frames_mult: float = 1.0, coverage_basis: str | None = None,
                 with_baselines: bool = False, tau_c: float | None = None):
        self.root = root
        self.overrides = dict(overrides or {})
        self.mada_algo, self.rlda_algo = mada_algo, rlda_algo
        self.classifier_type = classifier_type
        self.mada_frames_mult = mada_frames_mult
        self.coverage_basis = coverage_basis
        self.with_baselines = with_baselines
        if tau_c is not None:
            self.tau_c = float(tau_c)
        elif "coverage_target" in self.overrides:
            self.tau_c = float(self.overrides["coverage_target"])
        else:
            self.tau_c = float(TAU_C)
        self.logs = root / "logs"
        self.smoke = smoke
        self.estimator = estimator
        self.anchor_cfg = root / "conf" / "anchor.yaml"
        self.single_cfg = root / "conf" / "anchor_single.yaml"
        self.queues = {
            "big": [(ds, arm) for ds in datasets_big for arm in arms],
            "small": [(ds, arm) for ds in datasets_small for arm in arms],
        }

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.logs / "sweep_progress.log", "a") as fh:
            fh.write(line + "\n")

    def result_path(self, arm: str, dataset: str) -> Path:
        sub = self.mada_algo if arm == "mada" else self.rlda_algo
        return self.root / "results" / sub / (
            f"{dataset}__{arm}__seed{SEED}__tp{_tc_tag(TAU_P)}__tc{_tc_tag(self.tau_c)}.json"
        )

    def done(self, arm: str, dataset: str) -> bool:
        f = self.result_path(arm, dataset)
        return f.exists() and f.stat().st_size > 0

    def prepare(self) -> None:
        if self.coverage_basis:
            self.overrides.setdefault("coverage_basis", self.coverage_basis)
        write_estimator_yaml(REPO / "BenchMARL" / "conf" / "anchor.yaml", self.anchor_cfg, self.estimator,
                             self.overrides)
        # Overrides are MADA ablations unless RLDA is trained in this sweep; the
        # single-agent YAML keeps its multi-agent fields at 0 otherwise.
        rlda_in_sweep = any(a == "rlda" for q in self.queues.values() for _, a in q)
        write_estimator_yaml(REPO / "single_agent" / "conf" / "anchor_single.yaml", self.single_cfg,
                             self.estimator, self.overrides if rlda_in_sweep else None,
                             require_overrides=False)
        if self.overrides:
            self.log(f"overrides: {self.overrides}")
        (self.root / "classifiers").mkdir(parents=True, exist_ok=True)
        for q in self.queues.values():
            for ds in sorted({d for d, _ in q}):
                dst = self.root / "classifiers" / f"{ds}_seed{SEED}.pth"
                if dst.exists():
                    continue
                if self.classifier_type != "dnn":
                    # Fit once here so the RLDA and MADA jobs never race to fit it.
                    lp = self.logs / f"fit_{self.classifier_type}_{ds}_seed{SEED}.log"
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    rc = subprocess.run(
                        [PY, str(REPO / "revision" / "fit_shared_classifier.py"), "--dataset", ds,
                         "--out", str(dst), "--seed", str(SEED), "--device", "cpu",
                         "--classifier_type", self.classifier_type],
                        cwd=str(REPO), env=BASE_ENV, stdout=open(lp, "w"), stderr=subprocess.STDOUT,
                    ).returncode
                    if rc != 0 or not dst.exists():
                        raise SystemExit(f"fitting {self.classifier_type} classifier for {ds} failed (see {lp})")
                    self.log(f"classifier {ds} fitted ({self.classifier_type})")
                    continue
                src = paper_classifier(ds)
                if not src.exists():
                    raise SystemExit(f"missing paper classifier {src}")
                shutil.copy2(src, dst)
                self.log(f"classifier {ds} <- {src}")

    def budgets(self, dataset: str) -> tuple[int, int]:
        if self.smoke:
            # RLDA FidCov eval_freq is 3000 steps per class: give each class several evals.
            return EVAL_INTERVAL, 15000 * max(int(MADA_CFG[dataset]["n_classes"]), 1)
        mf = mada_frames(dataset)
        if self.mada_frames_mult != 1.0:
            mf = max(EVAL_INTERVAL, int(mf * self.mada_frames_mult) // EVAL_INTERVAL * EVAL_INTERVAL)
        return mf, rlda_timesteps(dataset)

    def cmd_for(self, arm: str, dataset: str) -> list[str]:
        mf, rt = self.budgets(dataset)
        common = ["--datasets", dataset, "--seed", str(SEED), "--device", "cpu",
                  "--root", str(self.root), "--force-train", "--force",
                  "--tau_p", str(TAU_P), "--tau_c", str(self.tau_c), "--k", str(K),
                  "--classifier_type", self.classifier_type]
        if arm == "mada":
            return [PY, str(REPO / "revision" / "run_mada_pipeline.py"), "--algo", self.mada_algo,
                    *common, "--max_n_frames", str(mf), "--anchor-config", str(self.anchor_cfg)]
        return [PY, str(REPO / "revision" / "run_rlda_pipeline.py"), "--algo", self.rlda_algo,
                *common, *([] if self.with_baselines else ["--skip-baselines"]),
                "--sa_timesteps", str(rt)]

    def env_for(self, arm: str) -> dict:
        env = dict(BASE_ENV)
        env["ANCHOR_CONFIG"] = str(self.anchor_cfg)
        env["ANCHOR_SINGLE_CONFIG"] = str(self.single_cfg)
        if self.coverage_basis:
            env["DYNANC_COVERAGE_BASIS"] = self.coverage_basis
        return env

    def run(self) -> int:
        self.prepare()
        for lane in self.queues:
            self.queues[lane] = [j for j in self.queues[lane] if not self.done(j[1], j[0])]
        self.log(f"PERTURB-FID PILOT root={self.root} smoke={self.smoke} "
                 f"estimator={self.estimator} tau_c={self.tau_c} "
                 f"arms={sorted({a for q in self.queues.values() for _, a in q})}")
        self.log(f"queues: big={self.queues['big']}  small={self.queues['small']}")
        running: dict = {}
        failures = 0

        def lane_count(lane):
            return sum(1 for v in running.values() if v[2] == lane)

        while any(self.queues.values()) or running:
            for lane, slots in LANE_SLOTS.items():
                other = "small" if lane == "big" else "big"
                while lane_count(lane) < slots:
                    src = lane if self.queues[lane] else other
                    if not self.queues[src]:
                        break
                    ds, arm = self.queues[src].pop(0)
                    lp = self.logs / f"seed{SEED}_{arm}_{ds}.log"
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    p = subprocess.Popen(
                        self.cmd_for(arm, ds), cwd=str(REPO), env=self.env_for(arm),
                        stdout=open(lp, "w"), stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    running[p] = (ds, arm, lane, time.time())
                    mf, rt = self.budgets(ds)
                    budget = f"mada={mf}f" if arm == "mada" else f"rlda={rt} total"
                    self.log(f"  start [{lane} slot{'' if src == lane else ' <- ' + src}] "
                             f"{ds} {arm} pid={p.pid} ({budget})")
            time.sleep(5 if self.smoke else 30)
            for p in list(running):
                if p.poll() is None:
                    continue
                ds, arm, lane, t0 = running.pop(p)
                ok = p.returncode == 0 and self.done(arm, ds)
                failures += 0 if ok else 1
                self.log(f"  {'DONE' if ok else 'FAILED'} [{lane}] {ds} {arm} rc={p.returncode} "
                         f"({(time.time() - t0) / 60:.1f} min)")
        self.log(f"ALL RUNS DONE  failures={failures}")
        if not self.smoke:
            self.finalize()
        self.log(f"ALL DONE  failures={failures}")
        return 1 if failures else 0

    def finalize(self) -> None:
        """Re-score selected rules under both estimators, then write comparison.md."""
        diag = self.root / "diagnostics" / "dual_estimator"
        for cmd, name in (
            ([PY, str(REPO / "revision" / "dual_estimator_rescore.py"),
              "--results_dir", str(self.root / "results"), "--out_dir", str(diag)], "rescore"),
            ([PY, str(REPO / "revision" / "compare_perturb_fid.py"), "--root", str(self.root)], "compare"),
        ):
            with open(self.logs / f"finalize_{name}.log", "w") as fh:
                rc = subprocess.run(cmd, cwd=str(REPO), env=BASE_ENV,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            self.log(f"  finalize {name}: rc={rc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="iris only, one eval interval")
    ap.add_argument("--root", default=None,
                    help="Run tree. Headline 2x2 cells live under runs/paper_final/<cell>/ .")
    ap.add_argument(
        "--rerun_mada", nargs="+", default=None, metavar="DATASET",
        help="Retrain only MADA for these datasets (e.g. jobs that ran before the "
             "box-support checkpoint fix). Their old MADA JSONs move to "
             "results/maddpg/_archive_<stamp>/ first.",
    )
    ap.add_argument(
        "--estimator", default="conditional", choices=["conditional", "empirical"],
        help="precision_estimator written into the run's YAMLs. 'empirical' + --arms mada "
             "reproduces the paper-lock MADA config on the current code.",
    )
    ap.add_argument("--arms", nargs="+", default=["rlda", "mada"], choices=["rlda", "mada"])
    ap.add_argument("--seed", type=int, default=42, help="Paper-lock seeds are 42-46.")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="Set one env_config key in the run's YAMLs (ablations).")
    ap.add_argument("--mada_algo", default="maddpg", choices=["maddpg", "masac"])
    ap.add_argument("--rlda_algo", default="ddpg", choices=["ddpg", "sac"])
    ap.add_argument("--classifier_type", default="dnn", choices=["dnn", "random_forest"])
    ap.add_argument("--mada_frames_mult", type=float, default=1.0)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--coverage_basis", default=None, choices=["true_label", "predicted"])
    ap.add_argument(
        "--tau_c", type=float, default=None,
        help="Eval τ_C written into result JSONs and success_rate. "
             "Default: coverage_target override if set, else 0.10.",
    )
    ap.add_argument("--with_baselines", action="store_true",
                    help="Also run the 4 empirical baselines (with the RLDA job).")
    args = ap.parse_args()
    global SEED
    SEED = int(args.seed)
    if args.smoke:
        root = Path(args.root or REPO / "runs" / "perturb_fid_smoke")
    else:
        root = Path(args.root or REPO / "runs" / "perturb_fid_seed42").resolve()
    if args.smoke:
        return Sweep(root, [], ["iris"], smoke=True).run()
    if args.rerun_mada:
        unknown = sorted(set(args.rerun_mada) - set(BIG) - set(SMALL))
        if unknown:
            raise SystemExit(f"unknown datasets: {unknown}")
        archive = root / "results" / "maddpg" / f"_archive_{datetime.now():%Y%m%d_%H%M%S}"
        archive.mkdir(parents=True, exist_ok=True)
        for ds in args.rerun_mada:
            for f in (root / "results" / "maddpg").glob(f"{ds}__mada__*seed{SEED}*.json"):
                shutil.move(str(f), archive / f.name)
            for f in [root / "logs" / f"seed{SEED}_mada_{ds}.log", root / "logs" / f"{ds}_maddpg"]:
                if f.exists():
                    shutil.move(str(f), archive / f"logs_{f.name}")
        return Sweep(root, [d for d in BIG if d in args.rerun_mada],
                     [d for d in SMALL if d in args.rerun_mada], smoke=False, arms=("mada",)).run()
    overrides = {}
    for kv in args.override:
        k, _, v = kv.partition("=")
        if not k or not v:
            raise SystemExit(f"bad --override {kv!r}, expected KEY=VALUE")
        overrides[k.strip()] = v.strip()
    big, small = BIG, SMALL
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(BIG) - set(SMALL))
        if unknown:
            raise SystemExit(f"unknown datasets: {unknown}")
        big = [d for d in BIG if d in args.datasets]
        small = [d for d in SMALL if d in args.datasets]
    return Sweep(root, big, small, smoke=False, arms=tuple(args.arms), estimator=args.estimator,
                 overrides=overrides, mada_algo=args.mada_algo, rlda_algo=args.rlda_algo,
                 classifier_type=args.classifier_type, mada_frames_mult=args.mada_frames_mult,
                 coverage_basis=args.coverage_basis, with_baselines=args.with_baselines,
                 tau_c=args.tau_c).run()


if __name__ == "__main__":
    sys.exit(main())
