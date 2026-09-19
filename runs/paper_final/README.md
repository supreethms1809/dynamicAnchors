# Paper results (2×2, C_train = f̂)

Trained from **main**. C_train = C_eval = predicted.

The previous predicted-basis lock (C_train on y, τ_C=0.10) is:

    .claude/worktrees/perturb-fid/runs/archived_ctrain_y_tc0p10_20260918/

## Launch (split seeds across machines)

Same git checkout on both machines. Paper-lock classifiers must exist under
`runs/paper_fiveseed_overlap075/classifiers/` and
`runs/wyodot_fiveseed_overlap075/dnn/classifiers/` (or set `PAPER_RUNS_DIR`).

```bash
# machine A (2 seeds)
bash revision/run_seed_major_grid.sh --go 42 43

# machine B (3 seeds)
bash revision/run_seed_major_grid.sh --go 44 45 46
```

Omit the seed list only if one machine should run all five (`42 43 44 45 46`).
JSON names include the seed, so the two trees merge with `rsync -a` into one
`runs/paper_final/`. Skip-if-exists is keyed off those JSON paths.

Per-machine log: `runs/paper_final/seed_major_<host>_s<firstseed>.out`.

## Parallelism (already on; do not serialize)

Each 2×2 **cell** (one estimator × one τ_C × one seed) runs **4 concurrent
(dataset, arm) jobs**:

| Lane | Slots | Datasets |
|---|---|---|
| big | 2 | housing, folktables_income_CA_2018, heloc, uci_adult, wyodot_kvdw_labeled, mammography |
| small | 2 | uci_credit, sick, synthetic, iris, breast_cancer, wine |

Empty lane steals from the other. RLDA and MADA for the same dataset are
separate jobs and can occupy two slots at once.

**RLDA class shards:** `run_rlda_pipeline.py` calls
`single_agent/run_parallel_classes.py` with `--parallel_classes=n_classes`
and `--n_envs=1`. Iris = 3 processes, housing = 4, wyodot = 5, all other
headline sets = 2. One shared `classifier.pth`; shards do not refit.

**MADA** is one process per dataset (all classes in one MADDPG job).

Peak load for one cell is therefore 4 pipelines, and any RLDA pipeline among
them multiplies by its class count (worst case wyodot: 5 shards in one slot).

## Live layout (source of truth; skip-if-exists keys off these)

```
emp_tc0p10/results/{ddpg,maddpg}/{ds}__{rlda,mada}__seedS__tp0p90__tc0p10.json
emp_tc0p20/results/{ddpg,maddpg}/...tc0p20.json
pert_tc0p10/results/{ddpg,maddpg}/...tc0p10.json
pert_tc0p20/results/{ddpg,maddpg}/...tc0p20.json
baselines_emp/{ds}__{cart,random_search,sp_anchors,greedy_anchors}__seedS__tp0p90__tc{0p10,0p20}.json
baselines_pert/{ds}__{cart,random_search}__...tc{0p10,0p20}.json
ablations/{family}/{arm}/results/{ddpg,maddpg}/...
```

## Collector copies (optional; tables also read live trees)

```bash
python revision/collect_paper_results.py          # copy present files
python revision/collect_paper_results.py --check  # map + missing counts
```

`results/predicted/RLDA-emp/` ← `emp_tc0p10/results/ddpg`, and the same
pattern for MADA / pert / tc020 / CART / RandS / SP-Anch / GreedyAnch.

`revision/write_paper_final_tables.py` headline ingest is empirical τ_C=0.10
(live `emp_tc0p10` + `baselines_emp`). Pert seed-42 subsection reads `pert_tc0p10`.

## Settings locked for this queue

- coverage_basis predicted (train YAML + eval `--coverage_basis`)
- precision_estimator empirical | conditional per cell
- coverage_target = eval `--tau_c` = 0.10 or 0.20 per cell
- k=1, τ_P=0.90, DDPG / MADDPG, overlap 0.75, 24k-scale
- classifiers: `runs/paper_fiveseed_overlap075` and wyodot DNN lock
- P3–P5 ablations vs emp τ_C=0.1, 11 datasets, no wyodot
- per seed order: pert 0.2 → emp 0.2 → emp 0.1 → pert 0.1 → baselines → ablations
