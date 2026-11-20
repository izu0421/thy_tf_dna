# AF3 batched run: n100_rho0.6 (6,400 rows / 32 unique protein-pair MSAs)

Self-contained folder - transfer as-is to the HPC.

## Status
Complete - all 6,400/6,400 inference jobs have a `_model.cif` (Stage 1 job
33870829, Stage 2 shards 33945389+, plus 6 rerun shards for stragglers).
`inputs_inference/` was deleted after completion (171 GB); rebuild it with
`sbatch scripts/03_build_inference.sbatch` - each job's input is also kept as
`outputs_inference/<name>_data.json`. Analysis notebooks live in `../notebooks/` and read `outputs_inference/` directly.

## What's here
- `data/` (symlink to the project-level `../data/`) - the input CSV and `tf_sequences_run.json` (23 full-length UniProt
  sequences for every TF used in this CSV; the 12 not already cached in the
  parent project's `tf_sequences.json` were fetched fresh and verified against
  UniProt).
- `scripts/01_build_msa_inputs.py` - writes 32 Stage-1 (MSA) job JSONs to
  `inputs_msa/`, one per unique (TF1, TF2, motif_combination). Already run
  locally - `inputs_msa/` ships with the folder.
- `scripts/02_msa.sbatch` - SLURM array (32 tasks): runs `run_data_pipeline.py`
  for each pair -> `outputs_msa/<name>_data.json` (MSA embedded).
- `scripts/03_build_inference_inputs.py` / `03_build_inference.sbatch` - clones
  each pair's `_data.json` once per CSV row, swapping in that row's real DNA
  (chain C) + reverse complement (chain D) -> `inputs_inference/` (6,400 files).
- `scripts/00_list_inference_jobs.py` / `scripts/generate_inference_shards.py` -
  already run locally. They split the 6,400 jobs into **64 shards of 100**,
  each with its own sbatch file in `slurm_shards/` requesting **1 GPU for
  10h**. At ~4 min/job, 100 jobs/shard takes ~6.7h, leaving headroom in the
  10h window for retries/queue variance.
- `submit_shards.sh` - submits all 64 shard jobs (optionally after a dependency).
- `submit_all.sh` - runs the whole chain: build Stage-1 inputs -> MSA array ->
  build Stage-2 inputs -> submit all 64 shards.

## Before running
Edit `config.sh`: `SIF`, `WEIGHTS_DIR`, `DB_DIR` (AlphaFast image / AF3 weights
/ MSA+template DBs - not included here, multi-GB/TB), and
`SLURM_PARTITION`/`SLURM_ACCOUNT`/`SLURM_GPU_GRES` to match the HPC.

On CSD3, `SLURM_ACCOUNT` also determines the GPU QoS and its concurrency cap
(`sacctmgr show qos <qos> format=maxtresperuser`). `TEICHMANN-SL3-GPU` maps to
QoS `gpu2` (max 32 concurrent GPUs); `TEICHLAB-SL2-GPU` maps to `gpu1` (max 64)
- since Stage 2 submits 64 one-GPU shards at once, prefer the account with the
  higher cap so all shards can run concurrently instead of half of them
  queuing. Partition-wide node availability (`sinfo -p ampere`) still governs
  actual queue time on top of this.

## Run
```
./submit_all.sh
```
Or run the stages yourself:
```
sbatch scripts/02_msa.sbatch                 # Stage 1: 32-task MSA array
sbatch --dependency=afterok:<msa_jobid> scripts/03_build_inference.sbatch
./submit_shards.sh <build_jobid>             # Stage 2: 64 one-GPU/10h shards
```
Every stage skips already-finished work (checks for `_data.json` /
`_model.cif`), so re-running after a partial failure, a shard timeout, or a
failed retry is safe - just resubmit.

## Troubleshooting
- **`.../config.sh: No such file or directory` in a job log, stage silently
  using stale/default env vars, or a downstream stage failing instantly with
  a `Dependency` failure**: SLURM copies submitted scripts into
  `/var/spool/slurm/slurmd/job<id>/slurm_script` before running them, so
  `$(dirname "${BASH_SOURCE[0]}")` no longer points at this folder inside the
  job. All `scripts/*.sbatch` and `slurm_shards/*.sbatch` now source
  `"${SLURM_SUBMIT_DIR}/config.sh"` instead, which SLURM sets to the directory
  `sbatch` was run from. If you hand-edit or regenerate any sbatch file
  (e.g. via `scripts/generate_inference_shards.py`), keep this pattern.
  `02_msa.sbatch` uses `set -uo pipefail` (no `-e`), so this failure mode was
  silent there (jobs still completed using inherited env vars from
  `submit_all.sh`'s own `source ./config.sh`); `03_build_inference.sbatch` uses
  `set -euo pipefail`, so it failed fast and cancelled all 64 dependent shard
  jobs.

## Estimated wall time (from prior runs, ~H100)
- Stage 1: ~20-28 min/job x 32 jobs, divided by `SLURM_MSA_ARRAY_THROTTLE`.
- Stage 2: 64 shards x ~6.7h each, running as many at once as the HPC's GPU
  allocation allows (fully parallel across shards - no shared state).

## Output
Everything lands in **`outputs_inference/`** as a flat directory:
```
outputs_inference/<TF1>_<TF2>_<combo>_<class>_bg<idx>_model.cif
outputs_inference/<TF1>_<TF2>_<combo>_<class>_bg<idx>_confidences.json
outputs_inference/<TF1>_<TF2>_<combo>_<class>_bg<idx>_summary_confidences.json
outputs_inference/<TF1>_<TF2>_<combo>_<class>_bg<idx>_ranking_scores.csv
```
Plus `seed-1_sample-{0..4}/`: the five per-sample models of every job (the
top-level files are AF3's top-ranked sample). SLURM logs are in `logs/`.
