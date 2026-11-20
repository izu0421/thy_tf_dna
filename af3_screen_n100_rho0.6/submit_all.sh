#!/bin/bash
# Chains the whole pipeline as SLURM dependencies: build MSA inputs (local,
# runs now) -> Stage 1 MSA array -> build inference inputs -> Stage 2 array.
# Safe to re-run: every stage skips work whose output already exists.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./config.sh

echo "=== Building Stage-1 (MSA) input JSONs ==="
python3 scripts/01_build_msa_inputs.py

# --nodes=1 --ntasks=1 is required by CSD3's site policy for single-node jobs.
SBATCH_COMMON=(--partition="${SLURM_PARTITION}" --nodes=1 --ntasks=1)
[[ -n "${SLURM_ACCOUNT}" ]] && SBATCH_COMMON+=(--account="${SLURM_ACCOUNT}")

echo "=== Submitting Stage 1: MSA array (32 jobs) ==="
MSA_JOBID=$(sbatch --parsable "${SBATCH_COMMON[@]}" \
    --gres="${SLURM_GPU_GRES}" \
    --array="1-32%${SLURM_MSA_ARRAY_THROTTLE}" \
    scripts/02_msa.sbatch)
echo "MSA array job: ${MSA_JOBID}"

# The build job is CPU-only, so it runs on the CPU partition/account, not the
# GPU one (CSD3 rejects GPU-partition jobs that don't request a --gres).
BUILD_SBATCH_COMMON=(--partition="${SLURM_CPU_PARTITION}" --nodes=1 --ntasks=1)
[[ -n "${SLURM_CPU_ACCOUNT}" ]] && BUILD_SBATCH_COMMON+=(--account="${SLURM_CPU_ACCOUNT}")

echo "=== Submitting inference-input build (after MSA array completes) ==="
BUILD_JOBID=$(sbatch --parsable "${BUILD_SBATCH_COMMON[@]}" \
    --dependency=afterok:"${MSA_JOBID}" \
    scripts/03_build_inference.sbatch)
echo "Build job: ${BUILD_JOBID}"

echo "=== Submitting Stage 2: 64 inference shards (100 jobs/shard, 1 GPU, 10h each) ==="
./submit_shards.sh "${BUILD_JOBID}"

echo
echo "Track with: squeue -u \$USER"
echo "Final structures land in: ${ROOT}/outputs_inference/"
