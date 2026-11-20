#!/bin/bash
# Edit these paths for the HPC before submitting anything.
# ROOT is auto-detected as the folder this file lives in - transfer the whole
# folder as one unit and these relative paths keep working.
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# AlphaFast (AlphaFold 3) singularity image and model weights.
# These are NOT included in this folder (multi-GB) - point at wherever they
# live on the HPC, or transfer them separately and update these paths.
export SIF="/rds/project/rds-C9woKbOCf2Y/yzy21/yy/alphafast/alphafast.sif"
export WEIGHTS_DIR="/home/yzy21/rds/hpc-work/Alphafold3"

# Only needed for Stage 1 (MSA search).
export DB_DIR="/rds/project/rds-C9woKbOCf2Y/yzy21/yy/alphafast/alphafastdb"          # contains mmseqs/ and mmcif_files/
export TEMPLATE_MODE="default"
export MAX_TEMPLATE_DATE="2026-05-01"

# SLURM resources - edit to match the HPC's partition/account/GPU naming.
export SLURM_PARTITION="ampere"
export SLURM_ACCOUNT="TEICHLAB-SL2-GPU"
export SLURM_CPU_PARTITION="icelake"
export SLURM_CPU_ACCOUNT="TEICHMANN-SL3-CPU"
export SLURM_GPU_GRES="gpu:1"
export SLURM_MSA_ARRAY_THROTTLE=8     # concurrent Stage-1 jobs (MMseqs2-GPU is memory-hungry)
# Stage 2 concurrency is no longer an array throttle - it's just how many of the
# 64 one-GPU/12h shard jobs (submit_shards.sh) the scheduler lets run at once.
