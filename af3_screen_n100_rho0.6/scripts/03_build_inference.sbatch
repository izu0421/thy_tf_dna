#!/bin/bash
#SBATCH --job-name=af3_build_inputs
#SBATCH --output=%x_%j.log
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Runs between Stage 1 and Stage 2: expands the 32 data.json templates into
# the 6400 per-row inference-only input JSONs.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/config.sh"
python3 "${ROOT}/scripts/03_build_inference_inputs.py"
