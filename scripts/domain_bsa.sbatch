#!/bin/bash
#SBATCH --job-name=domain_bsa
#SBATCH --output=logs/domain_bsa_%j.log
#SBATCH --account=TEICHMANN-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
# Domain-only TF1-TF2 BSA for all 6,400 screen models. Submit from the project root:
#   sbatch scripts/domain_bsa.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
/rfs/project/rfs-iCNyzSAaucw/yzy21/micromamba/envs/stcrpy/bin/python scripts/domain_bsa.py "${SLURM_CPUS_PER_TASK}"
