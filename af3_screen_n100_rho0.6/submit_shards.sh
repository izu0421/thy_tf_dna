#!/bin/bash
# Submit all 64 inference shards (1 GPU / 10h / ~100 jobs each).
# Run this AFTER Stage 1 (MSA) + inputs_inference/ have been built - i.e. after
# submit_all.sh's first two stages have completed, or pass a jobid to wait on:
#   ./submit_shards.sh                 # submit immediately
#   ./submit_shards.sh 123456          # submit with --dependency=afterok:123456
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./config.sh

DEP=()
[[ $# -ge 1 ]] && DEP=(--dependency=afterok:"$1")

# --nodes=1 --ntasks=1 is required by CSD3's site policy for single-node jobs.
SBATCH_COMMON=(--partition="${SLURM_PARTITION}" --nodes=1 --ntasks=1)
[[ -n "${SLURM_ACCOUNT}" ]] && SBATCH_COMMON+=(--account="${SLURM_ACCOUNT}")

n=0
for f in slurm_shards/infer_shard_*.sbatch; do
    jid=$(sbatch --parsable "${SBATCH_COMMON[@]}" "${DEP[@]}" "$f")
    echo "submitted $(basename "$f") -> job ${jid}"
    n=$((n + 1))
done
echo "=== ${n} shard jobs submitted ==="
echo "Track with: squeue -u \$USER"
