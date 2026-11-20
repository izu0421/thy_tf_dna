#!/bin/bash
# Submit the rerun shards produced by scripts/generate_rerun_shards.py
# (1 GPU / 36h / ~10 jobs each) for inference jobs missing an output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./config.sh

SBATCH_COMMON=(--partition="${SLURM_PARTITION}" --nodes=1 --ntasks=1)
[[ -n "${SLURM_ACCOUNT}" ]] && SBATCH_COMMON+=(--account="${SLURM_ACCOUNT}")

# Safe to re-run: shards already in the queue are skipped, and slurmctld socket
# timeouts (common when the scheduler is busy) are retried.
QUEUED=$(squeue -u "$USER" -h -o "%j" || true)

n=0
for f in slurm_shards/infer_rerun_*.sbatch; do
    name="af3_rerun_$(basename "$f" .sbatch | sed 's/^infer_rerun_//')"
    if grep -qx "${name}" <<<"${QUEUED}"; then
        echo "skip ${name}: already queued"
        continue
    fi
    for attempt in 1 2 3 4 5; do
        if jid=$(sbatch --parsable "${SBATCH_COMMON[@]}" "$f" 2>&1); then
            echo "submitted $(basename "$f") -> job ${jid}"
            n=$((n + 1))
            break
        fi
        echo "!!! ${name} submit attempt ${attempt} failed: ${jid}"
        sleep 15
    done
    sleep 2
done
echo "=== ${n} rerun jobs submitted ==="
