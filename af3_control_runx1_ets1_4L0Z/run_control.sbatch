#!/bin/bash
#SBATCH --job-name=af3_runx1_ets1_tcra
#SBATCH --output=logs/af3_runx1_ets1_tcra_%j.log
#SBATCH --account=TEICHMANN-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# RUNX1-ETS1 positive control vs 4L0Z (Shrivastava et al. 2014): 6 inference-only AF3 jobs x 5 seeds,
# sequential on one GPU. Inputs built by scripts/build_inputs.py (MSAs reused from the n100_rho0.6 screen).
# Same AlphaFast image/weights and retry logic as ../af3_screen_n100_rho0.6.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
SIF="/rds/project/rds-C9woKbOCf2Y/yzy21/yy/alphafast/alphafast.sif"
WEIGHTS_DIR="/home/yzy21/rds/hpc-work/Alphafold3"
IN_DIR="${ROOT}/inputs_inference"
OUT_DIR="${ROOT}/outputs_inference"
mkdir -p "${OUT_DIR}" "${ROOT}/logs"

MAX_RETRIES=4
while IFS= read -r NAME; do
    [[ -z "${NAME}" ]] && continue
    if [[ -f "${OUT_DIR}/${NAME}/${NAME}_model.cif" || -f "${OUT_DIR}/${NAME}_model.cif" ]]; then
        echo "skip ${NAME}: already done"
        continue
    fi
    ok=0
    for attempt in $(seq 1 "${MAX_RETRIES}"); do
        echo "--- ${NAME} (attempt ${attempt}/${MAX_RETRIES}) $(date) ---"
        XLA_EXTRA=""
        [[ "${attempt}" -ge 3 ]] && XLA_EXTRA="--xla_gpu_autotune_level=0"
        if SINGULARITY_BIND="${WEIGHTS_DIR}:/data/weights,${IN_DIR}:/data/af_input,${OUT_DIR}:/data/af_output" \
            SINGULARITYENV_XLA_FLAGS="${XLA_EXTRA}" \
            singularity run --nv "${SIF}" \
            python3 /app/alphafold/run_alphafold.py \
            --json_path "/data/af_input/${NAME}.json" \
            --output_dir /data/af_output \
            --model_dir /data/weights \
            --norun_data_pipeline; then
            ok=1
            break
        fi
        echo "!!! ${NAME} failed on attempt ${attempt}; retrying after 20s"
        sleep 20
    done
    [[ "${ok}" -eq 1 ]] && echo "OK ${NAME}" || echo "XXX ${NAME} FAILED after ${MAX_RETRIES} attempts"
done < "${ROOT}/job_list.txt"
echo "=== done $(date) ==="
