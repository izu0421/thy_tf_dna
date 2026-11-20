#!/bin/bash
#SBATCH --job-name=af3_infer
#SBATCH --output=%x_%A_%a.log
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-6400

# Stage 2: pure inference (--norun_data_pipeline) for every one of the 6400 rows.
# Embarrassingly parallel - no shared DB access, so this is safe to run at
# whatever concurrency the GPU allocation allows.
set -uo pipefail
source "${SLURM_SUBMIT_DIR}/config.sh"

NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ROOT}/job_list_inference.txt")"
IN_DIR="${ROOT}/inputs_inference"
OUT_DIR="${ROOT}/outputs_inference"
mkdir -p "${OUT_DIR}"

if [[ -f "${OUT_DIR}/${NAME}_model.cif" ]]; then
    echo "skip ${NAME}: already done"
    exit 0
fi

MAX_RETRIES=4
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "--- Predicting: ${NAME} (attempt ${attempt}/${MAX_RETRIES}) ---"
    if SINGULARITY_BIND="${WEIGHTS_DIR}:/data/weights,${IN_DIR}:/data/af_input,${OUT_DIR}:/data/af_output" \
        singularity run --nv "${SIF}" \
        python3 /app/alphafold/run_alphafold.py \
        --json_path "/data/af_input/${NAME}.json" \
        --output_dir /data/af_output \
        --model_dir /data/weights \
        --norun_data_pipeline; then
        echo "OK ${NAME}"
        exit 0
    fi
    echo "!!! ${NAME} failed on attempt ${attempt}; retrying after 20s"
    sleep 20
done
echo "XXX ${NAME} FAILED after ${MAX_RETRIES} attempts"
exit 1
