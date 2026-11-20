#!/bin/bash
#SBATCH --job-name=af3_msa
#SBATCH --output=%x_%A_%a.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-32

# Stage 1: MSA search for each of the 32 unique (TF1,TF2,combo) protein pairs.
# Submit with: sbatch --partition=$SLURM_PARTITION --gres=$SLURM_GPU_GRES \
#   --array=1-32%$SLURM_MSA_ARRAY_THROTTLE scripts/02_msa.sbatch
# (the wrapping is done by submit_all.sh - run that instead of this directly).
set -uo pipefail
source "${SLURM_SUBMIT_DIR}/config.sh"

NAME="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ROOT}/job_list_msa.txt")"
IN_DIR="${ROOT}/inputs_msa"
OUT_DIR="${ROOT}/outputs_msa"
mkdir -p "${OUT_DIR}"

if [[ -f "${OUT_DIR}/${NAME}_data.json" ]]; then
    echo "skip ${NAME}: already done"
    exit 0
fi

MAX_RETRIES=4
for attempt in $(seq 1 "${MAX_RETRIES}"); do
    echo "--- MSA: ${NAME} (attempt ${attempt}/${MAX_RETRIES}) ---"
    if SINGULARITY_BIND="${DB_DIR}:/data/db,${IN_DIR}:/data/af_input,${OUT_DIR}:/data/af_output" \
        singularity run --nv "${SIF}" \
        python3 /app/alphafold/run_data_pipeline.py \
        --json_path "/data/af_input/${NAME}.json" \
        --output_dir /data/af_output \
        --mmseqs_db_dir /data/db/mmseqs \
        --pdb_database_path /data/db/mmcif_files \
        --gpu_device 0 \
        --template_mode "${TEMPLATE_MODE}" \
        --max_template_date "${MAX_TEMPLATE_DATE}"; then
        # run_data_pipeline.py writes outputs_msa/<name>/<name>_data.json - flatten it.
        [[ -f "${OUT_DIR}/${NAME}/${NAME}_data.json" ]] && mv "${OUT_DIR}/${NAME}/${NAME}_data.json" "${OUT_DIR}/${NAME}_data.json"
        echo "OK ${NAME}"
        exit 0
    fi
    echo "!!! ${NAME} failed on attempt ${attempt}; retrying after 30s"
    sleep 30
done
echo "XXX ${NAME} FAILED after ${MAX_RETRIES} attempts"
exit 1
