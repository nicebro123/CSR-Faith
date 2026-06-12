#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/env.common.sh"

DOWNLOAD_MODEL=1
DOWNLOAD_DATA=1

for arg in "$@"; do
    case "${arg}" in
        --model-only)
            DOWNLOAD_MODEL=1
            DOWNLOAD_DATA=0
            ;;
        --data-only)
            DOWNLOAD_MODEL=0
            DOWNLOAD_DATA=1
            ;;
        --skip-model)
            DOWNLOAD_MODEL=0
            ;;
        --skip-data)
            DOWNLOAD_DATA=0
            ;;
        -h|--help)
            cat <<'USAGE'
Usage: bash scripts/prepare_assets.sh [--model-only|--data-only|--skip-model|--skip-data]

Reads scripts/env.local.sh when present, creates the configured asset directories,
and downloads MODEL_PATH plus DATA_FILE into HF_HOME when they are HuggingFace repo IDs.
USAGE
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${DATA_ROOT}" "${HF_HOME}" "${CKPT_ROOT}" "${CSR_CACHE_ROOT}"

if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
else
    echo "Missing Hugging Face CLI. Install it with:" >&2
    echo "  python3 -m pip install -U huggingface_hub" >&2
    exit 1
fi

is_local_path() {
    case "$1" in
        /*|./*|../*|~/*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

download_model() {
    if is_local_path "${MODEL_PATH}"; then
        if [ -d "${MODEL_PATH}" ]; then
            echo "[assets] MODEL_PATH is local and exists: ${MODEL_PATH}"
            return
        fi
        echo "[assets] MODEL_PATH is a local path but does not exist: ${MODEL_PATH}" >&2
        exit 1
    fi
    echo "[assets] downloading model: ${MODEL_PATH}"
    "${HF_DOWNLOAD[@]}" "${MODEL_PATH}"
}

download_dataset() {
    if is_local_path "${DATA_FILE}"; then
        if [ -e "${DATA_FILE}" ]; then
            echo "[assets] DATA_FILE is local and exists: ${DATA_FILE}"
            return
        fi
        echo "[assets] DATA_FILE is a local path but does not exist: ${DATA_FILE}" >&2
        exit 1
    fi
    echo "[assets] downloading dataset: ${DATA_FILE}"
    "${HF_DOWNLOAD[@]}" "${DATA_FILE}" --repo-type dataset
}

cat <<EOF
[assets] repo root: ${REPO_ROOT}
[assets] DATA_ROOT=${DATA_ROOT}
[assets] HF_HOME=${HF_HOME}
[assets] HF_ENDPOINT=${HF_ENDPOINT}
[assets] MODEL_PATH=${MODEL_PATH}
[assets] DATA_FILE=${DATA_FILE}
[assets] CAUSAL_CRITIC_PATH=${CAUSAL_CRITIC_PATH}
EOF

if [ "${DOWNLOAD_MODEL}" -eq 1 ]; then
    download_model
fi

if [ "${DOWNLOAD_DATA}" -eq 1 ]; then
    download_dataset
fi

echo "[assets] done"
