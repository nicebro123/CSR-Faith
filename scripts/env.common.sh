#!/usr/bin/env bash
# Shared machine/environment defaults for CSR-Faith scripts.
# Machine-specific overrides live in scripts/env.local.sh and are not committed.

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${COMMON_SCRIPT_DIR}/env.local.sh" ]; then
    source "${COMMON_SCRIPT_DIR}/env.local.sh"
fi

export DATA_ROOT="${DATA_ROOT:-../csr_faith_assets}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
export CKPT_ROOT="${CKPT_ROOT:-${DATA_ROOT}/ckpts}"
export CSR_CACHE_ROOT="${CSR_CACHE_ROOT:-${DATA_ROOT}/cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
export DATA_FILE="${DATA_FILE:-hunarbatra/STVQA-7K}"
export CAUSAL_CRITIC_PATH="${CAUSAL_CRITIC_PATH:-${CKPT_ROOT}/causal_spatial_critic}"

unset COMMON_SCRIPT_DIR
