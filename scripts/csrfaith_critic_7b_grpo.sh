#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.common.sh"

bash "${SCRIPT_DIR}/csrfaith_7b_grpo.sh" \
    algorithm.enable_causal_spatial_critic=True \
    algorithm.causal_critic_path="${CAUSAL_CRITIC_PATH}" \
    algorithm.causal_critic_use_online_fallback=True \
    trainer.experiment_name=csrfaith_critic_7B \
    "$@"
