#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/env.local.sh" ]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi

CAUSAL_CRITIC_PATH=${CAUSAL_CRITIC_PATH:-}

bash "${SCRIPT_DIR}/csrfaith_smoke.sh" \
    algorithm.enable_causal_spatial_critic=True \
    algorithm.causal_critic_path="${CAUSAL_CRITIC_PATH}" \
    algorithm.causal_critic_use_online_fallback=True \
    "$@"
