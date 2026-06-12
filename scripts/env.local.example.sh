# ──────────────────────────────────────────────────────────────
# 机器专属本地配置模板（Machine-local config TEMPLATE）
#
# 用法：复制为 scripts/env.local.sh 后按你的机器修改。
#   cp scripts/env.local.example.sh scripts/env.local.sh
#
# 训练脚本（csrfaith_7b_grpo.sh / csrfaith_smoke.sh）启动时会自动 source
# 同目录下的 env.local.sh（如果存在）。该文件已被 .gitignore，不会进仓库，
# 因此每台机器维护自己的一份，互不影响，也不污染公共代码。
# ──────────────────────────────────────────────────────────────

# 数据 / 权重 / checkpoint 根目录（强烈建议指向大盘，与代码分离）
export DATA_ROOT="${HOME}/csr_faith_assets"

# 使用哪几张 GPU，以及卡数（两者卡数必须一致）
export CUDA_VISIBLE_DEVICES="0,1"
export N_GPUS=2

# wandb 模式：offline（默认，纯本地）/ online（需 wandb login）/ disabled
export WANDB_MODE=offline

# HuggingFace 下载源：国内集群若无法直连 huggingface.co，取消下一行注释启用镜像
# export HF_ENDPOINT="https://hf-mirror.com"

# （可选）模型 / 数据集改用本地绝对路径而非自动下载
# export MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
# export DATA_FILE="/path/to/STVQA-7K"

# （可选）Causal Spatial Critic checkpoint，用于 csrfaith_critic_*.sh
# export CAUSAL_CRITIC_PATH="${DATA_ROOT}/ckpts/causal_spatial_critic"
