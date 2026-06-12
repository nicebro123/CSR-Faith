set -x

# 加载机器专属本地配置（GPU / 路径 / 镜像等），该文件不入库。见 scripts/env.local.example.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.common.sh"

# ── 以下全部可被环境变量 / env.local.sh 覆盖；下面是通用默认值 ──

# GPU：用哪几张卡 + 卡数（务必让 N_GPUS 与 CUDA_VISIBLE_DEVICES 的卡数一致）
# 默认两张卡（本机为物理卡 1、2）；其它机器在 env.local.sh 或 CLI 覆盖即可。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}
N_GPUS=${N_GPUS:-2}

python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files="${DATA_FILE}@train" \
    data.val_files="${DATA_FILE}@val" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function=spatial_sgg \
    trainer.experiment_name=csrfaith_7B \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.save_checkpoint_path=${CKPT_ROOT}/csrfaith_7B \
    trainer.save_freq=25 \
    trainer.save_limit=3 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=6 \
    trainer.max_steps=75 \
    trainer.total_episodes=75 \
    data.answer_key="answer" \
    data.image_key="images" \
    data.val_batch_size=8 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    worker.rollout.max_num_batched_tokens=8192 \
    algorithm.enable_citfaith=False \
    algorithm.enable_csrfaith=True \
    algorithm.csr_target_max_relations=4 \
    algorithm.csr_target_max_objects=6 \
    algorithm.csr_coverage_weight=0.4 \
    algorithm.csr_precision_weight=0.2 \
    algorithm.csr_compactness_weight=0.1 \
    algorithm.csr_sufficiency_weight=0.2 \
    algorithm.csr_necessity_weight=0.1 \
    algorithm.csr_step_cfs_alpha=0.1 \
    algorithm.csr_max_steps=6 \
    algorithm.csr_max_step_interventions=1 \
    algorithm.tau_coverage=0.7 \
    algorithm.tau_step_cfs=0.5 \
    algorithm.dual_lr=0.01 \
    "$@"
