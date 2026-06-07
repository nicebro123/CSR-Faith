set -x

# 加载机器专属本地配置（与正式训练脚本一致），该文件不入库。见 scripts/env.local.example.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/env.local.sh" ]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi

# 代码与数据/权重分离：下载缓存落到 DATA_ROOT（建议指向大盘）。
DATA_ROOT=${DATA_ROOT:-${HOME}/csr_faith_assets}
export HF_HOME=${HF_HOME:-${DATA_ROOT}/hf_cache}
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
DATA_FILE=${DATA_FILE:-hunarbatra/STVQA-7K}
# smoke 默认单卡；可被 env.local.sh / 外部 N_GPUS 覆盖
N_GPUS=${N_GPUS:-1}

python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files="${DATA_FILE}@train" \
    data.val_files="${DATA_FILE}@val" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function=spatial_sgg \
    trainer.experiment_name=csrfaith_smoke \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.max_steps=2 \
    trainer.total_episodes=1 \
    trainer.val_before_train=False \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.save_limit=3 \
    trainer.logger='["console"]' \
    data.rollout_batch_size=2 \
    data.val_batch_size=2 \
    data.answer_key="answer" \
    data.image_key="images" \
    data.max_prompt_length=6144 \
    data.max_response_length=512 \
    worker.actor.global_batch_size=2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.n=2 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=6656 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    algorithm.disable_kl=True \
    algorithm.use_kl_loss=False \
    algorithm.enable_citfaith=False \
    algorithm.enable_csrfaith=True \
    algorithm.csr_target_max_relations=4 \
    algorithm.csr_target_max_objects=6 \
    algorithm.csr_max_steps=2 \
    algorithm.csr_max_step_interventions=1 \
    algorithm.csr_step_cfs_alpha=0.1 \
    algorithm.tau_coverage=0.7 \
    algorithm.tau_step_cfs=0.5 \
    algorithm.dual_lr=0.01 \
    "$@"
