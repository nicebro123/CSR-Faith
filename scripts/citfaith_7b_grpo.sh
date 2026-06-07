set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}

# SPATIAL THINKER DATASETS
DATA_FILE=${DATA_FILE:-hunarbatra/STVQA-7K}

# Reviewer model for CIT-Faith (frozen, AWQ quantized)
REVIEWER_MODEL=${REVIEWER_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}
N_GPUS=${N_GPUS:-4}

python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files="${DATA_FILE}@train" \
    data.val_files="${DATA_FILE}@val" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function=spatial_sgg \
    trainer.experiment_name=citfaith_7B \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.save_checkpoint_path=ckpts/citfaith_7B \
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
    algorithm.enable_citfaith=True \
    algorithm.reviewer_model_path=${REVIEWER_MODEL} \
    algorithm.reviewer_gpu_memory=0.15 \
    algorithm.tau_sc=0.8 \
    algorithm.tau_pr=0.7 \
    algorithm.dual_lr=0.01 \
    algorithm.cfs_alpha=0.1 \
    algorithm.cf_max_tokens=50 \
    "$@"
