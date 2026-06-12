set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
# MODEL_PATH="../csr_faith_assets/local_models/Qwen2.5-VL-7B-Instruct"

# FORMAT_PROMPT="""<image> You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
#  The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put within <answer> </answer> tags, and only return the final answer text within the answer tags (without the option letter), e.g., <answer> {correct_answer} </answer>.
 
#  Q. """

# SPATIAL THINKER DATASETS
DATA_FILE=${DATA_FILE:-hunarbatra/STVQA-7K}
N_GPUS=${N_GPUS:-4}
# DATA_FILE="../csr_faith_assets/local_datasets/STVQA-7K/data"


python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    data.train_files="${DATA_FILE}@train" \
    data.val_files="${DATA_FILE}@val" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function=spatial_sgg \
    trainer.experiment_name=spatialthinker10k_7B \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.save_checkpoint_path=ckpts/spatialthinker10k_7B \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=8 \
    trainer.max_steps=75 \
    trainer.total_episodes=75 \
    data.answer_key="answer" \
    data.image_key="images" \
    data.val_batch_size=8 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    worker.rollout.max_num_batched_tokens=8192 \
    "$@"
    # data.format_prompt="${FORMAT_PROMPT}" 
    # data.max_prompt_length=8192 \
    # data.max_response_length=2048 \
    # worker.rollout.max_num_batched_tokens=10240 \
    # data.text_only=True 
    # algorithm.disable_kl=True
    # algorithm.kl_penalty=chi2
    # algorithm.kl_coef=3.0e-3 
    # worker.actor.model.freeze_vision_tower=True 
    # worker.rollout.tensor_parallel_size=1 
    
