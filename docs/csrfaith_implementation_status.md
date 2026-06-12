# CSR-Faith Implementation Status

## Implemented

- CSR target derivation from existing dataset fields, without modifying the original dataset:
  - `verl/utils/causal_rationale.py`
  - `scripts/build_csr_target_cache.py`
- Rationale scoring:
  - coverage
  - precision
  - compactness
  - sufficiency proxy
  - necessity proxy
  - weighted overall score
- Step-level causal intervention:
  - `verl/utils/step_causal.py`
  - relation flip
  - entity swap
  - step masking
  - prefix construction up to `<answer>`
  - normalized answer-change scoring
- Answer normalization:
  - `verl/utils/answer_normalization.py`
  - reused by sequence-level CFS and step-level CFS
- CSR-Faith training integration:
  - `algorithm.enable_csrfaith`
  - CSR rationale metrics
  - step CFS metrics
  - CSR GRPO advantage
  - CSR Lagrangian multipliers
  - `csrfaith_state.pt` checkpoint state
- Training and debugging scripts:
  - `scripts/csrfaith_7b_grpo.sh`
  - `scripts/csrfaith_smoke.sh`
  - `scripts/check_csrfaith_ready.py`
  - `scripts/debug_csr_batch.py`
  - `scripts/build_csr_target_cache.py`
- Causal Spatial Critic offline bridge pipeline:
  - `verl/utils/causal_critic_data.py`
  - `verl/models/causal_spatial_critic.py`
  - `scripts/build_causal_critic_dataset.py`
  - `scripts/train_causal_spatial_critic.py`
  - `scripts/evaluate_causal_spatial_critic.py`
  - `scripts/csrfaith_critic_smoke.sh`
  - `scripts/csrfaith_critic_7b_grpo.sh`
  - `scripts/env.common.sh`
  - `scripts/prepare_assets.sh`
  - JSONL schema with `curriculum_phase` and `reward_vector`
  - lightweight hashed logistic critic checkpoint at `critic.json`
  - optional trainer hook via `algorithm.enable_causal_spatial_critic`

## Paper-grade Extension Status

The current implementation now includes the offline Causal Spatial Critic
bridge pipeline. It follows an EVO-RAG-style staged organization:

1. build intervention-label examples
2. train a small causal reward model
3. evaluate the checkpoint independently
4. later consume critic scores inside CSR-GRPO

- paper idea: `docs/paper_idea_causal_spatial_critic.md`
- code plan: `docs/causal_spatial_critic_dev.md`
- proposed model: `CausalSpatialCritic(q, E, step, intervention_type) -> P(answer changes)`
- role: bridge expensive step-level intervention labels into dense CSR-GRPO rewards
- implemented: offline data, training, checkpoint, evaluation, optional trainer scoring hook, and tests
- pending: online refresh scheduler, GPU/Ray/vLLM verification, and paper-grade encoder option

This proposed critic is separate from the existing PPO value critic in
`verl/workers/critic/`.

## Verification Completed In This Environment

The current local environment does not have full training dependencies installed, but the lightweight checks below pass:

```bash
python3 -m unittest discover -s tests
python3 -m py_compile \
  scripts/csr_debug_io.py \
  scripts/check_csrfaith_ready.py \
  scripts/debug_csr_batch.py \
  scripts/build_csr_target_cache.py \
  scripts/build_causal_critic_dataset.py \
  scripts/train_causal_spatial_critic.py \
  scripts/evaluate_causal_spatial_critic.py \
  scripts/model_merger.py \
  tests/test_check_csrfaith_ready.py \
  tests/test_build_csr_target_cache.py \
  tests/test_causal_critic_data.py \
  tests/test_causal_critic_scripts.py \
  tests/test_causal_spatial_critic_static.py \
  tests/test_counterfactual.py \
  tests/test_dataset.py \
  tests/test_debug_csr_batch.py \
  tests/test_model_merger_static.py \
  tests/test_protocol.py \
  tests/test_reward_manager.py \
  tests/test_reviewer.py \
  tests/test_causal_rationale.py \
  tests/test_step_causal.py \
  tests/test_training_scripts.py \
  tests/test_trainer_static.py \
  tests/test_csr_advantage.py \
  verl/utils/answer_normalization.py \
  verl/utils/causal_critic_data.py \
  verl/utils/causal_rationale.py \
  verl/utils/step_causal.py \
  verl/utils/counterfactual.py \
  verl/models/causal_spatial_critic.py \
  verl/utils/reviewer.py \
  verl/utils/dataset.py \
  verl/protocol.py \
  verl/workers/reward/custom.py \
  verl/trainer/config.py \
  verl/trainer/core_algos.py \
  verl/trainer/ray_trainer.py
bash -n scripts/*.sh scripts/extras/*.sh
python3 scripts/check_csrfaith_ready.py --no-fail
python3 scripts/debug_csr_batch.py --problem "..." --ground-truth "..." --response "..." --jsonl
python3 scripts/build_csr_target_cache.py --input-json sample.json --output targets.jsonl
```

Current lightweight test count:

```text
87 tests, 13 skipped
```

The skipped tests require training/runtime dependencies such as `numpy`, `torch`, `datasets`, and `transformers`, which are not installed in this local Python environment.

## Verification Still Required In A Full Training Environment

Run the smoke script after installing the normal project dependencies and using a GPU/Ray/vLLM-capable environment:

```bash
bash scripts/csrfaith_smoke.sh
```

Expected minimum pass criteria:

- training starts without OmegaConf key errors
- CSR metrics appear in console logs
- `csr/rationale_*` metrics are finite
- `csr/step_interventions_mean` is greater than zero for well-formed rollouts
- no `DataProto` union conflict
- no shape mismatch after `_balance_batch`
- no failure in `actor_rollout_wg.generate_continuations`

Then run a short checkpoint test:

```bash
bash scripts/csrfaith_smoke.sh \
  trainer.save_freq=1 \
  trainer.save_limit=3 \
  trainer.save_checkpoint_path=ckpts/csrfaith_smoke
```

Expected checkpoint files:

```text
ckpts/csrfaith_smoke/global_step_1/actor/
ckpts/csrfaith_smoke/global_step_1/dataloader.pt
ckpts/csrfaith_smoke/global_step_1/csrfaith_state.pt
```

## Known Limits

- Step-level CFS uses deterministic text interventions, not a learned graph verifier.
- Necessity and sufficiency are Phase 1/2 proxies; paper-grade verifier filtering is still a future upgrade.
- If generated responses omit `<think>` or `<answer>`, step CFS is invalid and defaults to no modulation.
- If GT scene graph cannot be parsed from `ground_truth` or `problem`, CSR target is empty and only task reward remains active for that sample.
- Full training has not been executed in this local environment because `torch`, `numpy`, `omegaconf`, `ray`, and `tensordict` are missing.
