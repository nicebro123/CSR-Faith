# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

# CIT-Faith and CSR-Faith imports
from ..utils.reviewer import ReviewerModel, extract_answer_text
from ..utils.counterfactual import compute_cfs_batch_indexed
from ..utils.causal_critic_data import build_critic_examples
from ..utils.causal_rationale import build_causal_rationale_target, extract_gt_scene_and_answer, score_rationale
from ..utils.step_causal import (
    build_prefixes_for_step_interventions,
    compute_step_causal_score,
    generate_step_interventions,
)
from ..models.causal_spatial_critic import CausalSpatialCritic


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


def _normalize_advantage_estimator(adv_estimator: Any) -> AdvantageEstimator:
    if isinstance(adv_estimator, AdvantageEstimator):
        return adv_estimator
    try:
        return AdvantageEstimator(str(adv_estimator))
    except ValueError as exc:
        raise NotImplementedError(f"Unknown advantage estimator: {adv_estimator}.") from exc


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    if "ref_log_probs" in data.batch.keys():
        kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
        kld = kld * response_mask  # (batch_size, response_length)
    else:
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    # CIT-Faith parameters (optional)
    citfaith_enabled: bool = False,
    sc_scores: Optional[torch.Tensor] = None,
    pr_scores: Optional[torch.Tensor] = None,
    cfs_scores: Optional[torch.Tensor] = None,
    lambda_sc: float = 0.0,
    lambda_pr: float = 0.0,
    cfs_alpha: float = 0.1,
    # CSR-Faith parameters (optional)
    csrfaith_enabled: bool = False,
    rationale_scores: Optional[torch.Tensor] = None,
    step_cfs_scores: Optional[torch.Tensor] = None,
    lambda_coverage: float = 0.0,
    lambda_step_cfs: float = 0.0,
    csr_step_cfs_alpha: float = 0.1,
):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]

    if csrfaith_enabled and adv_estimator == AdvantageEstimator.GRPO and \
            rationale_scores is not None and step_cfs_scores is not None:
        # Use CSR-Faith advantage with derived rationale signals.
        advantages, returns = core_algos.compute_csrfaith_grpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            rationale_scores=rationale_scores,
            step_cfs_scores=step_cfs_scores,
            lambda_coverage=lambda_coverage,
            lambda_step_cfs=lambda_step_cfs,
            alpha=csr_step_cfs_alpha,
        )
    elif citfaith_enabled and adv_estimator == AdvantageEstimator.GRPO and \
            sc_scores is not None and pr_scores is not None and cfs_scores is not None:
        # Use CIT-Faith advantage with dual constraint + causal modulation
        advantages, returns = core_algos.compute_citfaith_grpo_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            sc_scores=sc_scores,
            pr_scores=pr_scores,
            cfs_scores=cfs_scores,
            lambda_sc=lambda_sc,
            lambda_pr=lambda_pr,
            alpha=cfs_alpha,
        )
    elif adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class _PrefixContinuationHelper:
    """Driver-side helper for policy prefix-continuation decode."""

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        prompt_ids_all,
        prompt_attn_all,
        cf_max_tokens: int,
        log_prefix: str,
        multi_modal_data_all=None,
        max_prefix_tokens: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.wg = actor_rollout_wg
        self.prompt_ids_all = prompt_ids_all
        self.prompt_attn_all = prompt_attn_all
        self.cf_max_tokens = cf_max_tokens
        self.log_prefix = log_prefix
        self.multi_modal_data_all = multi_modal_data_all
        self.max_prefix_tokens = max_prefix_tokens
        self.current_idx = 0

    def set_rollout_index(self, idx: int):
        self.current_idx = idx

    @staticmethod
    def _select_continuation_output(cf_output_raw: Any, expected_size: int) -> DataProto:
        if isinstance(cf_output_raw, DataProto):
            return cf_output_raw

        if isinstance(cf_output_raw, (list, tuple)):
            fallback = None
            for output in cf_output_raw:
                if not isinstance(output, DataProto):
                    continue
                if fallback is None:
                    fallback = output
                if "continuation_ids" in output.batch and output.batch["continuation_ids"].shape[0] == expected_size:
                    return output
            if fallback is not None:
                return fallback

        raise TypeError(f"Expected DataProto continuation output, got {type(cf_output_raw).__name__}.")

    def __call__(self, prefix_texts: List[str]):
        if not prefix_texts:
            return []

        idx = self.current_idx
        prompt_attn = self.prompt_attn_all[idx]
        valid_prompt_len = prompt_attn.sum().int().item()
        if valid_prompt_len <= 0:
            print(f"[{self.log_prefix}] Empty prompt prefix for rollout index {idx}.")
            return [None] * len(prefix_texts)
        prompt_ids = self.prompt_ids_all[idx, -valid_prompt_len:]

        all_prefix_ids = []
        prompt_token_ids = prompt_ids.tolist()
        for text in prefix_texts:
            resp_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if self.max_prefix_tokens is not None and len(prompt_token_ids) + len(resp_tokens) > self.max_prefix_tokens:
                max_resp_tokens = self.max_prefix_tokens - len(prompt_token_ids)
                if max_resp_tokens > 0:
                    resp_tokens = resp_tokens[-max_resp_tokens:]
                    full_ids = prompt_token_ids + resp_tokens
                else:
                    full_ids = prompt_token_ids[-self.max_prefix_tokens:]
            else:
                full_ids = prompt_token_ids + resp_tokens
            all_prefix_ids.append(full_ids)

        max_len = max(len(ids) for ids in all_prefix_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        n = len(all_prefix_ids)
        padded = torch.full((n, max_len), pad_id, dtype=torch.long)
        masks = torch.zeros(n, max_len, dtype=torch.long)
        for i, ids in enumerate(all_prefix_ids):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            masks[i, :len(ids)] = 1

        from tensordict import TensorDict

        non_tensors = {}
        if self.multi_modal_data_all is not None:
            multi_modal_data = self.multi_modal_data_all[idx]
            if multi_modal_data is not None:
                non_tensors["multi_modal_data"] = np.array([multi_modal_data for _ in range(n)], dtype=object)

        cf_batch = DataProto(
            batch=TensorDict({
                "prefix_ids": padded,
                "prefix_mask": masks,
            }, batch_size=n),
            non_tensor_batch=non_tensors,
            meta_info={"cf_max_tokens": self.cf_max_tokens},
        )

        try:
            cf_output_raw = self.wg.generate_continuations(cf_batch)
            cf_output = self._select_continuation_output(cf_output_raw, expected_size=len(prefix_texts))
            cont_ids = cf_output.batch["continuation_ids"]
            cont_mask = cf_output.batch["continuation_mask"]

            results = []
            for i in range(cont_ids.shape[0]):
                valid_len = cont_mask[i].sum().int().item()
                if valid_len > 0:
                    text = self.tokenizer.decode(
                        cont_ids[i, :valid_len],
                        skip_special_tokens=True,
                    ).strip()
                    results.append(text)
                else:
                    results.append(None)
            return results
        except Exception as e:
            print(f"[{self.log_prefix}] Continuation decode failed: {e}")
            return [None] * len(prefix_texts)


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
        val_reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.adv_estimator = _normalize_advantage_estimator(config.algorithm.adv_estimator)

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if self.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            self.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        # ── CIT-Faith: Initialize reviewer model and Lagrangian state ──
        self.citfaith_enabled = config.algorithm.enable_citfaith
        if self.citfaith_enabled:
            print("[CIT-Faith] Initializing faithfulness optimization framework...")
            self.reviewer = ReviewerModel(
                model_name_or_path=config.algorithm.reviewer_model_path,
                gpu_memory_utilization=config.algorithm.reviewer_gpu_memory,
            )
            # Lagrangian multipliers (initialized to 0, updated via dual ascent)
            self.lambda_sc = 0.0
            self.lambda_pr = 0.0
            print(f"[CIT-Faith] Config: tau_sc={config.algorithm.tau_sc}, "
                  f"tau_pr={config.algorithm.tau_pr}, alpha={config.algorithm.cfs_alpha}, "
                  f"dual_lr={config.algorithm.dual_lr}")
        else:
            self.reviewer = None
            self.lambda_sc = 0.0
            self.lambda_pr = 0.0

        # ── CSR-Faith: Initialize causal spatial rationale state ──
        self.csrfaith_enabled = config.algorithm.enable_csrfaith
        self.lambda_coverage = config.algorithm.lambda_coverage_init
        self.lambda_step_cfs = config.algorithm.lambda_step_cfs_init
        self.causal_spatial_critic = None
        if self.csrfaith_enabled:
            print("[CSR-Faith] Initializing causal spatial rationale optimization...")
            print(f"[CSR-Faith] Config: tau_coverage={config.algorithm.tau_coverage}, "
                  f"tau_step_cfs={config.algorithm.tau_step_cfs}, "
                  f"alpha={config.algorithm.csr_step_cfs_alpha}, "
                  f"dual_lr={config.algorithm.dual_lr}")
            if config.algorithm.enable_causal_spatial_critic:
                critic_path = config.algorithm.causal_critic_path
                if critic_path:
                    try:
                        self.causal_spatial_critic = CausalSpatialCritic.load(critic_path)
                        print(f"[CSR-Faith] Loaded Causal Spatial Critic from {critic_path}")
                    except Exception as exc:
                        if not config.algorithm.causal_critic_use_online_fallback:
                            raise
                        print(
                            "[CSR-Faith] WARNING: Failed to load Causal Spatial Critic; "
                            f"falling back to online step-CFS. Error: {exc}"
                        )
                elif not config.algorithm.causal_critic_use_online_fallback:
                    raise ValueError(
                        "algorithm.enable_causal_spatial_critic=True requires "
                        "algorithm.causal_critic_path unless online fallback is enabled."
                    )
                else:
                    print(
                        "[CSR-Faith] WARNING: Causal Spatial Critic enabled without "
                        "causal_critic_path; falling back to online step-CFS."
                    )

        self._create_dataloader()

    def _validation_enabled(self) -> bool:
        return bool(
            self.val_reward_fn is not None
            and (
                self.config.trainer.val_before_train
                or self.config.trainer.val_freq > 0
                or self.config.trainer.val_only
            )
        )

    def _create_dataloader(self) -> None:
        self.train_dataset = RLHFDataset(
            data_path=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            mixed_data=self.config.data.mixed_data,
            text_only=self.config.data.text_only,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            format_prompt=self.config.data.format_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        self.val_dataset = None
        self.val_dataloader = None
        if self._validation_enabled():
            self.val_dataset = RLHFDataset(
                data_path=self.config.data.val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                answer_key=self.config.data.answer_key,
                image_key=self.config.data.image_key,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation="right",
                format_prompt=self.config.data.format_prompt,
                min_pixels=self.config.data.min_pixels,
                max_pixels=self.config.data.max_pixels,
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=len(self.val_dataset)
                if self.config.data.val_batch_size == -1
                else self.config.data.val_batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=False,
            )

        assert len(self.train_dataloader) >= 1
        if self.val_dataloader is not None:
            assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        if self.val_dataloader is not None:
            print(f"Size of val dataloader: {len(self.val_dataloader)}")
        else:
            print("Validation dataloader disabled.")

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        if self.val_dataloader is None:
            raise RuntimeError("Validation dataloader is disabled. Set val_before_train=True or val_freq>0.")

        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                print(f"[DEBUG] multi_modal_inputs")
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                print(f"[DEBUG] text_only_inputs")
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            if len(test_output_gen_batch) != len(test_batch):
                if len(test_batch) == 0 or len(test_output_gen_batch) % len(test_batch) != 0:
                    raise ValueError(
                        "Validation rollout output batch size must equal the input batch size or be an integer "
                        f"multiple. Got output={len(test_output_gen_batch)}, input={len(test_batch)}."
                    )
                val_repeat_times = len(test_output_gen_batch) // len(test_batch)
                test_batch = test_batch.repeat(repeat_times=val_repeat_times, interleave=True)
                input_texts = [
                    text for text in input_texts for _ in range(val_repeat_times)
                ]
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        os.makedirs(folder_path, exist_ok=True)
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        # CIT-Faith: save Lagrangian multiplier state
        if self.citfaith_enabled:
            citfaith_state = {
                "lambda_sc": self.lambda_sc,
                "lambda_pr": self.lambda_pr,
            }
            citfaith_path = os.path.join(folder_path, "citfaith_state.pt")
            torch.save(citfaith_state, citfaith_path)

        # CSR-Faith: save rationale multiplier state
        if self.csrfaith_enabled:
            csrfaith_state = {
                "lambda_coverage": self.lambda_coverage,
                "lambda_step_cfs": self.lambda_step_cfs,
            }
            csrfaith_path = os.path.join(folder_path, "csrfaith_state.pt")
            torch.save(csrfaith_state, csrfaith_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

        # CIT-Faith: restore Lagrangian multiplier state
        if self.citfaith_enabled:
            citfaith_path = os.path.join(self.config.trainer.load_checkpoint_path, "citfaith_state.pt")
            if os.path.exists(citfaith_path):
                citfaith_state = torch.load(citfaith_path, weights_only=False)
                self.lambda_sc = citfaith_state.get("lambda_sc", 0.0)
                self.lambda_pr = citfaith_state.get("lambda_pr", 0.0)
                print(f"[CIT-Faith] Restored lambda_sc={self.lambda_sc:.4f}, lambda_pr={self.lambda_pr:.4f}")
            else:
                print("[CIT-Faith] No Lagrangian state found, starting from lambda=0.")

        # CSR-Faith: restore rationale multiplier state
        if self.csrfaith_enabled:
            csrfaith_path = os.path.join(self.config.trainer.load_checkpoint_path, "csrfaith_state.pt")
            if os.path.exists(csrfaith_path):
                csrfaith_state = torch.load(csrfaith_path, weights_only=False)
                self.lambda_coverage = csrfaith_state.get("lambda_coverage", self.lambda_coverage)
                self.lambda_step_cfs = csrfaith_state.get("lambda_step_cfs", self.lambda_step_cfs)
                print(f"[CSR-Faith] Restored lambda_coverage={self.lambda_coverage:.4f}, "
                      f"lambda_step_cfs={self.lambda_step_cfs:.4f}")
            else:
                print("[CSR-Faith] No rationale state found, starting from configured lambda values.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and (self.config.trainer.val_before_train or self.config.trainer.val_only):
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for _ in tqdm(range(self.config.trainer.total_episodes), desc="Episode", position=0):
            for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                if self.global_step >= self.training_steps:
                    break
                self.global_step += 1

                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                continuation_multi_modal_data = None
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                    continuation_multi_modal_data = gen_batch.non_tensor_batch.get("multi_modal_data")
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["temperature"] = 0
                            gen_baseline_batch.meta_info["n"] = 1
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, _ = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    if continuation_multi_modal_data is not None:
                        batch.non_tensor_batch["_continuation_multi_modal_data"] = np.repeat(
                            continuation_multi_modal_data,
                            self.config.worker.rollout.n,
                            axis=0,
                        )
                    batch = batch.union(gen_batch_output)

                    # compute reward
                    with _timer("reward", timing_raw):
                        if self.use_reward_model:
                            raise NotImplementedError("Reward model is not supported yet.")

                        # we combine with rule-based rm
                        reward_tensor, reward_metrics = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {
                            f"reward/{key}": value for key, value in reduce_metrics(reward_metrics).items()
                        }
                        metrics.update(reward_metrics)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        response_ids = batch.batch["responses"]
                        response_masks = batch.batch["response_mask"]
                        batch_size = response_ids.shape[0]
                        response_texts = None
                        max_continuation_prefix_tokens = max(
                            self.config.data.max_prompt_length
                            + self.config.data.max_response_length
                            - self.config.algorithm.cf_max_tokens,
                            1,
                        )

                        def _decode_response_texts():
                            decoded = []
                            for idx in range(batch_size):
                                valid_len = response_masks[idx].sum().int().item()
                                valid_ids = response_ids[idx, :valid_len]
                                text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                decoded.append(text)
                            return decoded

                        # ── CIT-Faith: Compute faithfulness scores ──
                        sc_scores_t, pr_scores_t, cfs_scores_t = None, None, None
                        if self.citfaith_enabled:
                            with _timer("citfaith_review", timing_raw):
                                response_texts = _decode_response_texts()

                                # 1. Reviewer evaluation: SC + PR
                                sc_scores_list, pr_scores_list = self.reviewer.evaluate_batch(response_texts)
                                sc_scores_t = torch.tensor(sc_scores_list, dtype=torch.float32)
                                pr_scores_t = torch.tensor(pr_scores_list, dtype=torch.float32)

                            with _timer("citfaith_cfs", timing_raw):
                                # 2. Counterfactual intervention: CFS
                                original_answers = [extract_answer_text(t) for t in response_texts]

                                # Get prompt ids for prefix-continuation (prompt + perturbed response)
                                prompt_ids_batch = batch.batch["prompts"]  # (bs, prompt_len)
                                prompt_attention = batch.batch["attention_mask"][:, :prompt_ids_batch.shape[1]]

                                cf_helper = _PrefixContinuationHelper(
                                    tokenizer=self.tokenizer,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    prompt_ids_all=prompt_ids_batch,
                                    prompt_attn_all=prompt_attention,
                                    cf_max_tokens=self.config.algorithm.cf_max_tokens,
                                    log_prefix="CIT-Faith",
                                    multi_modal_data_all=batch.non_tensor_batch.get("_continuation_multi_modal_data"),
                                    max_prefix_tokens=max_continuation_prefix_tokens,
                                )

                                # Use the indexed version of compute_cfs_batch
                                cfs_scores_list, cfs_type_metrics = compute_cfs_batch_indexed(
                                    response_texts=response_texts,
                                    original_answers=original_answers,
                                    continuation_helper=cf_helper,
                                )
                                cfs_scores_t = torch.tensor(cfs_scores_list, dtype=torch.float32)

                                # Log CFS validity
                                n_valid_cfs = sum(1 for s in cfs_scores_list if s >= 0)
                                if n_valid_cfs == 0 and self.global_step <= 1:
                                    print("[CIT-Faith] WARNING: All CFS scores are invalid (-1). "
                                          "Possible causes: all rollouts have format failures, "
                                          "or counterfactual interventions could not be generated. "
                                          "Causal modulation defaults to g=1 for these rollouts.")

                                # Log CIT-Faith metrics
                                valid_sc = [s for s in sc_scores_list]
                                valid_pr = [s for s in pr_scores_list]
                                valid_cfs = [s for s in cfs_scores_list if s >= 0]

                                metrics["citfaith/sc_mean"] = sum(valid_sc) / max(len(valid_sc), 1)
                                metrics["citfaith/pr_mean"] = sum(valid_pr) / max(len(valid_pr), 1)
                                metrics["citfaith/cfs_mean"] = sum(valid_cfs) / max(len(valid_cfs), 1) if valid_cfs else 0.0
                                metrics["citfaith/lambda_sc"] = self.lambda_sc
                                metrics["citfaith/lambda_pr"] = self.lambda_pr
                                metrics["citfaith/cfs_valid_ratio"] = len(valid_cfs) / max(batch_size, 1)

                        # ── CSR-Faith: Compute causal spatial rationale scores ──
                        rationale_scores_t, step_cfs_scores_t = None, None
                        csr_coverage_mean = -1.0
                        csr_step_cfs_mean = -1.0
                        if self.csrfaith_enabled:
                            with _timer("csr_rationale", timing_raw):
                                if response_texts is None:
                                    response_texts = _decode_response_texts()

                                ground_truths = batch.non_tensor_batch.get("ground_truth")
                                problems = batch.non_tensor_batch.get("problem")
                                if ground_truths is None:
                                    ground_truths = np.array([""] * batch_size, dtype=object)
                                if problems is None:
                                    problems = np.array([""] * batch_size, dtype=object)

                                targets, original_answers = [], []
                                target_confidences, coverages, valid_coverages, precisions = [], [], [], []
                                compactnesses, sufficiencies, necessities, overalls = [], [], [], []

                                for idx, response_text in enumerate(response_texts):
                                    gt_scene, gt_answer = extract_gt_scene_and_answer(str(ground_truths[idx]))
                                    if not gt_scene:
                                        problem_scene, _ = extract_gt_scene_and_answer(str(problems[idx]))
                                        gt_scene = problem_scene
                                    target = build_causal_rationale_target(
                                        problem=str(problems[idx]),
                                        gt_scene=gt_scene,
                                        gt_answer=gt_answer,
                                        max_relations=self.config.algorithm.csr_target_max_relations,
                                        max_objects=self.config.algorithm.csr_target_max_objects,
                                    )
                                    targets.append(target)
                                    original_answers.append(extract_answer_text(response_text))
                                    score = score_rationale(
                                        response_text=response_text,
                                        target=target,
                                        coverage_weight=self.config.algorithm.csr_coverage_weight,
                                        precision_weight=self.config.algorithm.csr_precision_weight,
                                        compactness_weight=self.config.algorithm.csr_compactness_weight,
                                        sufficiency_weight=self.config.algorithm.csr_sufficiency_weight,
                                        necessity_weight=self.config.algorithm.csr_necessity_weight,
                                    )
                                    target_confidences.append(target.confidence)
                                    coverages.append(score.coverage)
                                    if len(target.objects) + len(target.relations) > 0:
                                        valid_coverages.append(score.coverage)
                                    precisions.append(score.precision)
                                    compactnesses.append(score.compactness)
                                    sufficiencies.append(score.sufficiency)
                                    necessities.append(score.necessity)
                                    overalls.append(score.overall)

                                rationale_scores_t = torch.tensor(overalls, dtype=torch.float32)
                                csr_coverage_mean = (
                                    sum(valid_coverages) / len(valid_coverages) if valid_coverages else -1.0
                                )

                                metrics["csr/target_confidence_mean"] = (
                                    sum(target_confidences) / max(len(target_confidences), 1)
                                )
                                metrics["csr/rationale_coverage_mean"] = (
                                    csr_coverage_mean if csr_coverage_mean >= 0.0 else 0.0
                                )
                                metrics["csr/rationale_precision_mean"] = sum(precisions) / max(len(precisions), 1)
                                metrics["csr/rationale_compactness_mean"] = (
                                    sum(compactnesses) / max(len(compactnesses), 1)
                                )
                                metrics["csr/rationale_sufficiency_mean"] = (
                                    sum(sufficiencies) / max(len(sufficiencies), 1)
                                )
                                metrics["csr/rationale_necessity_mean"] = (
                                    sum(necessities) / max(len(necessities), 1)
                                )
                                metrics["csr/rationale_overall_mean"] = sum(overalls) / max(len(overalls), 1)

                            with _timer("csr_step_cfs", timing_raw):
                                step_cfs_scores = [-1.0] * batch_size
                                step_decode_valid_ratios = [0.0] * batch_size
                                step_max_scores = [0.0] * batch_size
                                step_intervention_counts = [0] * batch_size
                                causal_signal_is_critic = 0.0

                                def _score_with_causal_critic() -> bool:
                                    if self.causal_spatial_critic is None:
                                        return False
                                    any_scored = False
                                    for idx, response_text in enumerate(response_texts):
                                        if (
                                            targets[idx].confidence
                                            < self.config.algorithm.causal_critic_min_target_confidence
                                        ):
                                            continue
                                        interventions = generate_step_interventions(
                                            response_text=response_text,
                                            target=targets[idx],
                                            max_steps=self.config.algorithm.csr_max_steps,
                                            max_interventions_per_step=(
                                                self.config.algorithm.csr_max_step_interventions
                                            ),
                                            rollout_index=idx,
                                        )
                                        step_intervention_counts[idx] = len(interventions)
                                        if not interventions:
                                            continue

                                        critic_examples = build_critic_examples(
                                            problem=str(problems[idx]),
                                            target=targets[idx],
                                            response_text=response_text,
                                            interventions=interventions,
                                            counterfactual_answers=[],
                                            uid=f"global{self.global_step}:rollout{idx}",
                                            response_answer=original_answers[idx],
                                            policy_checkpoint=f"global_step_{self.global_step}",
                                            source="trainer_learned_critic",
                                        )
                                        critic_scores = self.causal_spatial_critic.score_batch(critic_examples)
                                        if not critic_scores:
                                            continue
                                        step_cfs_scores[idx] = sum(critic_scores) / len(critic_scores)
                                        step_decode_valid_ratios[idx] = 1.0
                                        step_max_scores[idx] = max(critic_scores)
                                        any_scored = True
                                    return any_scored

                                def _score_with_online_step_cfs() -> None:
                                    prompt_ids_batch = batch.batch["prompts"]
                                    prompt_attention = batch.batch["attention_mask"][:, :prompt_ids_batch.shape[1]]

                                    csr_cf_helper = _PrefixContinuationHelper(
                                        tokenizer=self.tokenizer,
                                        actor_rollout_wg=self.actor_rollout_wg,
                                        prompt_ids_all=prompt_ids_batch,
                                        prompt_attn_all=prompt_attention,
                                        cf_max_tokens=self.config.algorithm.cf_max_tokens,
                                        log_prefix="CSR-Faith",
                                        multi_modal_data_all=batch.non_tensor_batch.get("_continuation_multi_modal_data"),
                                        max_prefix_tokens=max_continuation_prefix_tokens,
                                    )

                                    for idx, response_text in enumerate(response_texts):
                                        if not original_answers[idx]:
                                            continue
                                        interventions = generate_step_interventions(
                                            response_text=response_text,
                                            target=targets[idx],
                                            max_steps=self.config.algorithm.csr_max_steps,
                                            max_interventions_per_step=(
                                                self.config.algorithm.csr_max_step_interventions
                                            ),
                                            rollout_index=idx,
                                        )
                                        step_intervention_counts[idx] = len(interventions)
                                        if not interventions:
                                            continue

                                        prefixes = build_prefixes_for_step_interventions(interventions)
                                        csr_cf_helper.set_rollout_index(idx)
                                        cf_answers = csr_cf_helper(prefixes)
                                        step_score = compute_step_causal_score(
                                            original_answer=original_answers[idx],
                                            interventions=interventions,
                                            counterfactual_answers=cf_answers,
                                        )
                                        step_cfs_scores[idx] = step_score.mean
                                        step_decode_valid_ratios[idx] = step_score.valid_ratio
                                        step_max_scores[idx] = step_score.max_score

                                if (
                                    self.config.algorithm.csr_max_steps > 0
                                    and self.config.algorithm.csr_max_step_interventions > 0
                                ):
                                    if self.config.algorithm.enable_causal_spatial_critic:
                                        try:
                                            causal_signal_is_critic = 1.0 if _score_with_causal_critic() else 0.0
                                        except Exception as exc:
                                            if not self.config.algorithm.causal_critic_use_online_fallback:
                                                raise
                                            print(
                                                "[CSR-Faith] WARNING: Causal Spatial Critic scoring failed; "
                                                f"falling back to online step-CFS. Error: {exc}"
                                            )
                                            causal_signal_is_critic = 0.0
                                            step_cfs_scores[:] = [-1.0] * batch_size
                                            step_decode_valid_ratios[:] = [0.0] * batch_size
                                            step_max_scores[:] = [0.0] * batch_size
                                            step_intervention_counts[:] = [0] * batch_size
                                            _score_with_online_step_cfs()
                                        else:
                                            if causal_signal_is_critic == 0.0:
                                                _score_with_online_step_cfs()
                                    else:
                                        _score_with_online_step_cfs()

                                step_cfs_scores_t = torch.tensor(step_cfs_scores, dtype=torch.float32)
                                valid_step_cfs = [score for score in step_cfs_scores if score >= 0]
                                csr_step_cfs_mean = (
                                    sum(valid_step_cfs) / len(valid_step_cfs) if valid_step_cfs else -1.0
                                )

                                metrics["csr/step_cfs_mean"] = (
                                    csr_step_cfs_mean if csr_step_cfs_mean >= 0.0 else 0.0
                                )
                                metrics["csr/step_cfs_valid_ratio"] = len(valid_step_cfs) / max(batch_size, 1)
                                metrics["csr/step_cfs_decode_valid_ratio"] = (
                                    sum(step_decode_valid_ratios) / max(len(step_decode_valid_ratios), 1)
                                )
                                metrics["csr/step_cfs_max_mean"] = (
                                    sum(step_max_scores) / max(len(step_max_scores), 1)
                                )
                                metrics["csr/step_interventions_mean"] = (
                                    sum(step_intervention_counts) / max(len(step_intervention_counts), 1)
                                )
                                metrics["csr/causal_signal_is_critic"] = causal_signal_is_critic
                                metrics["csr/critic_enabled"] = (
                                    1.0 if self.config.algorithm.enable_causal_spatial_critic else 0.0
                                )
                                metrics["csr/critic_causal_mean"] = (
                                    csr_step_cfs_mean if causal_signal_is_critic and csr_step_cfs_mean >= 0.0 else 0.0
                                )
                                metrics["csr/critic_valid_ratio"] = (
                                    len(valid_step_cfs) / max(batch_size, 1) if causal_signal_is_critic else 0.0
                                )
                                metrics["csr/lambda_coverage"] = self.lambda_coverage
                                metrics["csr/lambda_step_cfs"] = self.lambda_step_cfs

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            # CIT-Faith parameters
                            citfaith_enabled=self.citfaith_enabled,
                            sc_scores=sc_scores_t,
                            pr_scores=pr_scores_t,
                            cfs_scores=cfs_scores_t,
                            lambda_sc=self.lambda_sc,
                            lambda_pr=self.lambda_pr,
                            cfs_alpha=self.config.algorithm.cfs_alpha,
                            # CSR-Faith parameters
                            csrfaith_enabled=self.csrfaith_enabled,
                            rationale_scores=rationale_scores_t,
                            step_cfs_scores=step_cfs_scores_t,
                            lambda_coverage=self.lambda_coverage,
                            lambda_step_cfs=self.lambda_step_cfs,
                            csr_step_cfs_alpha=self.config.algorithm.csr_step_cfs_alpha,
                        )

                        # ── CIT-Faith: Update Lagrangian multipliers (dual ascent) ──
                        if self.citfaith_enabled and sc_scores_t is not None:
                            self.lambda_sc, self.lambda_pr = core_algos.update_lagrangian_multipliers(
                                lambda_sc=self.lambda_sc,
                                lambda_pr=self.lambda_pr,
                                batch_sc_mean=sc_scores_t.mean().item(),
                                batch_pr_mean=pr_scores_t.mean().item(),
                                tau_sc=self.config.algorithm.tau_sc,
                                tau_pr=self.config.algorithm.tau_pr,
                                eta=self.config.algorithm.dual_lr,
                            )

                        # ── CSR-Faith: Update rationale multipliers (dual ascent) ──
                        if self.csrfaith_enabled and rationale_scores_t is not None:
                            self.lambda_coverage, self.lambda_step_cfs = core_algos.update_csr_lagrangian_multipliers(
                                lambda_coverage=self.lambda_coverage,
                                lambda_step_cfs=self.lambda_step_cfs,
                                batch_coverage_mean=csr_coverage_mean,
                                batch_step_cfs_mean=csr_step_cfs_mean,
                                tau_coverage=self.config.algorithm.tau_coverage,
                                tau_step_cfs=self.config.algorithm.tau_step_cfs,
                                eta=self.config.algorithm.dual_lr,
                            )

                        batch.non_tensor_batch.pop("_continuation_multi_modal_data", None)

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with _timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                self.logger.log(data=metrics, step=self.global_step)

            if self.global_step >= self.training_steps:
                break

        # perform validation after training if validation is enabled
        if self.val_reward_fn is not None and self.config.trainer.val_freq > 0:
            if (
                val_metrics is None
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
