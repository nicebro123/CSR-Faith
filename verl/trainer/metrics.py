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

from typing import Any, Dict, List

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def _safe_tensor_stat(values: torch.Tensor, stat: str, default: float = 0.0) -> float:
    if values.numel() == 0:
        return default
    if stat == "mean":
        result = torch.mean(values)
    elif stat == "max":
        result = torch.max(values)
    elif stat == "min":
        result = torch.min(values)
    else:
        raise ValueError(f"Unknown tensor stat: {stat}")
    return result.detach().item()


def _safe_tensor_var(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    return torch.var(values, unbiased=False)


def _safe_divisor(value: float) -> float:
    return value if value > 0 else 1.0


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = _safe_tensor_var(valid_returns - valid_values)
        return_var = _safe_tensor_var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": _safe_tensor_stat(sequence_score, "mean"),
        "critic/score/max": _safe_tensor_stat(sequence_score, "max"),
        "critic/score/min": _safe_tensor_stat(sequence_score, "min"),
        # reward
        "critic/rewards/mean": _safe_tensor_stat(sequence_reward, "mean"),
        "critic/rewards/max": _safe_tensor_stat(sequence_reward, "max"),
        "critic/rewards/min": _safe_tensor_stat(sequence_reward, "min"),
        # adv
        "critic/advantages/mean": _safe_tensor_stat(valid_adv, "mean"),
        "critic/advantages/max": _safe_tensor_stat(valid_adv, "max"),
        "critic/advantages/min": _safe_tensor_stat(valid_adv, "min"),
        # returns
        "critic/returns/mean": _safe_tensor_stat(valid_returns, "mean"),
        "critic/returns/max": _safe_tensor_stat(valid_returns, "max"),
        "critic/returns/min": _safe_tensor_stat(valid_returns, "min"),
        **(
            {
                # values
                "critic/values/mean": _safe_tensor_stat(valid_values, "mean"),
                "critic/values/max": _safe_tensor_stat(valid_values, "max"),
                "critic/values/min": _safe_tensor_stat(valid_values, "min"),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / _safe_divisor(num_tokens_of_section[name])
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / _safe_divisor(time * n_gpus),
    }
