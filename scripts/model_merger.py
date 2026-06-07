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

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def _normalize_actor_dir(local_dir: str) -> str:
    local_dir = os.path.normpath(os.path.abspath(local_dir))
    if os.path.basename(local_dir) == "huggingface":
        raise ValueError("The local_dir should be the actor checkpoint directory, not its huggingface subdirectory.")
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Actor checkpoint directory does not exist: {local_dir}")
    return local_dir


def _discover_world_size(local_dir: str) -> int:
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            return int(match.group(1))
    raise FileNotFoundError(f"No model_world_size_*_rank_0.pt shard found in {local_dir}")


def _load_shard(local_dir: str, world_size: int, rank: int):
    model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model shard: {model_path}")
    return torch.load(model_path, map_location="cpu", weights_only=False)


def _select_auto_model(config):
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        raise ValueError("Config has no architectures field; cannot select AutoModel class.")
    architecture = architectures[0]
    if "ForTokenClassification" in architecture:
        return AutoModelForTokenClassification
    if "ForCausalLM" in architecture:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in architecture:
        return AutoModelForVision2Seq
    raise NotImplementedError(f"Unknown architecture {architectures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()

    local_dir = _normalize_actor_dir(args.local_dir)

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = _discover_world_size(local_dir)

    state_dict = _load_shard(local_dir, world_size, rank)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if not isinstance(weight, torch.distributed._tensor.DTensor):
        raise TypeError(f"Expected DTensor shard values, got {type(weight)} for key {pivot_key}.")
    # get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    if mesh_dim_names not in (("fsdp",),):
        raise NotImplementedError(f"Unsupported mesh_dim_names {mesh_dim_names}")

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing model shards with {total_shards} {mesh_shape} in total")

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        state_dict = _load_shard(local_dir, world_size, rank)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
        futures = [executor.submit(process_one_shard, rank) for rank in range(1, total_shards)]
        for future in futures:
            future.result()
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                raise KeyError(f"Model shard is missing key {key}") from None
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == "dp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            if len(placements) != 1:
                raise ValueError(f"Expected one placement for FSDP key {key}, got {placements}")
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    print("Writing to local disk")
    hf_path = os.path.join(local_dir, "huggingface")
    config = AutoConfig.from_pretrained(hf_path)
    auto_model = _select_auto_model(config)

    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)

    model.to_empty(device="cpu")

    print(f"Saving model to {hf_path}")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model
    if args.hf_upload_path:
        # Push to hugging face
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
        api.upload_folder(folder_path=hf_path, repo_id=args.hf_upload_path, repo_type="model")
