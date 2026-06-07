#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_MODULES = (
    ("torch", "tensor runtime and model training"),
    ("numpy", "array runtime used by DataProto and metrics"),
    ("omegaconf", "Hydra/OmegaConf config loading"),
    ("ray", "distributed trainer runtime"),
    ("tensordict", "DataProto tensor batch container"),
    ("transformers", "model/tokenizer loading"),
    ("datasets", "HuggingFace dataset loading"),
    ("vllm", "default rollout backend"),
)

OPTIONAL_MODULES = (
    ("wandb", "optional experiment logging"),
    ("flash_attn", "optional attention acceleration"),
)

REQUIRED_FILES = (
    "scripts/config.yaml",
    "scripts/csrfaith_7b_grpo.sh",
    "scripts/csrfaith_smoke.sh",
    "scripts/debug_csr_batch.py",
    "scripts/build_csr_target_cache.py",
    "scripts/model_merger.py",
    "verl/trainer/main.py",
    "verl/trainer/ray_trainer.py",
    "verl/trainer/core_algos.py",
    "verl/trainer/config.py",
    "verl/utils/causal_rationale.py",
    "verl/utils/step_causal.py",
    "verl/utils/answer_normalization.py",
)

CSR_CONFIG_KEYS = (
    "enable_csrfaith",
    "csr_target_max_relations",
    "csr_target_max_objects",
    "csr_max_steps",
    "csr_max_step_interventions",
    "csr_step_cfs_alpha",
    "tau_coverage",
    "tau_step_cfs",
)


@dataclass(frozen=True)
class ModuleCheck:
    name: str
    available: bool
    purpose: str


@dataclass(frozen=True)
class FileCheck:
    path: str
    exists: bool


@dataclass(frozen=True)
class TextCheck:
    name: str
    passed: bool
    details: str


def check_modules(modules: Iterable[Sequence[str]]) -> List[ModuleCheck]:
    checks: List[ModuleCheck] = []
    for module in modules:
        name, purpose = module
        checks.append(
            ModuleCheck(
                name=name,
                available=importlib.util.find_spec(name) is not None,
                purpose=purpose,
            )
        )
    return checks


def check_files(repo_root: str, paths: Iterable[str]) -> List[FileCheck]:
    return [FileCheck(path=path, exists=os.path.exists(os.path.join(repo_root, path))) for path in paths]


def _read_text(repo_root: str, relative_path: str) -> str:
    with open(os.path.join(repo_root, relative_path), "r", encoding="utf-8") as f:
        return f.read()


def _extract_int_override(script_text: str, key: str) -> int:
    match = re.search(rf"{re.escape(key)}=([0-9]+)", script_text)
    return int(match.group(1)) if match else -1


def _check_rollout_token_budget(script_name: str, script_text: str) -> TextCheck:
    prompt_length = _extract_int_override(script_text, "data.max_prompt_length")
    response_length = _extract_int_override(script_text, "data.max_response_length")
    max_batched_tokens = _extract_int_override(script_text, "worker.rollout.max_num_batched_tokens")
    required_tokens = prompt_length + response_length
    passed = (
        prompt_length > 0
        and response_length > 0
        and max_batched_tokens >= required_tokens
    )
    return TextCheck(
        name=f"{script_name}_rollout_token_budget",
        passed=passed,
        details=(
            f"{script_name}: max_num_batched_tokens={max_batched_tokens}, "
            f"max_prompt_length+max_response_length={required_tokens}"
        ),
    )


def check_config_text(repo_root: str) -> List[TextCheck]:
    config_path = os.path.join(repo_root, "verl/trainer/config.py")
    if not os.path.exists(config_path):
        return [TextCheck(name="csr_config_fields", passed=False, details="verl/trainer/config.py is missing")]

    config_text = _read_text(repo_root, "verl/trainer/config.py")
    missing = [key for key in CSR_CONFIG_KEYS if key not in config_text]
    return [
        TextCheck(
            name="csr_config_fields",
            passed=not missing,
            details=(
                "all CSR config fields present in verl/trainer/config.py"
                if not missing
                else f"missing fields: {', '.join(missing)}"
            ),
        )
    ]


def check_training_scripts(repo_root: str) -> List[TextCheck]:
    checks: List[TextCheck] = []

    csr_script = os.path.join(repo_root, "scripts/csrfaith_7b_grpo.sh")
    if os.path.exists(csr_script):
        text = _read_text(repo_root, "scripts/csrfaith_7b_grpo.sh")
        checks.append(
            TextCheck(
                name="csr_train_enables_csr",
                passed="algorithm.enable_csrfaith=True" in text,
                details="csrfaith_7b_grpo.sh enables CSR-Faith",
            )
        )
        checks.append(
            TextCheck(
                name="csr_train_disables_cit",
                passed="algorithm.enable_citfaith=False" in text,
                details="csrfaith_7b_grpo.sh runs CSR independently from CIT",
            )
        )
        checks.append(
            TextCheck(
                name="csr_train_passes_cli_overrides",
                passed='"$@"' in text,
                details="csrfaith_7b_grpo.sh forwards extra CLI overrides",
            )
        )
        checks.append(
            TextCheck(
                name="csr_train_gpu_count_overridable",
                passed="N_GPUS=" in text and "trainer.n_gpus_per_node=${N_GPUS}" in text,
                details="csrfaith_7b_grpo.sh lets N_GPUS override trainer.n_gpus_per_node",
            )
        )
        checks.append(
            TextCheck(
                name="csr_train_cuda_visible_devices_overridable",
                passed="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}" in text,
                details="csrfaith_7b_grpo.sh lets CUDA_VISIBLE_DEVICES override visible GPUs",
            )
        )
        checks.append(
            TextCheck(
                name="csr_train_keeps_intermediate_checkpoints",
                passed="trainer.save_freq=25" in text and "trainer.save_limit=3" in text,
                details="csrfaith_7b_grpo.sh keeps the 25/50/75 checkpoint cadence",
            )
        )
        checks.append(
            TextCheck(
                name="csr_train_uses_scene_graph_answer_key",
                passed='data.answer_key="answer"' in text and 'data.answer_key="answer_option_text"' not in text,
                details="csrfaith_7b_grpo.sh uses the scene-graph answer column for reward and CSR targets",
            )
        )
        checks.append(_check_rollout_token_budget("csr_train", text))
    else:
        checks.append(TextCheck(name="csr_train_script", passed=False, details="scripts/csrfaith_7b_grpo.sh is missing"))

    smoke_script = os.path.join(repo_root, "scripts/csrfaith_smoke.sh")
    if os.path.exists(smoke_script):
        text = _read_text(repo_root, "scripts/csrfaith_smoke.sh")
        checks.append(
            TextCheck(
                name="csr_smoke_is_short",
                passed="trainer.max_steps=2" in text and "worker.rollout.n=2" in text,
                details="csrfaith_smoke.sh keeps max_steps and rollout.n small",
            )
        )
        checks.append(
            TextCheck(
                name="csr_smoke_enables_csr",
                passed="algorithm.enable_csrfaith=True" in text,
                details="csrfaith_smoke.sh enables CSR-Faith",
            )
        )
        checks.append(
            TextCheck(
                name="csr_smoke_passes_cli_overrides",
                passed='"$@"' in text,
                details="csrfaith_smoke.sh forwards extra CLI overrides",
            )
        )
        checks.append(
            TextCheck(
                name="csr_smoke_gpu_count_overridable",
                passed="N_GPUS=" in text and "trainer.n_gpus_per_node=${N_GPUS}" in text,
                details="csrfaith_smoke.sh lets N_GPUS override trainer.n_gpus_per_node",
            )
        )
        checks.append(
            TextCheck(
                name="csr_smoke_keeps_checkpoint_smoke_steps",
                passed="trainer.save_limit=3" in text,
                details="csrfaith_smoke.sh can retain global_step_1 when save_freq is overridden to 1",
            )
        )
        checks.append(
            TextCheck(
                name="csr_smoke_disables_kl_reference_policy",
                passed="algorithm.disable_kl=True" in text and "algorithm.use_kl_loss=False" in text,
                details="csrfaith_smoke.sh disables KL/ref-policy work for the 1-GPU smoke path",
            )
        )
        checks.append(
            TextCheck(
                name="csr_smoke_uses_scene_graph_answer_key",
                passed='data.answer_key="answer"' in text and 'data.answer_key="answer_option_text"' not in text,
                details="csrfaith_smoke.sh uses the scene-graph answer column for reward and CSR targets",
            )
        )
        checks.append(_check_rollout_token_budget("csr_smoke", text))
    else:
        checks.append(TextCheck(name="csr_smoke_script", passed=False, details="scripts/csrfaith_smoke.sh is missing"))

    return checks


def build_report(repo_root: str) -> Dict[str, object]:
    required_modules = check_modules(REQUIRED_MODULES)
    optional_modules = check_modules(OPTIONAL_MODULES)
    files = check_files(repo_root, REQUIRED_FILES)
    text_checks = check_config_text(repo_root) + check_training_scripts(repo_root)

    missing_required_modules = [check.name for check in required_modules if not check.available]
    missing_files = [check.path for check in files if not check.exists]
    failed_text_checks = [check.name for check in text_checks if not check.passed]

    ready = not missing_required_modules and not missing_files and not failed_text_checks

    return {
        "repo_root": repo_root,
        "ready": ready,
        "required_modules": [asdict(check) for check in required_modules],
        "optional_modules": [asdict(check) for check in optional_modules],
        "files": [asdict(check) for check in files],
        "text_checks": [asdict(check) for check in text_checks],
        "missing_required_modules": missing_required_modules,
        "missing_files": missing_files,
        "failed_text_checks": failed_text_checks,
        "next_step": (
            "Run bash scripts/csrfaith_smoke.sh in a GPU/Ray/vLLM environment."
            if ready
            else "Install missing modules or restore missing files before running CSR-Faith smoke."
        ),
    }


def print_human_report(report: Dict[str, object]) -> None:
    status = "READY" if report["ready"] else "NOT READY"
    print(f"CSR-Faith readiness: {status}")
    print(f"repo_root: {report['repo_root']}")

    missing_modules = report["missing_required_modules"]
    if missing_modules:
        print("missing required modules:")
        for module in missing_modules:
            print(f"  - {module}")

    missing_files = report["missing_files"]
    if missing_files:
        print("missing files:")
        for path in missing_files:
            print(f"  - {path}")

    failed_text_checks = report["failed_text_checks"]
    if failed_text_checks:
        print("failed text checks:")
        for name in failed_text_checks:
            print(f"  - {name}")

    optional_missing = [check["name"] for check in report["optional_modules"] if not check["available"]]
    if optional_missing:
        print("missing optional modules:")
        for module in optional_missing:
            print(f"  - {module}")

    print(f"next_step: {report['next_step']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether the local environment can run CSR-Faith smoke/training.")
    parser.add_argument("--repo-root", default=REPO_ROOT, help="Repository root to inspect.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit with status 0.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = os.path.abspath(args.repo_root)
    report = build_report(repo_root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_human_report(report)
    if not report["ready"] and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
