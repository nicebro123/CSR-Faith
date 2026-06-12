#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.csr_debug_io import get_field, iter_input_records  # noqa: E402
from verl.utils.causal_critic_data import build_critic_examples, filter_labeled_examples, write_jsonl  # noqa: E402
from verl.utils.causal_rationale import build_causal_rationale_target, extract_gt_scene_and_answer  # noqa: E402
from verl.utils.step_causal import generate_step_interventions  # noqa: E402


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer_text(text: str) -> str:
    match = _ANSWER_RE.search(str(text or ""))
    return match.group(1).strip() if match else str(text or "").strip()


def _stable_record_id(record: Dict[str, Any], idx: int, problem: str, response: str) -> str:
    for key in ("id", "uid", "qid", "question_id", "sample_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    digest = hashlib.sha1(f"{idx}\n{problem}\n{response}".encode("utf-8")).hexdigest()
    return digest[:16]


def _resolve_ground_truth(record: Dict[str, Any], ground_truth_key: str) -> str:
    ground_truth = get_field(record, ground_truth_key)
    if ground_truth:
        return ground_truth
    for fallback_key in ("ground_truth", "answer", "answer_option_text"):
        ground_truth = get_field(record, fallback_key)
        if ground_truth:
            return ground_truth
    return ""


def _resolve_response(record: Dict[str, Any], response_key: str) -> str:
    response = get_field(record, response_key)
    if response:
        return response
    for fallback_key in ("response", "generated_response", "completion", "text"):
        response = get_field(record, fallback_key)
        if response:
            return response
    return ""


def _coerce_answer_list(value: Any) -> List[Optional[str]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [None if item is None else str(item) for item in value]
    if isinstance(value, tuple):
        return [None if item is None else str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [part.strip() for part in stripped.split("||")]
        return _coerce_answer_list(parsed)
    return [str(value)]


def _counterfactual_answers(record: Dict[str, Any], key: str) -> List[Optional[str]]:
    if key in record:
        return _coerce_answer_list(record.get(key))
    for fallback_key in ("counterfactual_answers", "cf_answers", "step_cf_answers"):
        if fallback_key in record:
            return _coerce_answer_list(record.get(fallback_key))
    return []


def build_dataset_records(records: List[Dict[str, Any]], args: argparse.Namespace):
    examples = []
    skipped_no_response = 0
    skipped_no_interventions = 0

    for idx, record in enumerate(records):
        problem = get_field(record, args.problem_key)
        ground_truth = _resolve_ground_truth(record, args.ground_truth_key)
        response = _resolve_response(record, args.response_key)
        if not response:
            skipped_no_response += 1
            continue

        gt_scene, gt_answer = extract_gt_scene_and_answer(ground_truth)
        if not gt_scene:
            problem_scene, _ = extract_gt_scene_and_answer(problem)
            gt_scene = problem_scene

        target = build_causal_rationale_target(
            problem=problem,
            gt_scene=gt_scene,
            gt_answer=gt_answer,
            max_relations=args.max_relations,
            max_objects=args.max_objects,
        )
        interventions = generate_step_interventions(
            response,
            target,
            max_steps=args.max_steps,
            max_interventions_per_step=args.max_interventions_per_step,
            rollout_index=idx,
        )
        if not interventions:
            skipped_no_interventions += 1
            continue

        cf_answers = _counterfactual_answers(record, args.counterfactual_answer_key)
        response_answer = get_field(record, args.response_answer_key) or _extract_answer_text(response)
        if not response_answer:
            response_answer = gt_answer

        record_uid = _stable_record_id(record, idx, problem, response)
        record_examples = build_critic_examples(
            problem=problem,
            target=target,
            response_text=response,
            interventions=interventions,
            counterfactual_answers=cf_answers,
            uid=record_uid,
            response_answer=response_answer,
            policy_checkpoint=args.policy_checkpoint,
            source=args.source,
        )
        examples.extend(record_examples)

    if args.drop_invalid_labels:
        examples = filter_labeled_examples(examples, min_target_confidence=args.min_target_confidence)

    return examples, {
        "skipped_no_response": skipped_no_response,
        "skipped_no_interventions": skipped_no_interventions,
    }


def summarize(examples, extra_counts: Dict[str, int], output_path: str) -> Dict[str, Any]:
    labeled = [example for example in examples if example.label_answer_changed is not None]
    positives = sum(int(example.label_answer_changed) for example in labeled)
    by_phase: Dict[str, int] = {}
    by_intervention: Dict[str, int] = {}
    for example in examples:
        by_phase[example.curriculum_phase] = by_phase.get(example.curriculum_phase, 0) + 1
        by_intervention[example.intervention_type] = by_intervention.get(example.intervention_type, 0) + 1
    return {
        "output": output_path,
        "num_examples": len(examples),
        "num_labeled": len(labeled),
        "positive_rate": positives / max(len(labeled), 1),
        "by_phase": by_phase,
        "by_intervention": by_intervention,
        **extra_counts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Causal Spatial Critic JSONL from rollout records.")
    parser.add_argument("--data", help="Optional HuggingFace/local dataset path with @split suffix.")
    parser.add_argument("--input-json", help="JSON or JSONL rollout records.")
    parser.add_argument("--output", required=True, help="Output JSONL path, e.g. cache/causal_critic/train.jsonl.")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--ground-truth-key", default="ground_truth")
    parser.add_argument("--response-key", default="response")
    parser.add_argument("--response-answer-key", default="response_answer")
    parser.add_argument("--counterfactual-answer-key", default="counterfactual_answers")
    parser.add_argument("--policy-checkpoint", default=None)
    parser.add_argument("--source", default="online_policy_continuation")
    parser.add_argument("--limit", type=int, default=100000000)
    parser.add_argument("--max-relations", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-interventions-per-step", type=int, default=1)
    parser.add_argument("--min-target-confidence", type=float, default=0.0)
    parser.add_argument("--drop-invalid-labels", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not (args.input_json or args.data):
        raise ValueError("Provide either --input-json or --data.")
    records = list(iter_input_records(input_json=args.input_json, data=args.data, limit=args.limit))
    examples, extra_counts = build_dataset_records(records, args)
    write_jsonl(examples, args.output)
    print(json.dumps(summarize(examples, extra_counts, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
