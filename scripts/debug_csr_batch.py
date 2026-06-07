#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Iterable


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from verl.utils.causal_rationale import (  # noqa: E402
    build_causal_rationale_target,
    extract_facts_from_response,
    extract_gt_scene_and_answer,
    score_rationale,
    split_thinking_steps,
)
from verl.utils.step_causal import build_prefixes_for_step_interventions, generate_step_interventions  # noqa: E402
from scripts.csr_debug_io import get_field, iter_input_records  # noqa: E402


def _record_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if not (args.problem and args.ground_truth and args.response):
        raise ValueError("--problem, --ground-truth, and --response must be provided together.")
    return {
        "problem": args.problem,
        "ground_truth": args.ground_truth,
        "response": args.response,
    }


def _iter_records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.input_json:
        yield from iter_input_records(input_json=args.input_json, limit=args.limit)
    elif args.data:
        yield from iter_input_records(data=args.data, limit=args.limit)
    else:
        yield _record_from_args(args)


def analyze_record(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    problem = get_field(record, args.problem_key)
    ground_truth = get_field(record, args.ground_truth_key)
    if not ground_truth:
        for fallback_key in ("ground_truth", "answer", "answer_option_text"):
            ground_truth = get_field(record, fallback_key)
            if ground_truth:
                break

    response = get_field(record, args.response_key)
    if not response:
        for fallback_key in ("response", "predict", "prediction", "generated_response"):
            response = get_field(record, fallback_key)
            if response:
                break
    if not response:
        response = ground_truth

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
    rationale_score = score_rationale(response, target)
    interventions = generate_step_interventions(
        response_text=response,
        target=target,
        max_steps=args.max_steps,
        max_interventions_per_step=args.max_step_interventions,
    )
    prefixes = build_prefixes_for_step_interventions(interventions)

    return {
        "problem_preview": problem[: args.preview_chars],
        "gt_answer": gt_answer,
        "response_answer": extract_gt_scene_and_answer(response)[1],
        "target": asdict(target),
        "response_facts": [asdict(fact) for fact in extract_facts_from_response(response)],
        "thinking_steps": split_thinking_steps(response),
        "rationale_score": asdict(rationale_score),
        "step_interventions": [
            {
                "step_index": intervention.step_index,
                "type": intervention.intervention_type,
                "original_step": intervention.original_step,
                "perturbed_preview": intervention.perturbed_response[: args.preview_chars],
                "prefix_preview": prefixes[idx][: args.preview_chars],
            }
            for idx, intervention in enumerate(interventions)
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug CSR-Faith rationale targets and step interventions.")
    parser.add_argument("--data", help="Optional HuggingFace/local dataset path with @split suffix.")
    parser.add_argument("--input-json", help="JSON or JSONL records with problem/ground_truth/response fields.")
    parser.add_argument("--problem", help="Single-record problem text.")
    parser.add_argument("--ground-truth", help="Single-record ground truth text with <scene>/<answer>.")
    parser.add_argument("--response", help="Single-record generated response text.")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--ground-truth-key", default="ground_truth")
    parser.add_argument("--response-key", default="response")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-relations", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-step-interventions", type=int, default=1)
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--jsonl", action="store_true", help="Emit one compact JSON object per line.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = list(_iter_records(args))
    outputs = [analyze_record(record, args) for record in records[: args.limit]]
    if args.jsonl:
        for output in outputs:
            print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
