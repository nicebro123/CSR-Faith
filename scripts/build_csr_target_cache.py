#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.csr_debug_io import get_field, iter_input_records  # noqa: E402
from verl.utils.causal_rationale import build_causal_rationale_target, extract_gt_scene_and_answer  # noqa: E402


def _stable_record_id(record: Dict[str, Any], idx: int, problem: str, ground_truth: str) -> str:
    for key in ("id", "uid", "qid", "question_id", "sample_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    digest = hashlib.sha1(f"{idx}\n{problem}\n{ground_truth}".encode("utf-8")).hexdigest()
    return digest


def _resolve_ground_truth(record: Dict[str, Any], ground_truth_key: str) -> str:
    ground_truth = get_field(record, ground_truth_key)
    if ground_truth:
        return ground_truth
    for fallback_key in ("ground_truth", "answer", "answer_option_text"):
        ground_truth = get_field(record, fallback_key)
        if ground_truth:
            return ground_truth
    return ""


def build_cache_records(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    outputs = []
    for idx, record in enumerate(records):
        problem = get_field(record, args.problem_key)
        ground_truth = _resolve_ground_truth(record, args.ground_truth_key)
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
        outputs.append(
            {
                "id": _stable_record_id(record, idx, problem, ground_truth),
                "index": idx,
                "gt_answer": gt_answer,
                "target": asdict(target),
                "target_fact_count": len(target.objects) + len(target.relations),
                "target_confidence": target.confidence,
                "target_source": target.source,
            }
        )
    return outputs


def write_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a CSR-Faith target cache JSONL from existing dataset fields.")
    parser.add_argument("--data", help="Optional HuggingFace/local dataset path with @split suffix.")
    parser.add_argument("--input-json", help="JSON or JSONL records with problem and ground_truth/answer fields.")
    parser.add_argument("--output", required=True, help="Output JSONL path, e.g. cache/csr_targets/stvqa/train.jsonl.")
    parser.add_argument("--problem-key", default="problem")
    parser.add_argument("--ground-truth-key", default="ground_truth")
    parser.add_argument("--limit", type=int, default=100000000)
    parser.add_argument("--max-relations", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=6)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not (args.input_json or args.data):
        raise ValueError("Provide either --input-json or --data.")
    records = list(iter_input_records(input_json=args.input_json, data=args.data, limit=args.limit))
    cache_records = build_cache_records(records, args)
    write_jsonl(cache_records, args.output)
    n_with_targets = sum(1 for record in cache_records if record["target_fact_count"] > 0)
    print(
        json.dumps(
            {
                "output": args.output,
                "num_records": len(cache_records),
                "num_with_targets": n_with_targets,
                "target_coverage": n_with_targets / max(len(cache_records), 1),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
