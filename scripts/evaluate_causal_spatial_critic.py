#!/usr/bin/env python3
import argparse
import json
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from verl.models.causal_spatial_critic import CausalSpatialCritic, majority_baseline_metrics  # noqa: E402
from verl.utils.causal_critic_data import filter_labeled_examples, read_jsonl  # noqa: E402


def _write_json(payload, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Causal Spatial Critic checkpoint.")
    parser.add_argument("--critic-path", required=True, help="Checkpoint directory or critic.json path.")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--min-target-confidence", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    critic = CausalSpatialCritic.load(args.critic_path)
    examples = filter_labeled_examples(
        read_jsonl(args.eval_jsonl),
        min_target_confidence=args.min_target_confidence,
    )
    metrics = {
        "eval": critic.evaluate(examples),
        "majority_baseline": majority_baseline_metrics(examples),
        "config": {
            "critic_path": args.critic_path,
            "eval_jsonl": args.eval_jsonl,
            "min_target_confidence": args.min_target_confidence,
        },
    }
    if args.output_json:
        _write_json(metrics, args.output_json)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
