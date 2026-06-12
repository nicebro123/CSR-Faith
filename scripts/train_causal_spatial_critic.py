#!/usr/bin/env python3
import argparse
import json
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from verl.models.causal_spatial_critic import (  # noqa: E402
    CausalSpatialCritic,
    majority_baseline_metrics,
    train_val_split,
)
from verl.utils.causal_critic_data import filter_labeled_examples, read_jsonl  # noqa: E402


def _write_json(payload, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight Causal Spatial Critic.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-target-confidence", type=float, default=0.0)
    parser.add_argument("--n-features", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    all_train = filter_labeled_examples(
        read_jsonl(args.train_jsonl),
        min_target_confidence=args.min_target_confidence,
    )
    if args.val_jsonl:
        train_examples = all_train
        val_examples = filter_labeled_examples(
            read_jsonl(args.val_jsonl),
            min_target_confidence=args.min_target_confidence,
        )
    else:
        train_examples, val_examples = train_val_split(all_train, val_ratio=args.val_ratio, seed=args.seed)

    if not train_examples:
        raise ValueError("No labeled training examples after filtering.")

    model = CausalSpatialCritic.train(
        train_examples,
        n_features=args.n_features,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    model.save(args.output_dir)

    train_metrics = model.evaluate(train_examples)
    val_metrics = model.evaluate(val_examples) if val_examples else {}
    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "majority_baseline_train": majority_baseline_metrics(train_examples),
        "majority_baseline_val": majority_baseline_metrics(val_examples) if val_examples else {},
        "config": {
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl,
            "min_target_confidence": args.min_target_confidence,
            "n_features": args.n_features,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
        },
    }
    _write_json(metrics, os.path.join(args.output_dir, "metrics.json"))
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
