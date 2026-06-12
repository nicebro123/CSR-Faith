import hashlib
import json
import math
import os
import random
import re
from typing import Dict, Iterable, List, Tuple

from verl.utils.causal_critic_data import CausalCriticExample, filter_labeled_examples, read_jsonl


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_.]+")
_CHECKPOINT_VERSION = 1


def serialize_critic_input(example: CausalCriticExample) -> str:
    relation_text = "; ".join(
        "{subject} {predicate} {object}".format(
            subject=relation.get("subject", ""),
            predicate=relation.get("predicate", ""),
            object=relation.get("object", ""),
        ).strip()
        for relation in example.target_relations
    )
    object_text = "; ".join(example.target_objects)
    return "\n".join(
        [
            "[QUESTION] " + example.question,
            "[TARGET_OBJECTS] " + object_text,
            "[TARGET_RELATIONS] " + relation_text,
            "[TARGET_CONFIDENCE] " + f"{example.target_confidence:.3f}",
            "[STEP] " + example.step_text,
            "[INTERVENTION] " + example.intervention_type,
            "[PHASE] " + example.curriculum_phase,
        ]
    )


def _stable_hash(value: str, n_features: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % n_features


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(str(text or ""))]


def _sigmoid(value: float) -> float:
    if value >= 30:
        return 1.0
    if value <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _feature_dict(example: CausalCriticExample, n_features: int) -> Dict[int, float]:
    serialized = serialize_critic_input(example)
    tokens = _tokenize(serialized)
    features: Dict[int, float] = {}

    for token in tokens:
        idx = _stable_hash("u:" + token, n_features)
        features[idx] = features.get(idx, 0.0) + 1.0

    for left, right in zip(tokens, tokens[1:]):
        idx = _stable_hash("b:" + left + "__" + right, n_features)
        features[idx] = features.get(idx, 0.0) + 1.0

    for name, value in [
        ("intervention", example.intervention_type),
        ("phase", example.curriculum_phase),
    ]:
        idx = _stable_hash("cat:" + name + "=" + str(value), n_features)
        features[idx] = features.get(idx, 0.0) + 1.0

    if example.target_confidence >= 0.75:
        bucket = "high"
    elif example.target_confidence >= 0.5:
        bucket = "mid"
    else:
        bucket = "low"
    idx = _stable_hash("target_confidence=" + bucket, n_features)
    features[idx] = features.get(idx, 0.0) + 1.0

    norm = math.sqrt(sum(value * value for value in features.values()))
    if norm > 0:
        features = {idx: value / norm for idx, value in features.items()}
    return features


def _labels(examples: Iterable[CausalCriticExample]) -> List[int]:
    return [int(example.label_answer_changed) for example in examples if example.label_answer_changed is not None]


def _classification_metrics(labels: List[int], scores: List[float], threshold: float) -> Dict[str, float]:
    if not labels:
        return {
            "num_examples": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.0,
            "positive_rate": 0.0,
            "mean_score": 0.0,
        }

    preds = [int(score >= threshold) for score in scores]
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "num_examples": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _auroc(labels, scores),
        "positive_rate": sum(labels) / len(labels),
        "mean_score": sum(scores) / len(scores),
    }


def _auroc(labels: List[int], scores: List[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ranked[k][0]] = avg_rank
        i = j

    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / max(positives * negatives, 1)


class CausalSpatialCritic:
    def __init__(
        self,
        *,
        n_features: int = 4096,
        threshold: float = 0.5,
        weights: Dict[int, float] = None,
        bias: float = 0.0,
        metadata: Dict[str, object] = None,
    ):
        self.n_features = int(n_features)
        self.threshold = float(threshold)
        self.weights: Dict[int, float] = dict(weights or {})
        self.bias = float(bias)
        self.metadata = dict(metadata or {})

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CausalSpatialCritic":
        checkpoint_path = path
        if os.path.isdir(path):
            checkpoint_path = os.path.join(path, "critic.json")
        with open(checkpoint_path, encoding="utf-8") as f:
            payload = json.load(f)
        if int(payload.get("version", 0)) != _CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported CausalSpatialCritic checkpoint version: {payload.get('version')}")
        weights = {int(key): float(value) for key, value in payload.get("weights", {}).items()}
        metadata = dict(payload.get("metadata", {}) or {})
        metadata["loaded_device"] = device
        return cls(
            n_features=int(payload.get("n_features", 4096)),
            threshold=float(payload.get("threshold", 0.5)),
            weights=weights,
            bias=float(payload.get("bias", 0.0)),
            metadata=metadata,
        )

    @classmethod
    def train(
        cls,
        examples: List[CausalCriticExample],
        *,
        n_features: int = 4096,
        epochs: int = 12,
        learning_rate: float = 0.3,
        l2: float = 1e-5,
        seed: int = 13,
    ) -> "CausalSpatialCritic":
        model = cls(n_features=n_features)
        model.fit(
            examples,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        return model

    @classmethod
    def train_from_jsonl(
        cls,
        train_jsonl: str,
        *,
        min_target_confidence: float = 0.0,
        n_features: int = 4096,
        epochs: int = 12,
        learning_rate: float = 0.3,
        l2: float = 1e-5,
        seed: int = 13,
    ) -> "CausalSpatialCritic":
        examples = filter_labeled_examples(read_jsonl(train_jsonl), min_target_confidence=min_target_confidence)
        return cls.train(
            examples,
            n_features=n_features,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )

    def fit(
        self,
        examples: List[CausalCriticExample],
        *,
        epochs: int = 12,
        learning_rate: float = 0.3,
        l2: float = 1e-5,
        seed: int = 13,
    ) -> None:
        labeled = filter_labeled_examples(examples)
        if not labeled:
            raise ValueError("No labeled critic examples available for training.")

        labels = _labels(labeled)
        positives = sum(labels)
        negatives = len(labels) - positives
        pos_weight = len(labels) / max(2 * positives, 1)
        neg_weight = len(labels) / max(2 * negatives, 1)
        features = [_feature_dict(example, self.n_features) for example in labeled]
        rng = random.Random(seed)
        order = list(range(len(labeled)))

        for _ in range(max(int(epochs), 1)):
            rng.shuffle(order)
            for idx in order:
                label = labels[idx]
                feats = features[idx]
                pred = _sigmoid(self._score_features(feats))
                class_weight = pos_weight if label == 1 else neg_weight
                grad = (pred - label) * class_weight
                for feat_idx, value in feats.items():
                    old = self.weights.get(feat_idx, 0.0)
                    self.weights[feat_idx] = old - learning_rate * (grad * value + l2 * old)
                self.bias -= learning_rate * grad

        self.threshold = self._calibrate_threshold(labeled)
        self.metadata.update(
            {
                "num_train_examples": len(labeled),
                "positive_rate": positives / len(labeled),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "l2": float(l2),
            }
        )

    def _score_features(self, features: Dict[int, float]) -> float:
        return self.bias + sum(self.weights.get(idx, 0.0) * value for idx, value in features.items())

    def predict_proba(self, example: CausalCriticExample) -> float:
        return _sigmoid(self._score_features(_feature_dict(example, self.n_features)))

    def score_batch(self, examples: List[CausalCriticExample]) -> List[float]:
        return [self.predict_proba(example) for example in examples]

    def evaluate(self, examples: List[CausalCriticExample]) -> Dict[str, float]:
        labeled = filter_labeled_examples(examples)
        labels = _labels(labeled)
        scores = self.score_batch(labeled)
        return _classification_metrics(labels, scores, self.threshold)

    def _calibrate_threshold(self, examples: List[CausalCriticExample]) -> float:
        labeled = filter_labeled_examples(examples)
        labels = _labels(labeled)
        scores = self.score_batch(labeled)
        if not labels:
            return 0.5

        candidates = sorted(set([0.5] + scores))
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            metrics = _classification_metrics(labels, scores, threshold)
            if metrics["f1"] > best_f1:
                best_threshold = threshold
                best_f1 = metrics["f1"]
        return best_threshold

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, "critic.json")
        payload = {
            "version": _CHECKPOINT_VERSION,
            "n_features": self.n_features,
            "threshold": self.threshold,
            "bias": self.bias,
            "weights": {str(key): value for key, value in sorted(self.weights.items())},
            "metadata": self.metadata,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def majority_baseline_metrics(examples: List[CausalCriticExample]) -> Dict[str, float]:
    labeled = filter_labeled_examples(examples)
    labels = _labels(labeled)
    if not labels:
        return _classification_metrics([], [], 0.5)
    positive_rate = sum(labels) / len(labels)
    majority_score = 1.0 if positive_rate >= 0.5 else 0.0
    scores = [majority_score for _ in labels]
    return _classification_metrics(labels, scores, 0.5)


def train_val_split(
    examples: List[CausalCriticExample],
    *,
    val_ratio: float = 0.2,
    seed: int = 13,
) -> Tuple[List[CausalCriticExample], List[CausalCriticExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * val_ratio))
    if val_size <= 0 and len(shuffled) > 1:
        val_size = 1
    if val_size >= len(shuffled):
        val_size = max(len(shuffled) - 1, 0)
    return shuffled[val_size:], shuffled[:val_size]
