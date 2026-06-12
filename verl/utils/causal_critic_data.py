import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .answer_normalization import answers_equal, normalize_answer
from .causal_rationale import RationaleTarget, SpatialFact, split_thinking_steps
from .step_causal import StepIntervention


CURRICULUM_DISCOVERY = "discovery"
CURRICULUM_REFINEMENT = "refinement"


@dataclass
class CausalCriticExample:
    uid: str
    question: str
    target_objects: List[str]
    target_relations: List[Dict[str, str]]
    target_confidence: float
    response_answer: str
    step_index: int
    step_text: str
    intervention_type: str
    perturbed_step_preview: str
    counterfactual_answer: Optional[str]
    label_answer_changed: Optional[int]
    source: str
    policy_checkpoint: Optional[str] = None
    curriculum_phase: str = CURRICULUM_DISCOVERY
    reward_vector: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "uid": self.uid,
            "question": self.question,
            "target_objects": list(self.target_objects),
            "target_relations": [dict(rel) for rel in self.target_relations],
            "target_confidence": float(self.target_confidence),
            "response_answer": self.response_answer,
            "step_index": int(self.step_index),
            "step_text": self.step_text,
            "intervention_type": self.intervention_type,
            "perturbed_step_preview": self.perturbed_step_preview,
            "counterfactual_answer": self.counterfactual_answer,
            "label_answer_changed": self.label_answer_changed,
            "source": self.source,
            "policy_checkpoint": self.policy_checkpoint,
            "curriculum_phase": self.curriculum_phase,
            "reward_vector": dict(self.reward_vector),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CausalCriticExample":
        return cls(
            uid=str(data.get("uid", "")),
            question=str(data.get("question", "")),
            target_objects=[str(item) for item in data.get("target_objects", []) or []],
            target_relations=[
                _normalize_relation_dict(item)
                for item in data.get("target_relations", []) or []
                if isinstance(item, dict)
            ],
            target_confidence=float(data.get("target_confidence", 0.0) or 0.0),
            response_answer=str(data.get("response_answer", "")),
            step_index=int(data.get("step_index", -1) or -1),
            step_text=str(data.get("step_text", "")),
            intervention_type=str(data.get("intervention_type", "")),
            perturbed_step_preview=str(data.get("perturbed_step_preview", "")),
            counterfactual_answer=(
                None if data.get("counterfactual_answer") is None else str(data.get("counterfactual_answer", ""))
            ),
            label_answer_changed=_coerce_optional_int(data.get("label_answer_changed")),
            source=str(data.get("source", "")),
            policy_checkpoint=(
                None if data.get("policy_checkpoint") is None else str(data.get("policy_checkpoint", ""))
            ),
            curriculum_phase=str(data.get("curriculum_phase", CURRICULUM_DISCOVERY) or CURRICULUM_DISCOVERY),
            reward_vector={
                str(key): float(value)
                for key, value in (data.get("reward_vector", {}) or {}).items()
            },
        )


def _coerce_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_relation_dict(relation: Dict[str, object]) -> Dict[str, str]:
    return {
        "subject": str(relation.get("subject", "")),
        "predicate": str(relation.get("predicate", "")),
        "object": str(relation.get("object", "")),
    }


def _relation_to_dict(relation: SpatialFact) -> Dict[str, str]:
    return {
        "subject": str(relation.subject or ""),
        "predicate": str(relation.predicate or ""),
        "object": str(relation.object or ""),
    }


def target_to_relation_dicts(target: RationaleTarget) -> List[Dict[str, str]]:
    return [_relation_to_dict(relation) for relation in target.relations]


def infer_curriculum_phase(step_index: int, target_confidence: float) -> str:
    """EVO-RAG-style two-stage schedule hook for later trainer integration."""
    if step_index <= 1 or target_confidence < 0.5:
        return CURRICULUM_DISCOVERY
    return CURRICULUM_REFINEMENT


def infer_answer_changed_label(response_answer: str, counterfactual_answer: Optional[str]) -> Optional[int]:
    if counterfactual_answer is None:
        return None
    if not normalize_answer(response_answer) or not normalize_answer(counterfactual_answer):
        return None
    return int(not answers_equal(response_answer, counterfactual_answer))


def build_reward_vector(
    *,
    label_answer_changed: Optional[int],
    target_confidence: float,
    intervention_type: str,
) -> Dict[str, float]:
    valid = 1.0 if label_answer_changed is not None else 0.0
    causal_effect = float(label_answer_changed or 0)
    return {
        "causal_effect": causal_effect,
        "intervention_valid": valid,
        "target_confidence": float(target_confidence),
        "relation_intervention": 1.0 if intervention_type == "relation" else 0.0,
        "entity_intervention": 1.0 if intervention_type == "entity" else 0.0,
        "mask_intervention": 1.0 if intervention_type == "mask" else 0.0,
    }


def _perturbed_step_preview(intervention: StepIntervention, max_chars: int = 240) -> str:
    steps = split_thinking_steps(intervention.perturbed_response)
    if 0 <= intervention.step_index < len(steps):
        preview = steps[intervention.step_index]
    else:
        preview = intervention.perturbed_response
    preview = " ".join(str(preview or "").split())
    return preview[:max_chars]


def build_critic_examples(
    problem: str,
    target: RationaleTarget,
    response_text: str,
    interventions: List[StepIntervention],
    counterfactual_answers: List[Optional[str]],
    *,
    uid: str,
    response_answer: str,
    policy_checkpoint: Optional[str] = None,
    source: str = "online_policy_continuation",
) -> List[CausalCriticExample]:
    examples: List[CausalCriticExample] = []
    relation_dicts = target_to_relation_dicts(target)
    for idx, intervention in enumerate(interventions):
        counterfactual_answer = counterfactual_answers[idx] if idx < len(counterfactual_answers) else None
        label = infer_answer_changed_label(response_answer, counterfactual_answer)
        phase = infer_curriculum_phase(intervention.step_index, target.confidence)
        example_uid = f"{uid}:step{intervention.step_index}:{intervention.intervention_type}:{idx}"
        examples.append(
            CausalCriticExample(
                uid=example_uid,
                question=str(problem or ""),
                target_objects=list(target.objects),
                target_relations=relation_dicts,
                target_confidence=float(target.confidence),
                response_answer=str(response_answer or ""),
                step_index=int(intervention.step_index),
                step_text=str(intervention.original_step or ""),
                intervention_type=str(intervention.intervention_type or ""),
                perturbed_step_preview=_perturbed_step_preview(intervention),
                counterfactual_answer=counterfactual_answer,
                label_answer_changed=label,
                source=source,
                policy_checkpoint=policy_checkpoint,
                curriculum_phase=phase,
                reward_vector=build_reward_vector(
                    label_answer_changed=label,
                    target_confidence=target.confidence,
                    intervention_type=intervention.intervention_type,
                ),
            )
        )
    return examples


def write_jsonl(examples: Iterable[CausalCriticExample], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[CausalCriticExample]:
    examples: List[CausalCriticExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(CausalCriticExample.from_dict(json.loads(line)))
    return examples


def filter_labeled_examples(
    examples: Iterable[CausalCriticExample],
    *,
    min_target_confidence: float = 0.0,
) -> List[CausalCriticExample]:
    return [
        example
        for example in examples
        if example.label_answer_changed is not None and example.target_confidence >= min_target_confidence
    ]
