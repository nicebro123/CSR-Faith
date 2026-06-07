import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


_SCENE_RE = re.compile(r"<scene>(.*?)</scene>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

_RELATION_SYNONYMS = {
    "left": {"left", "left of", "to the left of"},
    "right": {"right", "right of", "to the right of"},
    "above": {"above", "over", "on top of", "higher"},
    "below": {"below", "under", "beneath", "lower"},
    "front": {"front", "in front of"},
    "behind": {"behind", "back"},
    "inside": {"inside", "in"},
    "outside": {"outside"},
    "near": {"near", "next to", "beside", "close"},
    "far": {"far", "far from", "away"},
}


@dataclass
class SpatialFact:
    fact_type: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    bbox: Optional[List[float]] = None
    source: str = "gt_scene"


@dataclass
class RationaleTarget:
    objects: List[str]
    relations: List[SpatialFact]
    answer: str
    confidence: float
    source: str


@dataclass
class RationaleScore:
    coverage: float
    precision: float
    compactness: float
    sufficiency: float
    necessity: float
    overall: float


def refine_name(value: str) -> str:
    return str(value or "").replace("_", " ").replace("-", " ").strip().lower()


def base_name(value: str) -> str:
    return refine_name(str(value or "").split(".")[0])


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or ""))}


def _parse_json_scene(text: str) -> dict:
    match = _SCENE_RE.search(str(text or ""))
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1).strip())
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_answer(text: str) -> str:
    match = _ANSWER_RE.search(str(text or ""))
    return match.group(1).strip() if match else str(text or "").strip()


def extract_gt_scene_and_answer(ground_truth: str) -> Tuple[dict, str]:
    return _parse_json_scene(ground_truth), _extract_answer(ground_truth)


def _valid_objects(scene: dict) -> List[dict]:
    objects = scene.get("objects") if isinstance(scene, dict) else []
    if not isinstance(objects, list):
        return []
    return [obj for obj in objects if isinstance(obj, dict) and isinstance(obj.get("id"), str)]


def _valid_relations(scene: dict) -> List[dict]:
    relations = scene.get("relationships") if isinstance(scene, dict) else []
    if not isinstance(relations, list):
        return []
    valid = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        if all(isinstance(rel.get(key), str) for key in ("subject", "predicate", "object")):
            valid.append(rel)
    return valid


def _object_lookup(scene: dict) -> Dict[str, dict]:
    return {refine_name(obj["id"]): obj for obj in _valid_objects(scene)}


def _relation_to_fact(rel: dict) -> SpatialFact:
    return SpatialFact(
        fact_type="relation",
        subject=refine_name(rel["subject"]),
        predicate=refine_name(rel["predicate"]),
        object=refine_name(rel["object"]),
        source="gt_scene",
    )


def _relation_overlap_score(rel: dict, query_tokens: set[str]) -> int:
    rel_tokens = _tokens(
        " ".join(
            [
                base_name(rel.get("subject", "")),
                refine_name(rel.get("predicate", "")),
                base_name(rel.get("object", "")),
            ]
        )
    )
    return len(rel_tokens & query_tokens)


def build_causal_rationale_target(
    problem: str,
    gt_scene: dict,
    gt_answer: str,
    *,
    max_relations: int = 4,
    max_objects: int = 6,
) -> RationaleTarget:
    objects = _valid_objects(gt_scene)
    relations = _valid_relations(gt_scene)
    query_tokens = _tokens(problem) | _tokens(gt_answer)
    object_by_id = _object_lookup(gt_scene)

    scored_relations = [
        (_relation_overlap_score(rel, query_tokens), rel)
        for rel in relations
    ]
    selected_relations = [
        rel for score, rel in sorted(scored_relations, key=lambda item: item[0], reverse=True) if score > 0
    ][:max_relations]

    selected_objects = set()
    source = "relation_overlap" if selected_relations else "empty"
    confidence = 1.0 if selected_relations else 0.0

    for rel in selected_relations:
        selected_objects.add(refine_name(rel["subject"]))
        selected_objects.add(refine_name(rel["object"]))

    if not selected_relations:
        for obj in objects:
            obj_tokens = _tokens(base_name(obj["id"]))
            if obj_tokens & query_tokens:
                selected_objects.add(refine_name(obj["id"]))
        if selected_objects:
            source = "object_overlap"
            confidence = 0.7

    if not selected_relations and not selected_objects and relations:
        selected_relations = relations[:max_relations]
        for rel in selected_relations:
            selected_objects.add(refine_name(rel["subject"]))
            selected_objects.add(refine_name(rel["object"]))
        source = "relation_fallback"
        confidence = 0.4

    if not selected_objects and objects:
        selected_objects.update(refine_name(obj["id"]) for obj in objects[:max_objects])
        source = "object_fallback"
        confidence = max(confidence, 0.3)

    # Preserve GT object order for deterministic logs.
    ordered_objects = []
    for obj in objects:
        obj_id = refine_name(obj["id"])
        if obj_id in selected_objects and obj_id not in ordered_objects:
            ordered_objects.append(obj_id)

    for obj_id in sorted(selected_objects):
        if obj_id not in object_by_id and obj_id not in ordered_objects:
            ordered_objects.append(obj_id)

    ordered_objects = ordered_objects[:max_objects]
    selected_relations = selected_relations[:max_relations]

    return RationaleTarget(
        objects=ordered_objects,
        relations=[_relation_to_fact(rel) for rel in selected_relations],
        answer=str(gt_answer or "").strip(),
        confidence=confidence,
        source=source,
    )


def _extract_response_scene(response_text: str) -> dict:
    return _parse_json_scene(response_text)


def extract_facts_from_response(response_text: str) -> List[SpatialFact]:
    scene = _extract_response_scene(response_text)
    facts: List[SpatialFact] = []
    for obj in _valid_objects(scene):
        facts.append(
            SpatialFact(
                fact_type="object",
                subject=refine_name(obj["id"]),
                bbox=obj.get("bbox") if isinstance(obj.get("bbox"), list) else None,
                source="response_scene",
            )
        )
    for rel in _valid_relations(scene):
        fact = _relation_to_fact(rel)
        fact.source = "response_scene"
        facts.append(fact)
    return facts


def split_thinking_steps(response_text: str) -> List[str]:
    match = _THINK_RE.search(str(response_text or ""))
    if not match:
        return []
    thinking = match.group(1).strip()
    if not thinking:
        return []

    numbered = re.split(r"(?:^|\n|\s)(?:\d+[\).]|step\s+\d+[:.)])\s*", thinking, flags=re.IGNORECASE)
    numbered = [part.strip() for part in numbered if part.strip()]
    if len(numbered) > 1:
        return numbered

    steps = re.split(r"(?<=[.!?])\s+|\n+", thinking)
    return [step.strip() for step in steps if step.strip()]


def _fact_key(fact: SpatialFact) -> Tuple[str, str, str, str]:
    return (
        fact.fact_type,
        refine_name(fact.subject),
        refine_name(fact.predicate),
        refine_name(fact.object),
    )


def _relation_aliases(predicate: str) -> set[str]:
    pred = refine_name(predicate)
    aliases = {pred}
    pred_tokens = _tokens(pred)
    for canonical, values in _RELATION_SYNONYMS.items():
        value_tokens = set().union(*(_tokens(value) for value in values))
        if pred in values or pred_tokens & value_tokens or canonical in pred_tokens:
            aliases.update(values)
    return {refine_name(alias) for alias in aliases}


def _contains_object(text: str, obj_id: str) -> bool:
    text_norm = refine_name(text)
    obj_norm = refine_name(obj_id)
    obj_base = base_name(obj_id)
    if not obj_norm:
        return False
    return bool(
        re.search(rf"\b{re.escape(obj_norm)}\b", text_norm)
        or (obj_base and re.search(rf"\b{re.escape(obj_base)}\b", text_norm))
    )


def _contains_relation(text: str, relation: SpatialFact, response_fact_keys: set[Tuple[str, str, str, str]]) -> bool:
    relation_key = _fact_key(relation)
    if relation_key in response_fact_keys:
        return True

    subject_ok = _contains_object(text, relation.subject or "")
    object_ok = _contains_object(text, relation.object or "")
    text_norm = refine_name(text)
    predicate_ok = any(re.search(rf"\b{re.escape(alias)}\b", text_norm) for alias in _relation_aliases(relation.predicate or ""))
    return subject_ok and object_ok and predicate_ok


def _target_fact_count(target: RationaleTarget) -> int:
    return len(target.objects) + len(target.relations)


def score_rationale(
    response_text: str,
    target: RationaleTarget,
    *,
    coverage_weight: float = 0.4,
    precision_weight: float = 0.2,
    compactness_weight: float = 0.1,
    sufficiency_weight: float = 0.2,
    necessity_weight: float = 0.1,
) -> RationaleScore:
    response_facts = extract_facts_from_response(response_text)
    response_fact_keys = {_fact_key(fact) for fact in response_facts}
    mentioned_fact_count = len(response_fact_keys)
    target_count = _target_fact_count(target)

    if target_count == 0:
        return RationaleScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    matched_objects = sum(1 for obj_id in target.objects if _contains_object(response_text, obj_id))
    matched_relations = sum(
        1 for relation in target.relations if _contains_relation(response_text, relation, response_fact_keys)
    )
    matched_target_count = matched_objects + matched_relations

    coverage = matched_target_count / target_count
    precision = matched_target_count / mentioned_fact_count if mentioned_fact_count > 0 else 0.0
    precision = max(0.0, min(1.0, precision))
    compactness = min(1.0, target_count / max(mentioned_fact_count, 1))
    sufficiency = 1.0 if matched_target_count == target_count else coverage
    # Phase 1 proxy. Phase 2 will replace this with step-level intervention necessity.
    necessity = coverage

    total_weight = (
        coverage_weight
        + precision_weight
        + compactness_weight
        + sufficiency_weight
        + necessity_weight
    )
    if total_weight <= 0:
        overall = coverage
    else:
        overall = (
            coverage * coverage_weight
            + precision * precision_weight
            + compactness * compactness_weight
            + sufficiency * sufficiency_weight
            + necessity * necessity_weight
        ) / total_weight

    return RationaleScore(
        coverage=max(0.0, min(1.0, coverage)),
        precision=max(0.0, min(1.0, precision)),
        compactness=max(0.0, min(1.0, compactness)),
        sufficiency=max(0.0, min(1.0, sufficiency)),
        necessity=max(0.0, min(1.0, necessity)),
        overall=max(0.0, min(1.0, overall)),
    )
