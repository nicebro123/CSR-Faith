import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .answer_normalization import answers_equal
from .causal_rationale import RationaleTarget, SpatialFact, base_name, refine_name
from .counterfactual import OPPOSITE_RELATIONS, build_prefix_for_continuation


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_NUMBERED_STEP_RE = re.compile(r"(?:^|\n|\s)(?:\d+[\).]|step\s+\d+[:.)])\s*", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass
class StepIntervention:
    rollout_index: int
    step_index: int
    intervention_type: str
    original_step: str
    perturbed_response: str


@dataclass
class StepCausalScore:
    step_scores: List[float]
    mean: float
    valid_ratio: float
    max_score: float


def _thinking_span(response_text: str) -> Optional[Tuple[re.Match, str]]:
    match = _THINK_RE.search(str(response_text or ""))
    if not match:
        return None
    thinking = match.group(1)
    return match, thinking


def _split_thinking_step_spans(thinking: str) -> List[Tuple[str, int, int]]:
    marker_matches = list(_NUMBERED_STEP_RE.finditer(thinking))
    spans: List[Tuple[str, int, int]] = []
    if len(marker_matches) > 1 or (marker_matches and marker_matches[0].start() == 0):
        for idx, marker in enumerate(marker_matches):
            start = marker.end()
            end = marker_matches[idx + 1].start() if idx + 1 < len(marker_matches) else len(thinking)
            step = thinking[start:end].strip()
            if step:
                stripped_start = start + len(thinking[start:end]) - len(thinking[start:end].lstrip())
                stripped_end = end - (len(thinking[start:end]) - len(thinking[start:end].rstrip()))
                spans.append((step, stripped_start, stripped_end))
        if spans:
            return spans

    for match in _SENTENCE_RE.finditer(thinking):
        step = match.group(0).strip()
        if not step:
            continue
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        end = match.end() - (len(match.group(0)) - len(match.group(0).rstrip()))
        spans.append((step, start, end))
    return spans


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


def _target_relevance(step: str, target: RationaleTarget) -> int:
    score = 0
    for obj_id in target.objects:
        score += int(_contains_object(step, obj_id))
    step_norm = refine_name(step)
    for rel in target.relations:
        subject_hit = _contains_object(step, rel.subject or "")
        object_hit = _contains_object(step, rel.object or "")
        predicate_hit = bool(rel.predicate and re.search(rf"\b{re.escape(refine_name(rel.predicate))}\b", step_norm))
        score += int(subject_hit and object_hit) + int(predicate_hit)
    return score


def _replace_step(response_text: str, think_match: re.Match, start: int, end: int, replacement: str) -> str:
    abs_start = think_match.start(1) + start
    abs_end = think_match.start(1) + end
    return response_text[:abs_start] + replacement + response_text[abs_end:]


def _flip_relation(step: str, target: RationaleTarget) -> Optional[str]:
    candidates = [rel.predicate for rel in target.relations if rel.predicate]
    candidates.extend(OPPOSITE_RELATIONS.keys())
    for predicate in candidates:
        pred = refine_name(predicate)
        opposite = OPPOSITE_RELATIONS.get(pred)
        if not opposite:
            continue
        pattern = re.compile(rf"\b{re.escape(pred)}\b", re.IGNORECASE)
        matches = list(pattern.finditer(step))
        if not matches:
            continue
        match = matches[-1]
        return step[:match.start()] + opposite + step[match.end():]
    return None


def _swap_entity(step: str, target: RationaleTarget) -> Optional[str]:
    if len(target.objects) < 2:
        return None
    for source_obj in target.objects:
        if not _contains_object(step, source_obj):
            continue
        replacement = next((obj for obj in target.objects if obj != source_obj), None)
        if replacement is None:
            return None
        source_base = base_name(source_obj)
        replacement_base = base_name(replacement)

        source_variants = [
            str(source_obj or ""),
            refine_name(source_obj),
            source_base,
            source_base.replace(" ", "_") if source_base else "",
        ]
        replacement_variants = {
            str(source_obj or ""): str(replacement or ""),
            refine_name(source_obj): refine_name(replacement),
            source_base: replacement_base,
            source_base.replace(" ", "_") if source_base else "": replacement_base.replace(" ", "_"),
        }

        for needle in source_variants:
            repl = replacement_variants.get(needle, replacement_base)
            if not needle or needle == repl:
                continue
            pattern = re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)
            if pattern.search(step):
                return pattern.sub(repl, step, count=1)
    return None


def _mask_step(step: str) -> str:
    return "The cited spatial evidence is unavailable."


def _build_step_intervention(
    response_text: str,
    think_match: re.Match,
    step: str,
    start: int,
    end: int,
    rollout_index: int,
    step_index: int,
    intervention_type: str,
    replacement_step: str,
) -> StepIntervention:
    return StepIntervention(
        rollout_index=rollout_index,
        step_index=step_index,
        intervention_type=intervention_type,
        original_step=step,
        perturbed_response=_replace_step(response_text, think_match, start, end, replacement_step),
    )


def generate_step_interventions(
    response_text: str,
    target: RationaleTarget,
    *,
    max_steps: int = 6,
    max_interventions_per_step: int = 1,
    rollout_index: int = -1,
) -> List[StepIntervention]:
    """Generate deterministic one-step counterfactual interventions."""
    span = _thinking_span(response_text)
    if span is None or max_steps <= 0 or max_interventions_per_step <= 0:
        return []

    think_match, thinking = span
    step_spans = _split_thinking_step_spans(thinking)
    if not step_spans:
        return []

    indexed_steps = list(enumerate(step_spans))
    relevant_steps = [
        (idx, step_span)
        for idx, step_span in indexed_steps
        if _target_relevance(step_span[0], target) > 0
    ]
    selected_steps = (relevant_steps or indexed_steps)[:max_steps]

    interventions: List[StepIntervention] = []
    for step_index, (step, start, end) in selected_steps:
        replacements: List[Tuple[str, str]] = []

        flipped = _flip_relation(step, target)
        if flipped and flipped != step:
            replacements.append(("relation", flipped))

        swapped = _swap_entity(step, target)
        if swapped and swapped != step:
            replacements.append(("entity", swapped))

        masked = _mask_step(step)
        if masked != step:
            replacements.append(("mask", masked))

        for intervention_type, replacement_step in replacements[:max_interventions_per_step]:
            interventions.append(
                _build_step_intervention(
                    response_text=response_text,
                    think_match=think_match,
                    step=step,
                    start=start,
                    end=end,
                    rollout_index=rollout_index,
                    step_index=step_index,
                    intervention_type=intervention_type,
                    replacement_step=replacement_step,
                )
            )

    return interventions


def build_prefixes_for_step_interventions(interventions: List[StepIntervention]) -> List[str]:
    return [build_prefix_for_continuation(intervention.perturbed_response) for intervention in interventions]


def compute_step_causal_score(
    original_answer: str,
    interventions: List[StepIntervention],
    counterfactual_answers: List[Optional[str]],
) -> StepCausalScore:
    if not interventions:
        return StepCausalScore(step_scores=[], mean=-1.0, valid_ratio=0.0, max_score=0.0)

    valid_pairs = [
        (intervention, answer)
        for intervention, answer in zip(interventions, counterfactual_answers)
        if answer is not None
    ]
    if not valid_pairs:
        return StepCausalScore(step_scores=[], mean=-1.0, valid_ratio=0.0, max_score=0.0)

    by_step = {}
    for intervention, answer in valid_pairs:
        changed = float(not answers_equal(answer, original_answer))
        by_step.setdefault(intervention.step_index, []).append(changed)

    step_scores = [max(scores) for _, scores in sorted(by_step.items())]
    mean_score = sum(step_scores) / len(step_scores) if step_scores else -1.0
    valid_ratio = len(valid_pairs) / max(len(interventions), 1)
    max_score = max(step_scores) if step_scores else 0.0
    return StepCausalScore(
        step_scores=step_scores,
        mean=mean_score,
        valid_ratio=valid_ratio,
        max_score=max_score,
    )
