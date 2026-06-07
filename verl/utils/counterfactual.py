"""
CIT-Faith: Counterfactual Intervention for Causal Faithfulness (CFS)

Implements the prospective (forward-looking) faithfulness check:
  - Extract intervenable elements from the thinking segment
  - Apply semantically-opposite perturbations (entity / coord / relation)
  - Measure whether the policy model's answer changes via prefix-continuation decode

CFS = 1 means all interventions changed the answer (CoT participates in decisions)
CFS = 0 means no intervention changed the answer (CoT is decorative)
"""

import json
import re
import random
from typing import Dict, List, Optional, Tuple

from .answer_normalization import answers_equal


# ──────────────────────────────────────────────────────────────
# Opposite relation lookup table
# ──────────────────────────────────────────────────────────────

OPPOSITE_RELATIONS = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
    "in front of": "behind",
    "behind": "in front of",
    "on top of": "under",
    "under": "on top of",
    "inside": "outside",
    "outside": "inside",
    "near": "far from",
    "far from": "near",
    "next to": "far from",
    "beside": "far from",
    "on": "under",
    "over": "under",
    "beneath": "on top of",
    "to the left of": "to the right of",
    "to the right of": "to the left of",
    "taller than": "shorter than",
    "shorter than": "taller than",
    "larger than": "smaller than",
    "smaller than": "larger than",
    "bigger than": "smaller than",
    "closer than": "farther than",
    "farther than": "closer than",
    "facing": "facing away from",
}


# ──────────────────────────────────────────────────────────────
# Element extraction from thinking segment
# ──────────────────────────────────────────────────────────────

def extract_scene_graph(text: str) -> dict:
    """Extract parsed scene graph JSON from <scene> tag."""
    match = re.search(r"<scene>(.*?)</scene>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1).strip())
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_entities_from_scene(scene: dict) -> List[str]:
    """Extract entity IDs from scene graph objects."""
    objects = scene.get("objects", [])
    return [obj["id"] for obj in objects if isinstance(obj, dict) and "id" in obj]


def extract_coordinates_from_scene(scene: dict) -> List[Tuple[str, List[float]]]:
    """Extract (entity_id, bbox) pairs from scene graph."""
    objects = scene.get("objects", [])
    coords = []
    for obj in objects:
        if isinstance(obj, dict) and "id" in obj and "bbox" in obj:
            bbox = obj["bbox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                coords.append((obj["id"], bbox))
    return coords


def extract_relations_from_scene(scene: dict) -> List[Dict[str, str]]:
    """Extract relationship triples from scene graph."""
    rels = scene.get("relationships", [])
    valid = []
    for r in rels:
        if isinstance(r, dict) and all(k in r for k in ("subject", "predicate", "object")):
            valid.append(r)
    return valid


# ──────────────────────────────────────────────────────────────
# Perturbation functions
# ──────────────────────────────────────────────────────────────

def _find_entity_mentions(thinking: str, entity_id: str) -> List[Tuple[int, int]]:
    """Find all occurrences of entity_id (or its base name) in the thinking text."""
    # Try exact ID match first (e.g., "chair.1")
    positions = []
    base_name = entity_id.split(".")[0].replace("_", " ")
    
    # Match the full ID
    for m in re.finditer(re.escape(entity_id), thinking):
        positions.append((m.start(), m.end()))
    
    # Also match the base name (e.g., "chair") if no full matches found
    if not positions:
        for m in re.finditer(re.escape(base_name), thinking, re.IGNORECASE):
            positions.append((m.start(), m.end()))
    
    return positions


def perturb_entity(thinking: str, scene: dict, entities: List[str]) -> Optional[str]:
    """
    Entity intervention: replace a referenced entity with a semantically different one.
    Prioritizes entities mentioned closer to the conclusion.
    """
    if len(entities) < 2:
        return None

    # Find entities actually mentioned in the thinking segment
    mentioned = []
    for eid in entities:
        positions = _find_entity_mentions(thinking, eid)
        if positions:
            # Use last mention position (closer to conclusion)
            mentioned.append((eid, max(p[0] for p in positions)))

    if not mentioned:
        return None

    # Sort by position descending (prioritize entities near conclusion)
    mentioned.sort(key=lambda x: x[1], reverse=True)
    target_id = mentioned[0][0]

    # Find a replacement entity (different from target)
    candidates = [e for e in entities if e != target_id]
    if not candidates:
        return None

    replacement_id = random.choice(candidates)

    # Perform replacement in the thinking text
    target_base = target_id.split(".")[0].replace("_", " ")
    replacement_base = replacement_id.split(".")[0].replace("_", " ")

    perturbed = thinking.replace(target_id, replacement_id)
    # Also replace base name mentions
    if target_base != replacement_base:
        perturbed = re.sub(
            re.escape(target_base), replacement_base, perturbed, flags=re.IGNORECASE
        )

    return perturbed if perturbed != thinking else None


def perturb_coordinate(thinking: str, coords: List[Tuple[str, List[float]]]) -> Optional[str]:
    """
    Coordinate intervention: swap coordinates between two objects,
    or apply a significant offset (>= 0.3 * dimension).
    """
    if not coords:
        return None

    perturbed = thinking
    # Strategy 1: find two bbox patterns [x1, y1, x2, y2] in the text and swap them
    coord_pattern = re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]')
    matches = list(coord_pattern.finditer(thinking))
    
    if len(matches) >= 2:
        # Swap the first two bounding box occurrences
        m1, m2 = matches[-2], matches[-1]  # prioritize later occurrences (near conclusion)
        bbox1_str = m1.group(0)
        bbox2_str = m2.group(0)
        if bbox1_str != bbox2_str:
            # Use a sentinel to avoid double-replacement
            perturbed = thinking[:m1.start()] + "__CF_SWAP_A__" + thinking[m1.end():m2.start()] + "__CF_SWAP_B__" + thinking[m2.end():]
            perturbed = perturbed.replace("__CF_SWAP_A__", bbox2_str)
            perturbed = perturbed.replace("__CF_SWAP_B__", bbox1_str)
            return perturbed if perturbed != thinking else None

    # Strategy 2: find any numeric coordinate and apply a significant offset
    num_pattern = re.compile(r'\b(\d{2,4})(?:\.\d+)?\b')
    num_matches = list(num_pattern.finditer(thinking))
    if num_matches:
        # Pick the last numeric match (closer to conclusion)
        target_match = num_matches[-1]
        original_val = float(target_match.group(0))
        # Apply a large offset (>= 30% of value or at least 100)
        offset_val = original_val + max(100, original_val * 0.5)
        perturbed = (
            thinking[:target_match.start()]
            + f"{offset_val:.0f}"
            + thinking[target_match.end():]
        )
        return perturbed if perturbed != thinking else None

    return None


def perturb_relation(thinking: str, relations: List[Dict[str, str]]) -> Optional[str]:
    """
    Relation intervention: replace a spatial relation with its semantic opposite.
    E.g., "left of" → "right of", "above" → "below".
    """
    perturbed = thinking
    applied = False

    # Find relations that appear in the thinking text and have known opposites
    for rel in relations:
        predicate = rel.get("predicate", "").lower().strip()
        # Try to find this predicate in the thinking text
        opposite = OPPOSITE_RELATIONS.get(predicate)
        if not opposite:
            continue

        # Find and replace (case-insensitive, word boundary, preserve surrounding text)
        pattern = re.compile(r'\b' + re.escape(predicate) + r'\b', re.IGNORECASE)
        matches = list(pattern.finditer(perturbed))
        if matches:
            # Replace the last occurrence (closer to conclusion)
            m = matches[-1]
            perturbed = perturbed[:m.start()] + opposite + perturbed[m.end():]
            applied = True
            break  # One relation perturbation per intervention

    if not applied:
        # Try common spatial words not in the relation list
        for word, opp in OPPOSITE_RELATIONS.items():
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(perturbed))
            if matches:
                m = matches[-1]
                perturbed = perturbed[:m.start()] + opp + perturbed[m.end():]
                applied = True
                break

    return perturbed if applied else None


# ──────────────────────────────────────────────────────────────
# Main CFS computation
# ──────────────────────────────────────────────────────────────

def generate_counterfactual_inputs(
    response_text: str,
) -> List[Tuple[str, str]]:
    """
    Generate up to 3 counterfactual versions of the thinking segment.
    
    Args:
        response_text: Full rollout text containing <observe>, <scene>, <think>, <answer>
    
    Returns:
        List of (intervention_type, perturbed_response_text) tuples.
        Each perturbed text has the <think> segment modified but everything else intact.
    """
    scene = extract_scene_graph(response_text)
    think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE)
    if not think_match or not scene:
        return []

    thinking = think_match.group(1)  # Don't strip - preserve exact positions for replacement
    entities = extract_entities_from_scene(scene)
    coords = extract_coordinates_from_scene(scene)
    relations = extract_relations_from_scene(scene)

    interventions = []

    # 1. Entity intervention
    perturbed_think = perturb_entity(thinking, scene, entities)
    if perturbed_think is not None:
        perturbed_full = (
            response_text[:think_match.start(1)]
            + perturbed_think
            + response_text[think_match.end(1):]
        )
        interventions.append(("entity", perturbed_full))

    # 2. Coordinate intervention
    perturbed_think = perturb_coordinate(thinking, coords)
    if perturbed_think is not None:
        perturbed_full = (
            response_text[:think_match.start(1)]
            + perturbed_think
            + response_text[think_match.end(1):]
        )
        interventions.append(("coord", perturbed_full))

    # 3. Relation intervention
    perturbed_think = perturb_relation(thinking, relations)
    if perturbed_think is not None:
        perturbed_full = (
            response_text[:think_match.start(1)]
            + perturbed_think
            + response_text[think_match.end(1):]
        )
        interventions.append(("relation", perturbed_full))

    return interventions


def build_prefix_for_continuation(
    perturbed_response: str,
    answer_start_token: str = "<answer>",
) -> str:
    """
    Build the prefix for continuation decode.
    
    Takes everything up to and including the <answer> tag start,
    so the model only needs to generate the answer content.
    """
    pattern = re.compile(re.escape(answer_start_token), re.IGNORECASE)
    match = pattern.search(perturbed_response)
    if match is None:
        return perturbed_response
    return perturbed_response[:match.end()]


def compute_cfs_score(
    original_answer: str,
    counterfactual_answers: List[Optional[str]],
) -> float:
    """
    Compute the Causal Faithfulness Score (CFS) for a single rollout.
    
    CFS = (number of interventions that changed the answer) / (number of valid interventions)
    
    Args:
        original_answer: The original answer from the rollout
        counterfactual_answers: Answers from each counterfactual continuation.
                               None entries are invalid interventions (skipped).
    
    Returns:
        CFS score in [0, 1]. Returns -1 if no valid interventions.
    """
    valid_interventions = [a for a in counterfactual_answers if a is not None]
    if not valid_interventions:
        return -1.0  # Sentinel: no valid interventions, modulation defaults to g=1

    changed_count = sum(
        1 for a in valid_interventions
        if not answers_equal(a, original_answer)
    )

    return changed_count / len(valid_interventions)


def compute_cfs_batch(
    response_texts: List[str],
    original_answers: List[str],
    continuation_fn=None,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Compute CFS for a batch of rollouts.
    
    This is the main entry point called from the training loop.
    
    Args:
        response_texts: List of full rollout texts
        original_answers: List of original answers extracted from rollouts
        continuation_fn: Callable that takes a list of prefix strings and returns
                        a list of generated answer strings. If None, returns defaults.
    
    Returns:
        cfs_scores: List of CFS scores (one per rollout)
        detailed_metrics: Dict with per-type CFS breakdowns
    """
    all_cfs_scores = []
    type_scores = {"entity": [], "coord": [], "relation": []}

    # Early return if no continuation function is wired
    if continuation_fn is None:
        return [-1.0] * len(response_texts), type_scores

    for i, (resp, orig_ans) in enumerate(zip(response_texts, original_answers)):
        interventions = generate_counterfactual_inputs(resp)

        if not interventions:
            all_cfs_scores.append(-1.0)
            continue

        # Build prefixes for continuation
        prefixes = [build_prefix_for_continuation(perturbed) for _, perturbed in interventions]
        types = [t for t, _ in interventions]

        # Run continuation decode
        cf_answers = continuation_fn(prefixes)

        # Compute per-type and overall CFS
        valid_answers = []
        for j, (itype, cf_ans) in enumerate(zip(types, cf_answers)):
            if cf_ans is not None:
                valid_answers.append(cf_ans)
                changed = not answers_equal(cf_ans, orig_ans)
                type_scores[itype].append(float(changed))

        cfs = compute_cfs_score(orig_ans, cf_answers)
        all_cfs_scores.append(cfs)

    return all_cfs_scores, type_scores

def compute_cfs_batch_indexed(
    response_texts: List[str],
    original_answers: List[str],
    continuation_helper=None,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Compute CFS for a batch using an indexed continuation helper.
    
    Unlike compute_cfs_batch which takes a stateless continuation_fn,
    this version uses a helper object that tracks the rollout index,
    so it can prepend the correct prompt ids for each rollout.
    
    Args:
        response_texts: List of full rollout texts
        original_answers: List of original answers
        continuation_helper: Object with set_rollout_index(i) and __call__(prefixes)
    
    Returns:
        cfs_scores, type_metrics
    """
    all_cfs_scores = []
    type_scores = {"entity": [], "coord": [], "relation": []}

    if continuation_helper is None:
        return [-1.0] * len(response_texts), type_scores

    for i, (resp, orig_ans) in enumerate(zip(response_texts, original_answers)):
        interventions = generate_counterfactual_inputs(resp)

        if not interventions:
            all_cfs_scores.append(-1.0)
            continue

        # Tell the helper which rollout we're processing
        continuation_helper.set_rollout_index(i)

        # Build prefixes for continuation
        prefixes = [build_prefix_for_continuation(perturbed) for _, perturbed in interventions]
        types = [t for t, _ in interventions]

        # Run continuation decode
        cf_answers = continuation_helper(prefixes)

        # Compute per-type and overall CFS
        for j, (itype, cf_ans) in enumerate(zip(types, cf_answers)):
            if cf_ans is not None:
                changed = not answers_equal(cf_ans, orig_ans)
                type_scores[itype].append(float(changed))

        cfs = compute_cfs_score(orig_ans, cf_answers)
        all_cfs_scores.append(cfs)

    return all_cfs_scores, type_scores
