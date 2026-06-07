"""
CIT-Faith: Frozen Reviewer Model (LLM Judge)

Implements the independent LLM judge for evaluating:
  - Self-Consistency (SC): introspective check on thinking segment T
  - Perception-Thinking Coherence (PR): retrospective check P -> T

The reviewer model is parameter-frozen throughout training, ensuring
evaluation standards do not drift with the policy model.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────
# Prompt templates for the reviewer model
# ──────────────────────────────────────────────────────────────

SC_PROMPT_TEMPLATE = """You are evaluating the internal logical consistency of a spatial reasoning process.

Given the following thinking segment from a spatial reasoning task, evaluate it on these 4 dimensions (each 0.0 to 1.0):

1. **Spatial assertion consistency**: Are spatial relationship claims consistent throughout? (e.g., no "A is left of B" then later "A is right of B" without correction)
2. **Quantity consistency**: Are object counts and attributes consistent throughout?
3. **Transitivity consistency**: Are transitive spatial inferences logically valid? (e.g., "A left of B, B left of C" should imply "A left of C", not "A right of C")
4. **Conclusion consistency**: Does the final conclusion logically follow from the preceding analysis?

If the thinking segment contains explicit self-corrections, contradictions before the correction point should NOT be penalized.

Thinking segment:
<think>
{thinking}
</think>

Respond ONLY with a JSON object (no markdown, no explanation):
{{"spatial_assertion": <float>, "quantity": <float>, "transitivity": <float>, "conclusion": <float>}}"""


PR_PROMPT_TEMPLATE = """You are evaluating whether a spatial reasoning process is faithful to the perceived scene information.

Given the scene graph (structured perception output) and the thinking segment, detect violations in these 4 categories (each 0 or 1, where 1 = violation found):

1. **Direct contradiction**: The thinking segment makes spatial assertions that directly contradict the scene graph triples.
2. **Quantity mismatch**: The thinking segment uses object counts inconsistent with the scene graph.
3. **Identity confusion**: The thinking segment attributes properties of one entity to another.
4. **Fabrication**: The thinking segment introduces spatial information that does not exist in the scene graph and cannot be derived from it.

Legitimate derived inferences (e.g., inferring "A left of C" from "A left of B" and "B left of C") are NOT violations.

Scene graph:
<scene>
{scene}
</scene>

Thinking segment:
<think>
{thinking}
</think>

Respond ONLY with a JSON object (no markdown, no explanation):
{{"direct_contradiction": <0 or 1>, "quantity_mismatch": <0 or 1>, "identity_confusion": <0 or 1>, "fabrication": <0 or 1>}}"""


# Violation weights for PR score (from paper Appendix A)
PR_VIOLATION_WEIGHTS = {
    "direct_contradiction": 0.35,
    "quantity_mismatch": 0.20,
    "identity_confusion": 0.25,
    "fabrication": 0.20,
}


# ──────────────────────────────────────────────────────────────
# Structured output extraction helpers
# ──────────────────────────────────────────────────────────────

def extract_think(text: str) -> str:
    """Extract the <think>...</think> segment from a rollout."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_scene_text(text: str) -> str:
    """Extract the raw <scene>...</scene> text from a rollout."""
    match = re.search(r"<scene>(.*?)</scene>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_answer_text(text: str) -> str:
    """Extract the <answer>...</answer> text from a rollout."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ──────────────────────────────────────────────────────────────
# Reviewer Model class
# ──────────────────────────────────────────────────────────────

class ReviewerModel:
    """
    Frozen reviewer model (LLM judge) for CIT-Faith.
    
    Uses a quantized language model (e.g., Qwen2.5-7B-Instruct-AWQ)
    deployed via vLLM for deterministic evaluation of CoT faithfulness.
    
    Parameters are NEVER updated during training.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        temperature: float = 0.0,
        max_tokens: int = 256,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.15,
    ):
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._engine = None
        self._tp_size = tensor_parallel_size
        self._gpu_mem = gpu_memory_utilization

    def _lazy_init(self):
        """Lazy-initialize vLLM engine on first use."""
        if self._engine is not None:
            return
        try:
            from vllm import LLM, SamplingParams
            self._engine = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=self._tp_size,
                gpu_memory_utilization=self._gpu_mem,
                trust_remote_code=True,
                quantization="awq",
                enforce_eager=True,
                max_model_len=4096,
            )
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            print(f"[CIT-Faith] Reviewer model loaded: {self.model_name_or_path}")
        except Exception as e:
            print(f"[CIT-Faith] WARNING: Could not load vLLM reviewer: {e}")
            print("[CIT-Faith] Falling back to dummy reviewer (returns default scores).")
            self._engine = "dummy"

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        self._lazy_init()
        if self._engine == "dummy":
            return ["{}"] * len(prompts)
        outputs = self._engine.generate(prompts, self._sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def _parse_json(self, text: str, default: dict) -> dict:
        """Parse JSON from reviewer output with fallback."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return default

    # ── Public API ──

    def evaluate_self_consistency(self, thinking_segments: List[str]) -> List[float]:
        """
        Evaluate self-consistency (SC) for a batch of thinking segments.
        
        Returns a list of SC scores in [0, 1], where 1 = fully consistent.
        SC = mean of 4 dimension scores.
        """
        prompts = [
            SC_PROMPT_TEMPLATE.format(thinking=t) for t in thinking_segments
        ]
        responses = self._generate_batch(prompts)

        sc_scores = []
        default = {"spatial_assertion": 0.5, "quantity": 0.5, "transitivity": 0.5, "conclusion": 0.5}
        for resp in responses:
            parsed = self._parse_json(resp, default)
            dim_keys = ["spatial_assertion", "quantity", "transitivity", "conclusion"]
            dims = []
            for k in dim_keys:
                try:
                    v = float(parsed.get(k, 0.5))
                except (ValueError, TypeError):
                    v = 0.5
                dims.append(max(0.0, min(1.0, v)))
            sc_scores.append(sum(dims) / 4.0)

        return sc_scores

    def evaluate_coherence(
        self, scene_texts: List[str], thinking_segments: List[str]
    ) -> List[float]:
        """
        Evaluate perception-thinking coherence (PR) for a batch.
        
        Returns a list of PR scores in [0, 1], where 1 = fully coherent.
        PR = 1 - weighted sum of violations.
        """
        prompts = [
            PR_PROMPT_TEMPLATE.format(scene=s, thinking=t)
            for s, t in zip(scene_texts, thinking_segments)
        ]
        responses = self._generate_batch(prompts)

        pr_scores = []
        default = {"direct_contradiction": 0, "quantity_mismatch": 0, "identity_confusion": 0, "fabrication": 0}
        for resp in responses:
            parsed = self._parse_json(resp, default)
            weighted_violation = 0.0
            for k, w in PR_VIOLATION_WEIGHTS.items():
                try:
                    v = float(parsed.get(k, 0))
                except (ValueError, TypeError):
                    v = 0.0
                weighted_violation += w * v
            pr_score = max(0.0, min(1.0, 1.0 - weighted_violation))
            pr_scores.append(pr_score)

        return pr_scores

    def evaluate_batch(
        self,
        response_texts: List[str],
    ) -> Tuple[List[float], List[float]]:
        """
        Evaluate both SC and PR for a batch of full rollout texts.
        
        Extracts <think> and <scene> segments, then runs both evaluations.
        
        Returns:
            sc_scores: List of self-consistency scores
            pr_scores: List of perception-thinking coherence scores
        """
        thinking_segments = [extract_think(t) for t in response_texts]
        scene_texts = [extract_scene_text(t) for t in response_texts]

        # Filter out empty segments (format failures)
        valid_mask = [
            bool(think) and bool(scene)
            for think, scene in zip(thinking_segments, scene_texts)
        ]

        # For valid entries, evaluate; for invalid, return defaults
        valid_thinks = [t for t, v in zip(thinking_segments, valid_mask) if v]
        valid_scenes = [s for s, v in zip(scene_texts, valid_mask) if v]

        if valid_thinks:
            valid_sc = self.evaluate_self_consistency(valid_thinks)
            valid_pr = self.evaluate_coherence(valid_scenes, valid_thinks)
        else:
            valid_sc, valid_pr = [], []

        # Reconstruct full lists with defaults for invalid entries
        sc_scores, pr_scores = [], []
        valid_idx = 0
        for v in valid_mask:
            if v:
                sc_scores.append(valid_sc[valid_idx])
                pr_scores.append(valid_pr[valid_idx])
                valid_idx += 1
            else:
                # Invalid format: neutral defaults (don't penalize format failures
                # through faithfulness channel - format_reward handles that)
                sc_scores.append(0.5)
                pr_scores.append(0.5)

        return sc_scores, pr_scores
