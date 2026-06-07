# Causal Spatial Rationales Development Plan

## Goal

Build the next version of CIT-Faith around this claim:

> A faithful multimodal spatial chain of thought should cite visual-spatial facts that are necessary and sufficient for the answer, and each reasoning step should have measurable causal effect under counterfactual intervention.

The implementation must keep the current dataset unchanged. All supervision should be derived from fields already present in the batch:

- `problem`: prompt text, including image size for `spatial_sgg`
- `ground_truth`: answer string selected by `data.answer_key`
- `ground_truth` embedded `<scene>...</scene>` and `<answer>...</answer>` tags
- generated response tags: `<observe>`, `<scene>`, `<think>`, `<answer>`

This should extend the existing CIT-Faith path rather than replace the whole trainer.

## Existing Code Anchors

- Training entry: `verl/trainer/main.py`
- Training loop: `verl/trainer/ray_trainer.py`
- Main CIT-Faith hook: `ray_trainer.py`, inside the `adv` timer after reward and KL preparation
- Current advantage implementation: `verl/trainer/core_algos.py`
- Current reviewer: `verl/utils/reviewer.py`
- Current counterfactual intervention code: `verl/utils/counterfactual.py`
- Current prefix-continuation worker path:
  - `verl/workers/fsdp_workers.py::generate_continuations`
  - `verl/workers/rollout/vllm_rollout_spmd.py::generate_continuations`
- Current spatial task reward: `verl/utils/reward_score/spatial_sgg.py`
- Current training script: `scripts/citfaith_7b_grpo.sh`

## Proposed Name

Use `CSR-Faith` internally:

- CSR = Causal Spatial Rationales
- Keep `CIT-Faith` as the base project name
- New script: `scripts/csrfaith_7b_grpo.sh`
- New config gate: `algorithm.enable_csrfaith`

## High-Level Design

CSR-Faith has three components.

1. Grounded causal evidence extraction
   - Derive a pseudo-label set of necessary visual-spatial facts from the existing GT scene graph.
   - Output object IDs and relation triples that are likely needed to answer the question.

2. Minimal sufficient rationale scoring
   - Reward generated `<think>` for covering necessary facts.
   - Penalize irrelevant or hallucinated spatial facts.
   - Prefer compact reasoning over decorative CoT.

3. Step-level causal credit
   - Split `<think>` into reasoning steps.
   - Intervene on one step at a time.
   - Use the current policy model's prefix-continuation decode to test whether the final answer changes.
   - Assign causal credit to individual steps instead of treating the whole CoT as one block.

## New Files

### `verl/utils/causal_rationale.py`

Main module for deriving and scoring spatial rationales.

Recommended public API:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SpatialFact:
    fact_type: str  # "object" or "relation"
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
```

Functions:

```python
def extract_gt_scene_and_answer(ground_truth: str) -> Tuple[dict, str]:
    """Parse <scene> and <answer> from the existing ground_truth string."""


def build_causal_rationale_target(
    problem: str,
    gt_scene: dict,
    gt_answer: str,
    *,
    max_relations: int = 4,
    max_objects: int = 6,
) -> RationaleTarget:
    """Generate pseudo-label evidence without changing the dataset."""


def extract_facts_from_response(response_text: str) -> List[SpatialFact]:
    """Extract object and relation mentions from generated <think> and <scene>."""


def score_rationale(
    response_text: str,
    target: RationaleTarget,
) -> RationaleScore:
    """Compute coverage, precision, compactness, sufficiency, and necessity proxies."""


def split_thinking_steps(response_text: str) -> List[str]:
    """Split <think> into sentence-level or numbered-step units."""
```

Initial target extraction should be heuristic and deterministic:

- Parse all GT objects and GT relations from `<scene>`.
- Prefer relations whose `subject`, `object`, or `predicate` lexically overlap with `problem` or `gt_answer`.
- Include endpoint objects for every selected relation.
- If no relation matches, fall back to objects whose base names overlap with `problem` or `gt_answer`.
- If still empty, include top-k GT relations and their endpoint objects, with lower confidence.

This gives a working baseline without a new model. A later paper-quality version can add a frozen graph-only LLM verifier, but the first implementation should avoid another dependency.

### `verl/utils/step_causal.py`

Step-level intervention and CFS computation.

Recommended public API:

```python
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
```

Functions:

```python
def generate_step_interventions(
    response_text: str,
    target: RationaleTarget,
    *,
    max_steps: int = 6,
    max_interventions_per_step: int = 1,
) -> List[StepIntervention]:
    """Perturb one reasoning step at a time."""


def build_prefixes_for_step_interventions(
    interventions: List[StepIntervention],
) -> List[str]:
    """Return text prefixes ending at <answer> for continuation decode."""


def compute_step_causal_score(
    original_answer: str,
    interventions: List[StepIntervention],
    counterfactual_answers: List[Optional[str]],
) -> StepCausalScore:
    """Return step-level causal effect scores."""
```

Interventions should reuse the logic in `counterfactual.py`:

- entity swap
- relation antonym flip
- coordinate perturbation
- deletion or masking of a cited necessary fact

Important change from current CFS:

- Current CFS mutates the whole `<think>` and returns one scalar.
- CSR-Faith mutates one step at a time and returns a vector, then aggregates it.

### `verl/utils/answer_normalization.py`

The current CFS compares raw answer strings. That is too brittle.

Add:

```python
def normalize_answer(text: str) -> str:
    """Remove tags, punctuation, casing, extra explanation, and option prefixes."""


def answers_equal(a: str, b: str) -> bool:
    """Normalized exact match with small alias handling."""
```

Use this in both `counterfactual.py` and `step_causal.py`.

## Modified Files

### `verl/trainer/config.py`

Add fields to `AlgorithmConfig`:

```python
# CSR-Faith
enable_csrfaith: bool = False
csr_target_max_relations: int = 4
csr_target_max_objects: int = 6
csr_coverage_weight: float = 0.4
csr_precision_weight: float = 0.2
csr_compactness_weight: float = 0.1
csr_sufficiency_weight: float = 0.2
csr_necessity_weight: float = 0.1
csr_step_cfs_alpha: float = 0.1
csr_max_steps: int = 6
csr_max_step_interventions: int = 1
tau_coverage: float = 0.7
tau_step_cfs: float = 0.5
lambda_coverage_init: float = 0.0
lambda_step_cfs_init: float = 0.0
```

Keep `enable_citfaith` and `enable_csrfaith` independent at first. Later they can be merged.

### `verl/trainer/core_algos.py`

Add:

```python
def compute_csrfaith_grpo_advantage(
    token_level_rewards,
    response_mask,
    index,
    rationale_scores,
    step_cfs_scores,
    lambda_coverage,
    lambda_step_cfs,
    alpha,
    eps=1e-6,
):
    ...
```

Recommended formula:

```text
A_task = group_zscore(task_reward)
A_rat  = group_zscore(rationale_overall)
A_step = group_zscore(step_cfs_mean)

A_base = A_task + lambda_coverage * A_rat + lambda_step_cfs * A_step
g_step = alpha + (1 - alpha) * step_cfs_mean
A_csr  = A_base * g_step
```

Add:

```python
def update_csr_lagrangian_multipliers(
    lambda_coverage,
    lambda_step_cfs,
    batch_coverage_mean,
    batch_step_cfs_mean,
    tau_coverage,
    tau_step_cfs,
    eta,
):
    ...
```

### `verl/trainer/ray_trainer.py`

Add CSR-Faith initialization next to the existing CIT-Faith initialization:

```python
self.csrfaith_enabled = config.algorithm.enable_csrfaith
self.lambda_coverage = config.algorithm.lambda_coverage_init
self.lambda_step_cfs = config.algorithm.lambda_step_cfs_init
```

Inside the existing `adv` section, after reward/KL and before `compute_advantage`, add:

1. Decode generated responses once.
2. Parse `ground_truth` and `problem` from `batch.non_tensor_batch`.
3. Build `RationaleTarget` for each rollout.
4. Compute `RationaleScore` for each response.
5. Generate step interventions.
6. Reuse `_ContinuationHelper` and `actor_rollout_wg.generate_continuations`.
7. Compute `StepCausalScore`.
8. Convert scores to tensors.
9. Pass tensors into `compute_csrfaith_grpo_advantage`.
10. Log metrics.

Important ordering issue:

- `_balance_batch()` reorders the batch before advantage computation.
- CSR-Faith scores must be computed after `_balance_batch()` or must be reordered with the batch.
- The simplest first implementation is to compute CSR-Faith after `_balance_batch()`, same as current CIT-Faith.

New metrics:

```text
csr/target_confidence_mean
csr/rationale_coverage_mean
csr/rationale_precision_mean
csr/rationale_compactness_mean
csr/rationale_sufficiency_mean
csr/rationale_necessity_mean
csr/rationale_overall_mean
csr/step_cfs_mean
csr/step_cfs_valid_ratio
csr/lambda_coverage
csr/lambda_step_cfs
```

Checkpoint:

- Extend `citfaith_state.pt` or create `csrfaith_state.pt`.
- Recommended: create `csrfaith_state.pt` to keep the new experiment separable.

Fields:

```python
{
    "lambda_coverage": self.lambda_coverage,
    "lambda_step_cfs": self.lambda_step_cfs,
}
```

### `verl/utils/counterfactual.py`

Small refactor:

- Import `answers_equal` from `answer_normalization.py`.
- Replace raw string equality in `compute_cfs_score`.
- Expose common perturbation helpers so `step_causal.py` can reuse them.

### `scripts/csrfaith_7b_grpo.sh`

Copy `scripts/citfaith_7b_grpo.sh` and change:

```bash
trainer.experiment_name=csrfaith_7B
trainer.save_checkpoint_path=ckpts/csrfaith_7B
algorithm.enable_csrfaith=True
algorithm.enable_citfaith=False
algorithm.csr_target_max_relations=4
algorithm.csr_max_steps=6
algorithm.csr_step_cfs_alpha=0.1
algorithm.tau_coverage=0.7
algorithm.tau_step_cfs=0.5
```

Later ablations should run:

- baseline GRPO
- CIT-Faith current full
- CSR rationale only
- CSR step CFS only
- full CSR-Faith
- CIT-Faith + CSR-Faith combined

## Scoring Details

### Coverage

Measures whether generated reasoning mentions the target necessary facts.

```text
coverage = matched_target_facts / target_facts
```

Matching should be relaxed:

- object ID exact match
- object base-name match, e.g. `chair.1` -> `chair`
- relation predicate synonym or opposite lookup
- relation endpoint object overlap

### Precision

Measures whether generated reasoning avoids irrelevant or hallucinated spatial facts.

```text
precision = mentioned_target_facts / mentioned_spatial_facts
```

If no spatial facts are mentioned, precision should be `0.0`.

### Compactness

Penalizes decorative long CoT.

```text
compactness = min(1.0, target_fact_count / max(mentioned_fact_count, 1))
```

Do not use raw token length as the main compactness term. Long reasoning can be valid if it cites dense evidence.

### Sufficiency Proxy

First implementation:

```text
sufficiency = 1.0 if all answer-linked target objects/relations are covered else coverage
```

Paper-quality implementation:

- Use a graph-only verifier to answer from the selected evidence subgraph.
- If verifier predicts the original answer from the selected subgraph, evidence is sufficient.

### Necessity Proxy

First implementation:

```text
necessity = fraction of target facts whose perturbation changes the answer in step-level CFS
```

Paper-quality implementation:

- Remove each selected target relation/object from the GT graph.
- Ask graph-only verifier whether the answer changes.
- Keep only facts with positive causal effect as necessary pseudo-labels.

## Step-Level Intervention Data Flow

For each rollout:

1. `response_text = decode(responses[i])`
2. `steps = split_thinking_steps(response_text)`
3. `target = build_causal_rationale_target(problem[i], gt_scene[i], gt_answer[i])`
4. `interventions = generate_step_interventions(response_text, target)`
5. `prefix = response_with_one_step_perturbed[: "<answer>" included]`
6. `_ContinuationHelper(prefixes)` sends prompt + prefix to vLLM
7. Decode counterfactual answer
8. Compare with original answer using `answers_equal`
9. Aggregate step scores

Use a hard cap:

```text
max prefixes per rollout = csr_max_steps * csr_max_step_interventions
```

With rollout batch size 512 and rollout.n 6, unrestricted step interventions would be too expensive. Start with:

```text
csr_max_steps = 3
csr_max_step_interventions = 1
cf_max_tokens = 32 or 50
```

Then increase after profiling.

## Implementation Phases

### Phase 1: Deterministic CSR Scoring, No Extra Generation

Goal: add causal-rationale metrics and advantage signal without step continuation.

Tasks:

1. Add `causal_rationale.py`.
2. Add config fields.
3. In `ray_trainer.py`, compute `RationaleTarget` and `RationaleScore`.
4. Add `compute_csrfaith_grpo_advantage` using `rationale_overall`.
5. Log CSR metrics.
6. Add `scripts/csrfaith_7b_grpo.sh`.

Expected output:

- Training still runs with roughly current speed.
- `csr/rationale_coverage_mean` and `csr/rationale_precision_mean` appear in logs.

### Phase 2: Step-Level CFS

Goal: add causal credit for individual reasoning steps.

Tasks:

1. Add `step_causal.py`.
2. Refactor answer normalization.
3. Reuse existing `_ContinuationHelper`.
4. Add step CFS tensors to advantage.
5. Add `csr/step_cfs_*` metrics.
6. Add checkpoint state for CSR multipliers.

Expected output:

- Step CFS valid ratio should be above 0.4 initially.
- If valid ratio is low, intervention generation or answer extraction is too brittle.

### Phase 3: Pseudo-Label Quality Upgrade

Goal: make the "minimal sufficient evidence" claim stronger for paper submission.

Tasks:

1. Add optional graph-only verifier using the existing reviewer model or a smaller frozen model.
2. Cache verifier outputs by hash of `(problem, gt_scene, gt_answer)`.
3. Filter target facts by necessity and sufficiency.
4. Add ablation:
   - heuristic target only
   - verifier target only
   - verifier target + step CFS

Recommended cache path:

```text
cache/csr_targets/{dataset_name}/{split}.jsonl
```

The cache is derived from the same dataset, so this does not change the dataset.

## Testing Plan

### Unit Tests

Create `tests/test_causal_rationale.py`:

- parse GT scene and answer
- target extraction with relation overlap
- fallback target extraction
- fact extraction from generated response
- coverage and precision scoring
- compactness scoring

Create `tests/test_step_causal.py`:

- split `<think>` into steps
- perturb one step only
- preserve all tags
- build prefix up to `<answer>`
- normalize answers

Create `tests/test_csr_advantage.py`:

- group z-score shape and dtype
- invalid step CFS defaults to no modulation
- lambda update is non-negative
- GRPO group behavior is unchanged when CSR disabled

### Smoke Test

Add `scripts/debug_csr_batch.py`:

Inputs:

```bash
python3 scripts/debug_csr_batch.py \
  --data hunarbatra/STVQA-7K@train \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --num-examples 4
```

Outputs:

- GT answer
- selected target facts
- generated response
- extracted response facts
- rationale score breakdown
- generated step interventions

This script should run without starting a full RL training job.

### Training Smoke

Use:

```bash
python3 -m verl.trainer.main \
  config=scripts/config.yaml \
  ... \
  trainer.max_steps=2 \
  trainer.total_episodes=1 \
  worker.rollout.n=2 \
  algorithm.enable_csrfaith=True
```

Pass criteria:

- no shape mismatch after `_balance_batch`
- no DataProto union conflict
- metrics log
- checkpoint save/load includes CSR state

## Paper-Oriented Ablations

Minimum ablation table:

| Method | Accuracy | Spatial score | Rationale coverage | Rationale precision | Step CFS | No-image sensitivity |
| --- | --- | --- | --- | --- | --- | --- |
| GRPO | | | | | | |
| CIT-Faith | | | | | | |
| CSR rationale only | | | | | | |
| CSR step CFS only | | | | | | |
| Full CSR-Faith | | | | | | |

Important: do not overclaim "visual grounding" if PR only compares generated `<scene>` to generated `<think>`. CSR-Faith should report grounding against GT-derived spatial evidence.

## Main Risks

1. Pseudo-label evidence may be noisy.
   - Mitigation: log target confidence and use low-confidence targets with smaller weight.

2. Step-level CFS may be expensive.
   - Mitigation: cap steps, cache prefixes, lower `cf_max_tokens`, start with Phase 1.

3. Exact answer comparison may inflate CFS.
   - Mitigation: implement answer normalization before Phase 2.

4. The model may learn to mention target facts without using them.
   - Mitigation: step-level intervention must be part of the final full method, not only a metric.

5. Reviewer/verifier bias may weaken claims.
   - Mitigation: keep the first implementation deterministic; use verifier only as Phase 3 and report ablation.

## Recommended First PR Scope

Implement only Phase 1:

1. `verl/utils/causal_rationale.py`
2. config fields for CSR-Faith
3. CSR scoring hook in `ray_trainer.py`
4. `compute_csrfaith_grpo_advantage`
5. `scripts/csrfaith_7b_grpo.sh`
6. unit tests for rationale parsing/scoring

Do not add graph-only verifier or step-level generation in the first PR. That keeps the first change reviewable and gives immediate metrics to inspect.
