# Code Development Plan: Causal Spatial Critic / Bridge Network

This document maps the paper idea in `docs/paper_idea_causal_spatial_critic.md` to concrete repository changes. It is intentionally implementation-facing: every paper component has a code anchor, data artifact, configuration entry, and verification path.

The key method change is a small trainable bridge model:

```text
CausalSpatialCritic(q, E, step, intervention_type) -> P(answer changes)
```

This is not the PPO value critic already present in `verl/workers/critic/`.
It is a separate causal reward model that transfers expensive step-level
intervention labels into cheap dense rewards for CSR-GRPO.

## Current State

Already implemented:

| Paper component | Current code |
| --- | --- |
| derive causal spatial target `E` | `verl/utils/causal_rationale.py` |
| score rationale quality | `score_rationale` in `verl/utils/causal_rationale.py` |
| generate step interventions | `verl/utils/step_causal.py` |
| answer normalization | `verl/utils/answer_normalization.py` |
| prefix continuation | `FSDPWorker.generate_continuations`, `vLLMRollout.generate_continuations` |
| CSR advantage | `compute_csrfaith_grpo_advantage` in `verl/trainer/core_algos.py` |
| trainer integration | `RayPPOTrainer.fit` in `verl/trainer/ray_trainer.py` |
| smoke / train scripts | `scripts/csrfaith_smoke.sh`, `scripts/csrfaith_7b_grpo.sh` |
| critic example schema | `verl/utils/causal_critic_data.py` |
| lightweight bridge critic | `verl/models/causal_spatial_critic.py` |
| offline critic build/train/eval | `scripts/build_causal_critic_dataset.py`, `scripts/train_causal_spatial_critic.py`, `scripts/evaluate_causal_spatial_critic.py` |
| optional critic trainer hook | `algorithm.enable_causal_spatial_critic`, `algorithm.causal_critic_path` |
| critic run wrappers | `scripts/csrfaith_critic_smoke.sh`, `scripts/csrfaith_critic_7b_grpo.sh` |

Missing for the proposed paper:

- online refresh scheduler
- GPU/Ray/vLLM verification of critic-enabled CSR-GRPO
- paper-grade encoder option beyond the current hashed logistic baseline
- trainer-side runtime tests for critic scoring and advantage path

## Target Architecture

```text
existing dataset batch
    |
    v
build_causal_rationale_target(q, G, y)
    |
    v
policy rollout with <think>/<answer>
    |
    v
generate_step_interventions(...)
    |
    v
online continuation oracle -> label z = answer_changed
    |
    +--> write bridge-model training examples
    |
    v
train CausalSpatialCritic bridge model
    |
    v
bridge model predicts dense step causal score during GRPO
```

The model-level contribution is this middle layer. The online continuation
oracle creates sparse and expensive labels; the bridge model learns a reusable
causal scoring function; the trainer consumes that score as a reward signal.

## New Data Schema

Add a JSONL schema for critic examples:

```json
{
  "uid": "sample-or-rollout-id",
  "question": "Where is the chair relative to the table?",
  "target_objects": ["chair.1", "table.1"],
  "target_relations": [
    {"subject": "chair.1", "predicate": "left of", "object": "table.1"}
  ],
  "target_confidence": 1.0,
  "response_answer": "left",
  "step_index": 0,
  "step_text": "The chair is left of the table.",
  "intervention_type": "relation",
  "perturbed_step_preview": "The chair is right of the table.",
  "counterfactual_answer": "right",
  "label_answer_changed": 1,
  "source": "online_policy_continuation",
  "policy_checkpoint": "global_step_25"
}
```

Rules:

- store derived artifacts outside the original dataset directory by default
- never overwrite the original dataset
- allow missing `counterfactual_answer` with `label_answer_changed = null` for invalid examples
- include `target_confidence` so low-confidence pseudo-labels can be filtered

## Submodel Design Contract

Name in paper: **Causal Spatial Critic**.

Recommended code name: `CausalSpatialCritic`.

Metric/logging prefix: `csr_causal_critic/` or `csr/critic_*`, not bare
`critic/`, because bare `critic/` is already used by the PPO value critic.

Inputs:

| Field | Source | Purpose |
| --- | --- | --- |
| `question` | existing prompt | identify asked spatial relation |
| `target_objects`, `target_relations` | derived CSR target `E` | compress visual evidence without editing dataset |
| `step_text` | generated CoT step | candidate causal rationale step |
| `intervention_type` | generated intervention | relation/entity/mask sensitivity context |

Output:

```text
score in [0, 1] = estimated probability that intervening on this step changes the answer
```

Training objective:

```text
L = BCEWithLogitsLoss(logit_score, label_answer_changed)
```

Required capabilities:

- train offline from JSONL labels
- score batches during GRPO without backpropagating through the policy
- save and load a checkpoint plus serialization config
- expose calibration metrics so thresholds are not arbitrary
- support a lightweight local fallback for unit tests without GPU or heavy model dependencies

First implementation recommendation:

| Variant | Encoder | Head | Use case |
| --- | --- | --- | --- |
| local/static | bag-of-words or hashed n-grams | logistic regression / tiny MLP | tests and smoke |
| paper v1 | frozen sentence encoder | 2-layer MLP | cheap, stable training |
| paper v2 | small Transformer encoder | MLP with intervention embedding | stronger ablation |

The first paper version should start with the frozen-encoder path. It is easier
to train, cheap to refresh, and clearly separates the proposed bridge from a
large external VLM judge.

## New Files

### `verl/utils/causal_critic_data.py`

Purpose: construct, validate, serialize, and load critic examples.

Public API:

```python
@dataclass
class CausalCriticExample:
    uid: str
    question: str
    target_objects: list[str]
    target_relations: list[dict]
    target_confidence: float
    response_answer: str
    step_index: int
    step_text: str
    intervention_type: str
    perturbed_step_preview: str
    counterfactual_answer: str | None
    label_answer_changed: int | None
    source: str
    policy_checkpoint: str | None = None


def build_critic_examples(
    problem: str,
    target: RationaleTarget,
    response_text: str,
    interventions: list[StepIntervention],
    counterfactual_answers: list[str | None],
    *,
    uid: str,
    policy_checkpoint: str | None = None,
) -> list[CausalCriticExample]:
    ...


def write_jsonl(examples: Iterable[CausalCriticExample], path: str) -> None:
    ...


def read_jsonl(path: str) -> list[CausalCriticExample]:
    ...
```

Tests:

- `tests/test_causal_critic_data.py`
- invalid counterfactual answer yields `label_answer_changed is None`
- normalized answer equality is reused
- JSONL roundtrip preserves relation fields

### `verl/models/causal_spatial_critic.py`

Purpose: lightweight classifier for `P(answer_changed | q, E, step, intervention_type)`.

Recommended first implementation:

- text serialization:
  - `[QUESTION] ...`
  - `[TARGET_FACTS] object: ...; relation: ...`
  - `[STEP] ...`
  - `[INTERVENTION] relation/entity/mask`
- encoder:
  - first version can use `sentence-transformers`-style frozen embeddings if available, or a HuggingFace encoder
  - keep a fallback bag-of-words / logistic regression implementation for local tests without heavy dependencies

Minimal API:

```python
class CausalSpatialCritic:
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CausalSpatialCritic":
        ...

    def score_batch(self, examples: list[CausalCriticExample]) -> list[float]:
        """Return causal-effect probabilities in [0, 1]."""
```

For training:

```python
def serialize_critic_input(example: CausalCriticExample) -> str:
    ...
```

Tests:

- `tests/test_causal_spatial_critic_static.py`
- serialization includes question, target facts, step, and intervention
- score output range is `[0, 1]`
- load failure produces a clear error when `enable_causal_spatial_critic=True`
- local fallback model is deterministic

### `scripts/build_causal_critic_dataset.py`

Purpose: generate critic-label JSONL from rollout records or debug input.

Inputs:

- existing dataset rows or saved rollout JSONL
- generated response text
- optional `policy_checkpoint`

Outputs:

- `cache/causal_critic/{split}.jsonl`

Important options:

```bash
python3 scripts/build_causal_critic_dataset.py \
  --input-jsonl rollouts.jsonl \
  --output cache/causal_critic/train.jsonl \
  --max-steps 6 \
  --max-interventions-per-step 1 \
  --min-target-confidence 0.4
```

### `scripts/train_causal_spatial_critic.py`

Purpose: train the critic from generated labels.

Arguments:

```bash
python3 scripts/train_causal_spatial_critic.py \
  --train-jsonl cache/causal_critic/train.jsonl \
  --val-jsonl cache/causal_critic/val.jsonl \
  --output-dir ckpts/causal_spatial_critic \
  --encoder sentence-transformers/all-MiniLM-L6-v2
```

Outputs:

- critic checkpoint
- `metrics.json`
- threshold calibration file

Metrics:

- accuracy
- F1
- AUROC if sklearn is available
- calibration / expected positive rate
- oracle agreement by intervention type
- held-out policy-checkpoint agreement

### `scripts/evaluate_causal_spatial_critic.py`

Purpose: compare critic predictions against fresh online interventions.

This matters because the critic can overfit stale policy behavior. Evaluation should support:

- held-out prompts/images
- held-out intervention types
- held-out policy checkpoints

Minimum acceptance criteria before using the critic in paper experiments:

- validation AUROC or F1 is better than a majority-class baseline
- agreement holds on at least one held-out intervention type or policy checkpoint
- critic ranking correlates with online step-CFS on fresh rollouts

## Config Additions

Add to `AlgorithmConfig` in `verl/trainer/config.py`:

```python
enable_causal_spatial_critic: bool = False
causal_critic_path: str = ""
causal_critic_weight: float = 1.0
causal_critic_min_target_confidence: float = 0.4
causal_critic_refresh_freq: int = -1
causal_critic_use_online_fallback: bool = True
```

Meaning:

- `enable_causal_spatial_critic`: use learned critic in CSR advantage
- `causal_critic_path`: local checkpoint path
- `causal_critic_weight`: additive weight before group z-score or inside CSR advantage
- `causal_critic_min_target_confidence`: ignore low-confidence targets for critic reward
- `causal_critic_refresh_freq`: optional periodic relabeling/retraining cadence
- `causal_critic_use_online_fallback`: use online step-CFS if critic unavailable

## Trainer Integration

Current hook location:

`RayPPOTrainer.fit`, inside the `adv` timer after response decoding and CSR rationale scoring.

Integration plan:

1. Build `targets` and `response_texts` exactly as current CSR path does.
2. If `enable_causal_spatial_critic`:
   - split steps
   - build critic examples without running online continuation
   - call `critic.score_batch`
   - aggregate per rollout into `critic_causal_scores_t`
3. Else:
   - keep current online step-CFS path
4. Feed one causal score tensor into `compute_csrfaith_grpo_advantage`.

Important invariant:

```text
len(causal_scores_t) == batch_size after _balance_batch
```

The current code computes CSR after `_balance_batch`, so critic scores should
also be computed after `_balance_batch`.

Operational constraints:

- load the Causal Spatial Critic once during trainer initialization or the first
  enabled CSR step
- keep it in eval mode during policy optimization
- do not route it through the existing PPO critic worker group
- allow CPU scoring for small batches; allow optional GPU scoring for full runs
- wrap scoring failures with a config-controlled fallback to online step-CFS

## Advantage Function Update

Option A: preserve current function signature and pass critic scores as `step_cfs_scores`.

Pros:

- smallest implementation change
- existing tests mostly survive

Cons:

- logs become ambiguous

Option B: extend function:

```python
def compute_csrfaith_grpo_advantage(
    ...,
    rationale_scores: torch.Tensor,
    step_cfs_scores: torch.Tensor | None,
    critic_causal_scores: torch.Tensor | None,
    causal_signal_mode: str = "online_step_cfs",
):
    ...
```

Recommended: **Option B** for paper clarity.

Log separate metrics:

- `csr/step_cfs_mean`
- `csr/critic_causal_mean`
- `csr/critic_enabled`
- `csr/critic_valid_ratio`
- `csr/critic_source` (`online_step_cfs`, `learned_critic`, or `fallback_online`)
- `csr_causal_critic/score_mean`
- `csr_causal_critic/score_std`

## Scripts

Add:

- `scripts/build_causal_critic_dataset.py`
- `scripts/train_causal_spatial_critic.py`
- `scripts/evaluate_causal_spatial_critic.py`
- `scripts/csrfaith_critic_7b_grpo.sh`
- `scripts/csrfaith_critic_smoke.sh`

Update:

- `scripts/check_csrfaith_ready.py`
  - check critic path if enabled
  - check new scripts exist
- `scripts/env.local.example.sh`
  - optional `CAUSAL_CRITIC_PATH`

## Documentation Updates

Update:

- `README.md`
  - describe critic as the planned paper-grade method module, separate from current implemented CSR baseline
- `docs/csrfaith_run_guide.md`
  - add stage for critic dataset generation, critic training, critic-RL run
- `docs/csrfaith_implementation_status.md`
  - mark critic path as proposed until implemented

Keep:

- `docs/causal_spatial_rationales_dev.md`
  - historical plan for current CSR implementation

## Test Plan

Lightweight tests that should pass without GPU:

```bash
python3 -m unittest \
  tests.test_causal_critic_data \
  tests.test_causal_spatial_critic_static \
  tests.test_csr_advantage \
  tests.test_trainer_static
```

Full local static suite:

```bash
python3 -m unittest discover -s tests
python3 -m compileall -q verl scripts tests
bash -n scripts/*.sh scripts/extras/*.sh
```

Full GPU/Ray/vLLM verification:

```bash
bash scripts/csrfaith_critic_smoke.sh
```

Pass criteria:

- standard CSR metrics still appear
- critic metrics appear when enabled
- no shape mismatch after `_balance_batch`
- no `DataProto` union conflict
- checkpoint saves critic config/state reference

## Milestones

### Milestone 1: Offline Critic Dataset

- implement `causal_critic_data.py`
- implement JSONL builder script
- unit tests pass
- generate a small example cache from debug records

### Milestone 2: Critic Model

- implement text serialization
- implement baseline train/eval script
- save checkpoint and metrics
- validate on held-out JSONL
- compare against majority, lexical-overlap, and random baselines

### Milestone 3: CSR Trainer Integration

- add config fields
- load critic in trainer when enabled
- log critic metrics
- integrate critic score into CSR advantage
- verify no naming collision with the existing PPO value critic

### Milestone 4: Paper Experiments

- online step-CFS baseline
- critic-only reward
- full CSR + critic
- cost and faithfulness comparison

## Paper-to-Code Traceability Matrix

| Paper claim | Code artifact | Verification |
| --- | --- | --- |
| no original dataset modification | derived JSONL cache outside dataset path | builder test + docs |
| target facts from existing scene graph | `build_causal_rationale_target` | `tests/test_causal_rationale.py` |
| step-level intervention oracle | `generate_step_interventions`, continuation helper | `tests/test_step_causal.py`, GPU smoke |
| critic learns causal effect labels | `causal_critic_data.py`, critic train script | critic eval metrics |
| critic provides dense causal reward | trainer integration + CSR advantage | `tests/test_csr_advantage.py`, smoke logs |
| method is distinct from judge-based FGRPO | no external judge required for CSR critic path | config/script checks |
| proposed submodel is real, trainable, and optimized | `CausalSpatialCritic` checkpoint + train/eval scripts | held-out critic metrics + ablations |

## Non-goals for First Implementation

- no joint end-to-end training of critic and policy in the same optimizer
- no image encoder inside critic v1
- no human rationale annotation
- no replacement of the current online step-CFS path
- no modification of original datasets
