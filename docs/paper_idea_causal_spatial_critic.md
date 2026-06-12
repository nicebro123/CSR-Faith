# Paper Idea: Causal Spatial Rationales with a Learned Causal Bridge

## Working Title

**Causal Spatial Rationales for Faithful Multimodal Reasoning**

Internal method name: **CSR-Faith + Causal Spatial Critic**.

The A-conference version should make the critic a first-class model component,
not an optional reward trick. The critic is a small trainable bridge between
derived spatial evidence and policy optimization: it learns whether a reasoning
step is causally necessary for the policy's answer.

## One-sentence Thesis

Faithful multimodal spatial reasoning should not only produce correct answers or visually plausible CoT; each cited spatial reasoning step should be necessary for the answer under counterfactual intervention, and this causal signal can be distilled into a lightweight critic for scalable RL optimization.

## Motivation

Multimodal RL methods can improve answer accuracy while leaving the generated chain-of-thought weakly tied to the final answer. A model can mention objects and relations that look reasonable, yet those steps may be decorative: changing or deleting them does not change the answer.

Existing faithful-RL work such as **Faithful GRPO** constrains logical consistency and visual grounding during GRPO. That is valuable, but it answers a different question:

- consistency: does the CoT entail the answer?
- grounding: does the CoT describe the image correctly?

CSR-Faith asks a stricter causal question:

- causal necessity: if this spatial reasoning step is counterfactually changed, does the policy's answer change?

The paper should therefore avoid claiming that consistency/grounding are new.
The novelty is the **causal spatial bridge loop**: derive spatial evidence from
existing scene annotations, intervene on individual reasoning steps, convert
answer changes into causal supervision, and train a compact critic that transfers
this expensive intervention signal back into GRPO.

## Why the Small Network Matters

Without a trainable submodel, CSR-Faith can look like a careful engineering
pipeline: extract evidence, run interventions, add a reward. That may be useful,
but it is weaker as an A-conference method.

The stronger claim is that the paper proposes a **Causal Spatial Critic**:

```text
C_theta(q, E, s_t, intervention_type) -> P(answer changes)
```

This critic is the method's learned intermediary:

- it receives the question, derived spatial evidence, one reasoning step, and
  the intervention type
- it predicts whether perturbing that step would change the policy answer
- it replaces expensive online intervention calls during most GRPO updates
- it can be calibrated, refreshed, ablated, and evaluated as its own model

Reviewer-facing sentence:

> We introduce a lightweight causal bridge network that distills step-level
> counterfactual continuation outcomes into dense faithfulness rewards, enabling
> scalable optimization of spatially necessary rationales.

## Core Hypothesis

For spatial reasoning tasks, a CoT is more faithful when it satisfies four properties:

1. **Evidence coverage**: it cites the spatial facts needed for the answer.
2. **Minimal sufficiency**: it avoids irrelevant or unsupported facts.
3. **Step-level causal effect**: intervening on a cited reasoning step changes the model's answer distribution or final answer.
4. **Learned causal transfer**: a compact critic can predict this effect from spatial evidence and step text, avoiding full online interventions at every update.

The strongest paper claim should be:

> A lightweight causal bridge trained from counterfactual continuation labels can
> provide denser, cheaper, and more stable causal faithfulness rewards than
> running full step-level interventions at every RL update.

## Current Baseline in This Repository

The repository already implements a non-learned CSR-Faith pipeline:

| Component | Current implementation | Status |
| --- | --- | --- |
| evidence target extraction | `verl/utils/causal_rationale.py` | implemented |
| rationale scoring | coverage / precision / compactness / sufficiency / necessity proxy | implemented |
| step-level intervention | `verl/utils/step_causal.py` | implemented |
| answer-change CFS | policy prefix continuation via vLLM | implemented |
| GRPO integration | `compute_csrfaith_grpo_advantage` in `verl/trainer/core_algos.py` | implemented |
| training loop hook | `RayPPOTrainer.fit` in `verl/trainer/ray_trainer.py` | implemented |
| learned Causal Spatial Critic | not implemented | central proposed module |

Without the critic, the current repository is a solid CSR reward pipeline. With
the critic, the paper has a clear trainable module, a scalability argument, and
a model-level contribution that can be optimized and evaluated independently.

## Proposed Method

### Notation

For each training sample:

- `I`: image
- `q`: question / prompt
- `G`: ground-truth scene graph from existing dataset fields
- `y`: ground-truth answer
- `r`: generated response containing `<scene>`, `<think>`, `<answer>`
- `S = {s_1, ..., s_T}`: reasoning steps split from `<think>`
- `E = phi(q, G, y)`: causal spatial rationale target derived from existing fields

No original dataset file is modified. `E` is derived on the fly or cached as an auxiliary artifact.

### Stage 1: Derive Causal Spatial Evidence

Build a target evidence set:

```text
E = {objects, spatial relations, answer}
```

The current implementation uses deterministic scene-graph heuristics:

- select relations overlapping with question or answer tokens
- include endpoint objects
- fall back to object overlap
- fall back to low-confidence top-k relations when no lexical signal exists

Paper language must be careful: these are **GT-derived evidence targets**, not guaranteed human-minimal rationales.

### Stage 2: Generate Step-level Counterfactual Labels

For each generated step `s_t`, construct interventions:

- relation flip: `left of -> right of`, `above -> below`, etc.
- entity swap: replace a cited entity with another target entity
- mask: replace the step with an unavailable-evidence statement

Then run prefix continuation with the same policy:

```text
a_original = answer(pi(I, q, r))
a_cf       = answer(pi(I, q, intervene(r, s_t)))
z_t        = 1[normalize(a_cf) != normalize(a_original)]
```

`z_t = 1` means the step had observed causal effect under this intervention. `z_t = 0` means the answer did not change, so the step may be non-causal, redundant, or the intervention may be weak.

### Stage 3: Train a Causal Spatial Critic

Train a lightweight model:

```text
C_theta(q, E, s_t, intervention_type) -> p(answer_changes)
```

Recommended first version:

- input is text only: question, serialized target facts, step text, intervention type
- encoder can be a small Transformer or frozen sentence encoder + MLP
- target is the automatically generated label `z_t`
- loss is binary cross entropy with class balancing and calibration

```text
L_critic = BCE(C_theta(q, E, s_t, k_t), z_t)
```

where `k_t` is the intervention type. The critic is intentionally not the same
as the PPO value critic already present in the training framework. It is a
faithfulness reward model trained from causal intervention labels.

Minimal architecture:

```text
serialize(q, E, s_t, intervention_type)
        |
small text encoder or frozen sentence encoder
        |
MLP reward head
        |
P(answer_changes)
```

Paper-grade architecture:

```text
question encoder        -> h_q
target-fact encoder     -> h_E
step encoder            -> h_s
intervention embedding  -> h_k
concat(h_q, h_E, h_s, h_k) -> MLP -> sigmoid
```

Why no image input in v1:

- visual content is already compressed into `E`
- it keeps the critic cheap and stable
- it makes the critic clearly distinct from an external VLM judge

Optional v2:

- add compact visual features, object crops, or bbox encodings if text-only facts are insufficient

### Stage 4: Use Critic Reward in GRPO

The policy receives:

```text
R_task      = answer/spatial task reward
R_rat       = rationale evidence score
R_causal    = mean_t C_theta(q, E, s_t, intervention_type)
```

GRPO advantage can be written as:

```text
A_task   = group_zscore(R_task)
A_rat    = group_zscore(R_rat)
A_causal = group_zscore(R_causal)

A_base = A_task + lambda_rat * A_rat + lambda_causal * A_causal
A_final = A_base * (alpha + (1 - alpha) * R_causal)
```

The current repository already implements the analogous formula with oracle
step-CFS. The critic replaces or complements expensive online interventions.
The training loop becomes:

1. collect policy rollouts
2. run a limited number of online step interventions to label examples
3. train or refresh `C_theta`
4. use critic scores as dense causal rewards in later GRPO updates
5. periodically evaluate critic scores against fresh online interventions

## Differentiation from Faithful GRPO

Literature anchor for writing: **Faithful GRPO: Improving Visual Spatial
Reasoning in Multimodal Language Models via Constrained Policy Optimization**
([arXiv:2604.08476](https://arxiv.org/abs/2604.08476)).

| Dimension | Faithful GRPO | CSR-Faith + Causal Spatial Critic |
| --- | --- | --- |
| faithful target | logical consistency and visual grounding | causal necessity of spatial reasoning steps |
| supervision source | LLM/VLM judges and grounding reward | counterfactual answer-change labels derived from the current policy |
| model component | constrained GRPO objective | learned causal bridge / critic distilled from intervention oracle |
| granularity | sentence/batch-level constraints | step-level causal credit |
| data requirement | judge scoring during RL | no human rationale labels; no original dataset modification |
| key question | "Is this CoT consistent and grounded?" | "Does changing this step change the answer?" |

Positioning sentence:

> Faithful GRPO prevents accuracy gains from degrading consistency and grounding; CSR-Faith studies whether spatial CoT steps are causally used by the policy, and distills that intervention signal into a trainable critic.

This framing makes the methods complementary rather than mutually exclusive.

## Main Contributions

The final paper should claim four contributions:

1. **Causal spatial rationale construction** from existing scene graph and answer fields, without extra human annotation or dataset modification.
2. **Step-level counterfactual intervention oracle** for measuring whether individual spatial reasoning steps affect policy answers.
3. **Causal Spatial Critic**, a lightweight trainable bridge model that predicts step-level answer sensitivity from spatial evidence and reasoning text.
4. **CSR-GRPO training objective** that combines task reward, rationale sufficiency, and critic-predicted causal effect with adaptive Lagrangian constraints.

## Experimental Plan

### Primary Questions

1. Does CSR-Faith improve answer accuracy over standard GRPO?
2. Does it improve rationale faithfulness under intervention, not just textual plausibility?
3. Does the learned critic approximate the intervention oracle well enough to reduce online intervention cost?
4. Does the critic improve stability compared with sparse online step-CFS rewards?

### Baselines

| Baseline | Purpose |
| --- | --- |
| SFT / original checkpoint | non-RL baseline |
| standard GRPO | answer-only RL |
| CIT-Faith | judge-based faithfulness baseline in this repository |
| Faithful GRPO-style constraints | consistency/grounding comparison if reimplemented or fairly proxied |
| CSR rationale only | effect of evidence reward without causal step reward |
| online step-CFS only | effect of intervention oracle without critic |
| critic reward only | effect of distilled causal reward |
| full CSR-Faith + critic | complete method |

### Metrics

| Metric | What it tests |
| --- | --- |
| answer accuracy | task success |
| spatial score | task-specific spatial correctness |
| rationale coverage | cites target evidence |
| rationale precision | avoids irrelevant facts |
| step-CFS mean | intervention changes answers |
| step-CFS valid ratio | intervention pipeline reliability |
| critic AUROC / F1 | critic label prediction quality |
| intervention budget | cost reduction versus online step-CFS |
| CoT length | compactness / anti-verbosity |

### Ablations

- no critic, online step-CFS only
- frozen critic versus jointly refreshed critic
- critic trained on relation/entity/mask labels separately
- no rationale target, critic only
- no causal modulation, additive reward only
- different target construction confidence thresholds
- small MLP critic versus small Transformer critic
- critic trained on stale policy labels versus periodically refreshed labels

## A-conference Novelty Test

This project is worth positioning as an A-conference method only if the learned
critic earns at least one of these outcomes:

1. **Faster**: it reduces online intervention calls by a clear factor while
   preserving faithfulness improvements.
2. **Higher**: it improves answer accuracy or spatial score over online step-CFS
   by providing smoother reward.
3. **Stronger**: it generalizes to held-out intervention types, prompts, or
   policy checkpoints better than simple lexical heuristics.

If the critic cannot beat a cheap heuristic or cannot approximate the online
oracle, the paper should fall back to a narrower workshop-style CSR reward
paper.

## Claims and Non-claims

### Safe Claims

- The method does not require editing the original dataset.
- The current target construction is automatic and deterministic.
- Step-CFS measures policy sensitivity under specific interventions.
- The critic is a learned approximation of this intervention signal.

### Claims to Avoid Until Verified

- Do not claim true human-minimal rationales unless human annotation is added.
- Do not claim complete visual grounding from text-only target facts.
- Do not claim causal effect in the world; it is causal effect on the policy's generated answer under controlled textual interventions.
- Do not claim lower compute until online intervention budget is measured.

## Risks

| Risk | Mitigation |
| --- | --- |
| target facts are noisy | log target confidence; filter low-confidence samples; ablate target construction |
| interventions are weak | maintain intervention type metrics; add paraphrased or graph-guided interventions |
| critic learns artifacts | hold out prompts/images; evaluate against fresh online interventions |
| reward hacking | monitor CoT length, evidence precision, and answer accuracy jointly |
| critic drift | periodically refresh critic labels from current policy |
| critic adds no benefit | include oracle-only and heuristic-only baselines; pivot if critic does not improve cost or faithfulness |

## Paper-quality Version Checklist

- full GPU smoke succeeds with CSR metrics
- intervention dataset can be generated reproducibly
- critic training script produces stable validation metrics
- critic reward can replace at least part of online step-CFS with similar ranking
- README and run guide expose the critic path
- experiments include both task accuracy and faithfulness metrics

## Corresponding Code Development Document

Implementation details for this paper idea are specified in:

`docs/causal_spatial_critic_dev.md`
