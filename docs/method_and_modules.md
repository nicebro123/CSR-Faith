# CIT-Faith: Faithful Spatial Reasoning via Counterfactual Intervention Training

> 基于独立 LLM Judge 的多模态空间推理忠实性优化框架，在 [SpatialThinker](https://github.com/hunarbatra/SpatialThinker) 代码基座上实现。

---

## 概述

CIT-Faith 在 SpatialThinker 的 GRPO 训练流程中引入三维忠实性检测与优化机制，解决 RL 训练下思维链（CoT）沦为装饰性文本的问题。核心改进：

- **独立 LLM Judge**：参数冻结的审查模型（Qwen2.5-7B-Instruct-AWQ）独立评估 CoT 质量，避免策略模型自评的信号污染
- **三向忠实性检测**：自洽性（SC）、感知-思考一致性（PR）、因果忠实性（CFS）
- **双机制优化**：SC/PR 通过拉格朗日对偶约束嵌入 advantage，CFS 通过反事实干预 + 乘性调制压制装饰性 CoT

当前代码还实现了 **CSR-Faith（Causal Spatial Rationales）** 扩展：不修改原数据集，直接从现有 GT scene graph 派生最小空间证据目标，训练模型生成必要、充分、紧凑且具有 step-level 因果作用的空间 CoT。

---

## 环境要求

与 SpatialThinker 基线相同，额外需要：

```
Python >= 3.9
transformers >= 4.49.0
vllm >= 0.7.3（推荐 0.8.0）
flash-attn >= 2.4.3
torch >= 2.1.0
ray >= 2.9.0
```

审查模型额外占用约 2-3GB GPU 显存（AWQ 4-bit 量化）。

## 安装

```bash
git clone <repo_url>
cd CIT-Faith
pip install -e .
```

---

## 快速开始

### 1. 使用 CIT-Faith 训练

```bash
bash scripts/citfaith_7b_grpo.sh
```

该脚本在 SpatialThinker 的训练流程基础上，启用 CIT-Faith 的全部忠实性优化组件。关键参数说明：

```bash
# 启用 CIT-Faith（设为 False 则退化为原始 SpatialThinker GRPO）
algorithm.enable_citfaith=True

# 审查模型路径（冻结，AWQ 量化，不参与梯度更新）
algorithm.reviewer_model_path="Qwen/Qwen2.5-7B-Instruct-AWQ"

# 拉格朗日约束阈值
algorithm.tau_sc=0.8       # 自洽性阈值，batch 均分低于此值时加强约束
algorithm.tau_pr=0.7       # 感知-思考一致性阈值

# 对偶变量学习率
algorithm.dual_lr=0.01     # 拉格朗日乘子的自适应更新步长

# 因果调制参数
algorithm.cfs_alpha=0.1    # g(r) = 0.1 + 0.9r，CFS=0 时 advantage 被压缩到 10%

# 反事实续写长度
algorithm.cf_max_tokens=50
```

### 2. 使用原始 SpatialThinker 基线训练（不受影响）

```bash
bash scripts/spatialthinker_7b_grpo.sh
```

所有 CIT-Faith 修改通过 `enable_citfaith=False`（默认值）完全跳过，原始训练路径不受任何影响。

### 3. 使用 CSR-Faith 训练

```bash
bash scripts/csrfaith_7b_grpo.sh
```

CSR-Faith 与 CIT-Faith 可独立开关。默认 CSR 脚本关闭 `enable_citfaith`，只启用 causal spatial rationale 训练：

```bash
algorithm.enable_citfaith=False
algorithm.enable_csrfaith=True
algorithm.csr_target_max_relations=4
algorithm.csr_target_max_objects=6
algorithm.csr_max_steps=6
algorithm.csr_max_step_interventions=1
algorithm.tau_coverage=0.7
algorithm.tau_step_cfs=0.5
```

先做 CPU 级运行前检查（不会启动 Ray 或加载模型）：

```bash
python3 scripts/check_csrfaith_ready.py --no-fail
```

快速 smoke（需要完整训练依赖和 GPU/Ray 环境）：

```bash
bash scripts/csrfaith_smoke.sh
```

该脚本将训练缩到 `trainer.max_steps=2`、`rollout.n=2`、小 batch、关闭验证、中间保存和 KL/ref policy，用于先检查 CSR 训练链路是否能跑通；关闭验证时不会加载 val dataloader，训练结束仍会保存最终 checkpoint。

### 4. 离线调试 CSR 派生监督

不启动模型或 Ray，仅检查 target、rationale score 和 step intervention：

```bash
python3 scripts/debug_csr_batch.py \
    --input-json sample.json \
    --jsonl
```

也可以直接传单条样本：

```bash
python3 scripts/debug_csr_batch.py \
    --problem "<question>" \
    --ground-truth "<scene>...</scene><answer>...</answer>" \
    --response "<observe>...</observe><scene>...</scene><think>...</think><answer>...</answer>"
```

构建可追溯 CSR target cache：

```bash
python3 scripts/build_csr_target_cache.py \
    --input-json sample.jsonl \
    --output cache/csr_targets/stvqa/train.jsonl
```

如果本地安装了 `datasets`，也可使用：

```bash
python3 scripts/build_csr_target_cache.py \
    --data hunarbatra/STVQA-7K@train \
    --ground-truth-key answer \
    --output cache/csr_targets/stvqa/train.jsonl
```

### 5. 合并 Checkpoint

```bash
python3 scripts/model_merger.py --local_dir path_to_your_last_actor_checkpoint
```

### 6. 评估

```bash
python3 evaluation/evals.py \
    --dataset blink-spatial \
    --template spatial_thinker \
    --model_path <checkpoint_path> \
    --cuda 0 \
    --batch_size 4
```

---

## 超参数调优指南

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `tau_sc` | 0.8 | SC 约束阈值 | 训练早期可降低到 0.6-0.7，后期提高到 0.85 |
| `tau_pr` | 0.7 | PR 约束阈值 | 与 tau_sc 同步调整，通常略低 0.05-0.1 |
| `dual_lr` | 0.01 | 拉格朗日乘子学习率 | 过大会导致约束震荡，过小约束响应迟钝；建议 0.005-0.02 |
| `cfs_alpha` | 0.1 | CFS 调制下界 | 控制对装饰性 CoT 的惩罚力度；0.0=完全压制，0.3=温和惩罚 |
| `cf_max_tokens` | 50 | 反事实续写长度 | 答案通常很短，50 足够；复杂任务可增加到 100 |
| `reviewer_gpu_memory` | 0.15 | 审查模型显存占比 | AWQ 模型占用少，0.15 通常足够；显存紧张时可降到 0.10 |
| `csr_target_max_relations` | 4 | CSR target 中最多保留的 GT relation 数 | 复杂 scene 可提高到 6-8，但会增加匹配噪声 |
| `csr_max_steps` | 6 | step-level 干预最多检查的 `<think>` 步数 | smoke 可设为 2，正式训练建议 3-6 |
| `csr_max_step_interventions` | 1 | 每个 step 的最大干预数 | 1 最省；2-3 会增加开销但覆盖 entity/mask |
| `tau_coverage` | 0.7 | rationale coverage 约束阈值 | 如果 target 噪声较大可降到 0.5-0.6 |
| `tau_step_cfs` | 0.5 | step-level CFS 约束阈值 | 初期可设 0.3-0.5，观察 valid ratio 后调整 |

---

## 项目结构与代码改动

### 新增文件

```
verl/utils/reviewer.py          # 冻结审查模型（LLM Judge）
verl/utils/counterfactual.py     # 反事实干预与 CFS 计算
verl/utils/answer_normalization.py # 答案归一化比较
verl/utils/causal_rationale.py   # CSR target 派生与 rationale scoring
verl/utils/step_causal.py        # step-level 反事实干预与 CFS
scripts/citfaith_7b_grpo.sh      # CIT-Faith 训练启动脚本
scripts/csrfaith_7b_grpo.sh      # CSR-Faith 训练启动脚本
scripts/csrfaith_smoke.sh        # CSR-Faith 最小训练 smoke
scripts/check_csrfaith_ready.py  # CSR-Faith 环境与入口预检
scripts/debug_csr_batch.py       # CSR 离线样本调试
scripts/build_csr_target_cache.py # CSR target cache 构建
```

### 修改文件

```
verl/trainer/config.py                    # AlgorithmConfig 新增 CIT-Faith 字段
verl/trainer/core_algos.py                # 新增 CIT/CSR advantage 计算
verl/trainer/ray_trainer.py               # 训练循环集成审查评估、CFS、CSR scoring
verl/workers/fsdp_workers.py              # 新增 generate_continuations worker 方法
verl/workers/rollout/vllm_rollout_spmd.py # 新增 vLLM prefix-continuation decode
```

---

## 模块详解

### 1. 审查模型 (`verl/utils/reviewer.py`)

使用 Qwen2.5-7B-Instruct-AWQ 通过 vLLM 部署。参数全程冻结，不参与训练梯度更新。

**自洽性评估（SC）**：输入 `<think>` 段，从四个维度打分——空间断言一致性、数量一致性、传递性一致性、结论一致性。SC = 四维度均值。

**感知-思考一致性评估（PR）**：同时输入 `<scene>` 和 `<think>` 段，检测四类违规——直接矛盾、数量不符、身份错乱、凭空引入。PR = 1 - 加权违规分。

```python
from verl.utils.reviewer import ReviewerModel

reviewer = ReviewerModel(model_name_or_path="Qwen/Qwen2.5-7B-Instruct-AWQ")
sc_scores, pr_scores = reviewer.evaluate_batch(response_texts)
# sc_scores: List[float]，每条 rollout 的自洽性分数 [0, 1]
# pr_scores: List[float]，每条 rollout 的感知一致性分数 [0, 1]
```

### 2. 反事实干预 (`verl/utils/counterfactual.py`)

对 `<think>` 段施加三类语义对立扰动，然后通过策略模型的 prefix-continuation decode 检测答案是否改变。

```python
from verl.utils.counterfactual import generate_counterfactual_inputs, compute_cfs_score

# 生成反事实变体
interventions = generate_counterfactual_inputs(rollout_text)
# 返回: [("entity", perturbed_text), ("coord", ...), ("relation", ...)]

# CFS = 改变答案的干预数 / 有效干预数
cfs = compute_cfs_score(original_answer, counterfactual_answers)
# cfs=1.0: CoT 真正参与决策（所有干预都改变了答案）
# cfs=0.0: CoT 是装饰性的（没有干预改变答案）
```

### 3. 优化算法 (`verl/trainer/core_algos.py`)

```python
from verl.trainer.core_algos import compute_citfaith_grpo_advantage, update_lagrangian_multipliers

# Step 1: 对偶约束 advantage
# A_base = A_acc + λ_SC · A_SC + λ_PR · A_PR

# Step 2: 因果调制
# A_CIT = A_base × g(CFS),  g(r) = α + (1-α)·r

advantages, returns = compute_citfaith_grpo_advantage(
    token_level_rewards, response_mask, index,
    sc_scores, pr_scores, cfs_scores,
    lambda_sc, lambda_pr, alpha,
)

# Step 3: 对偶上升更新乘子
lambda_sc, lambda_pr = update_lagrangian_multipliers(
    lambda_sc, lambda_pr, batch_sc_mean, batch_pr_mean, tau_sc, tau_pr, eta,
)
```

### 4. CSR-Faith (`verl/utils/causal_rationale.py`, `verl/utils/step_causal.py`)

CSR-Faith 从现有 GT scene graph 自动派生 target facts，不需要额外人工标注或修改原数据集：

```python
from verl.utils.causal_rationale import (
    extract_gt_scene_and_answer,
    build_causal_rationale_target,
    score_rationale,
)

gt_scene, gt_answer = extract_gt_scene_and_answer(ground_truth_text)
target = build_causal_rationale_target(problem, gt_scene, gt_answer)
rationale_score = score_rationale(response_text, target)
```

训练时 CSR-Faith 还会对 `<think>` 做 step-level intervention：

```python
from verl.utils.step_causal import (
    generate_step_interventions,
    build_prefixes_for_step_interventions,
    compute_step_causal_score,
)

interventions = generate_step_interventions(response_text, target, max_steps=6)
prefixes = build_prefixes_for_step_interventions(interventions)
# prefixes 交给当前策略模型 prefix-continuation decode
step_score = compute_step_causal_score(original_answer, interventions, cf_answers)
```

CSR-Faith 的 advantage 形式为：

```
A_task = group_zscore(task_reward)
A_rat  = group_zscore(rationale_overall)
A_step = group_zscore(step_cfs_mean)
A_base = A_task + λ_cov * A_rat + λ_step * A_step
A_csr  = A_base * (α + (1 - α) * step_cfs_mean)
```

---

## 训练监控

CIT-Faith 在每个训练步骤向 wandb 输出以下指标：

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| `citfaith/sc_mean` | Batch 平均自洽性 | 收敛到 tau_sc 附近 |
| `citfaith/pr_mean` | Batch 平均感知一致性 | 收敛到 tau_pr 附近 |
| `citfaith/cfs_mean` | Batch 平均因果忠实性 | 持续上升（核心指标） |
| `citfaith/lambda_sc` | SC 拉格朗日乘子 | 先上升后稳定 |
| `citfaith/lambda_pr` | PR 拉格朗日乘子 | 先上升后稳定 |
| `citfaith/cfs_valid_ratio` | CFS 有效干预比例 | 0.7 以上为正常 |

CSR-Faith 额外输出：

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| `csr/target_confidence_mean` | 自动 target 派生置信度 | 越高说明 question/GT scene overlap 越稳定 |
| `csr/rationale_coverage_mean` | `<think>/<scene>` 覆盖 target facts 比例 | 上升并接近 `tau_coverage` |
| `csr/rationale_precision_mean` | 生成 facts 中命中 target 的比例 | 上升，避免装饰性事实 |
| `csr/rationale_compactness_mean` | target facts / mentioned facts | 防止 CoT 过度膨胀 |
| `csr/rationale_overall_mean` | coverage/precision/compactness/sufficiency/necessity 加权分 | 稳定上升 |
| `csr/step_cfs_mean` | step-level 反事实答案改变率 | 上升，核心因果指标 |
| `csr/step_cfs_valid_ratio` | 有效 step CFS 样本比例 | 越高越好；低值说明格式或干预生成失败 |
| `csr/step_interventions_mean` | 每条 rollout 的 step 干预数量 | 用于监控开销 |
| `csr/lambda_coverage` | coverage 拉格朗日乘子 | 低于阈值时上升 |
| `csr/lambda_step_cfs` | step CFS 拉格朗日乘子 | 低于阈值时上升 |

---

## Checkpoint 恢复

CIT-Faith 的拉格朗日乘子状态（`lambda_sc`, `lambda_pr`）会随 checkpoint 一起保存在 `citfaith_state.pt` 中。从 checkpoint 恢复训练时会自动加载，确保约束强度连续。

```
checkpoints/
└── citfaith_7B/
    └── global_step_30/
        ├── actor/
        ├── dataloader.pt
        └── citfaith_state.pt    # 拉格朗日乘子状态
```

CSR-Faith 的 `lambda_coverage` 和 `lambda_step_cfs` 会保存在同级 `csrfaith_state.pt` 中。

---

## 常见问题

**Q: CIT-Faith 增加了多少训练开销？**

约 1.25 倍。审查模型评估（SC + PR）约占 8-10% 额外时间（AWQ 量化 + vLLM 批量推理），反事实干预（CFS）约占 10-15%（每条 rollout 最多 3 次 prefix-continuation decode，每次仅生成 50 token）。

**Q: 不安装 vLLM 能否运行？**

可以。审查模型会自动回退到 dummy 模式（返回中性默认分数），CIT-Faith 退化为仅做对偶约束的软版本。但建议安装 vLLM 以获得完整效果。

**Q: 能否在非空间推理任务上使用？**

CIT-Faith 的审查 prompt 模板是为空间推理设计的（检查空间断言、坐标一致性等），但框架本身是通用的。替换 `reviewer.py` 中的 prompt 模板和 `counterfactual.py` 中的扰动策略即可适配其他结构化推理任务。

**Q: 如何只启用部分 CIT-Faith 组件？**

通过超参数控制：设置 `cfs_alpha=1.0` 可以关闭因果调制（g 恒等于 1），设置 `tau_sc=0.0, tau_pr=0.0` 可以使拉格朗日乘子始终为 0 从而关闭对偶约束。

---

## 引用

```bibtex
@misc{citfaith2026,
  title={CIT-Faith: Faithful Spatial Reasoning via Small-Large Model Collaborative Training with Counterfactual Intervention},
  year={2026},
}
```

## 致谢

本项目基于以下开源工作：
- [SpatialThinker](https://github.com/hunarbatra/SpatialThinker) — 3D 空间推理 MLLM
- [EasyR1](https://github.com/hiyouga/EasyR1) — 多模态 RL 训练框架
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) — 多模态大模型
