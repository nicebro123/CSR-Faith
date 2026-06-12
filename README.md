# Causal Spatial Rationales for Faithful Multimodal Reasoning

[![Python](https://img.shields.io/badge/python-%3E%3D3.9-blue)](#-环境安装)
[![Method](https://img.shields.io/badge/method-CSR--Faith-green)](#核心贡献)
[![Dataset](https://img.shields.io/badge/dataset-unchanged-orange)](#核心贡献)

**CSR-Faith** 是一个面向多模态空间推理的忠实 CoT 强化学习框架。它不修改原始数据集，而是从已有场景图标注中自动派生空间因果证据，训练模型生成必要、充分、紧凑的视觉推理，并用 step-level counterfactual intervention 验证推理步骤是否真的影响答案。

本仓库基于 [SpatialThinker](https://github.com/hunarbatra/SpatialThinker) / [EasyR1](https://github.com/hiyouga/EasyR1)，保留 **CIT-Faith** 作为可选对照路径，重点实现 **CSR-Faith: Causal Spatial Rationales**。

强化学习训练多模态空间推理时，思维链（CoT）容易沦为「装饰性文本」：看起来在解释，实际答案并不依赖它。CSR-Faith 把这个问题拆成三个可度量目标：

1. 自动构造空间因果证据：从已有 `<scene>` / `<answer>` 派生 rationale target。
2. 奖励忠实视觉推理：覆盖必要事实，避免无关事实，保持紧凑且充分。
3. 做步级因果验证：逐步干预 CoT，让策略模型续写答案，检查答案是否改变。

📚 深入文档：[方法/模块详解](docs/method_and_modules.md) · [设计文档](docs/causal_spatial_rationales_dev.md) · [论文设想](docs/paper_idea_causal_spatial_critic.md) · [critic 开发计划](docs/causal_spatial_critic_dev.md) · [逐步执行手册](docs/csrfaith_run_guide.md)

---

## 核心贡献

| 模块 | 作用 | 额外数据/模型 |
| --- | --- | --- |
| **Causal Spatial Evidence** | 从原数据字段里的 GT scene graph 自动抽取对象、关系和答案相关证据 | 不改数据集 |
| **CSR Reward** | 用 coverage / precision / compactness / sufficiency / necessity 奖励忠实空间 CoT | 不需要额外模型 |
| **Step-level CFS** | 对每个推理步做关系翻转、实体交换、步掩码，再前缀续写答案 | 使用当前策略模型 |
| **GRPO Integration** | 将 rationale score 与 step-CFS 注入 GRPO advantage，并保存/恢复拉格朗日乘子 | 训练框架内完成 |
| **Causal Spatial Critic** | 小型因果桥接模型，学习预测推理步被干预后答案是否改变，用于替代部分昂贵 step-CFS | 离线数据/训练/评估 + 可选 GRPO 接入已实现 |
| **CIT-Faith Baseline** | LLM Judge 评估 SC/PR + counterfactual CFS，可作为对照或组合实验 | 需审查模型(AWQ) |

---

## Causal Spatial Critic 离线闭环

当前已按 EVO-RAG 式阶段组织实现离线 bridge 训练链路：先生成 intervention-label JSONL，再训练轻量 critic，最后独立评估 checkpoint。

```bash
python3 scripts/build_causal_critic_dataset.py \
  --input-json rollouts.jsonl \
  --output cache/causal_critic/train.jsonl \
  --drop-invalid-labels

python3 scripts/train_causal_spatial_critic.py \
  --train-jsonl cache/causal_critic/train.jsonl \
  --output-dir ckpts/causal_spatial_critic

python3 scripts/evaluate_causal_spatial_critic.py \
  --critic-path ckpts/causal_spatial_critic \
  --eval-jsonl cache/causal_critic/train.jsonl
```

`rollouts.jsonl` 应包含原始 `problem` / `ground_truth` / `response`，以及在线 step-CFS 得到的 `counterfactual_answers`。原数据集仍不修改。训练时可设置 `CAUSAL_CRITIC_PATH` 后运行 `scripts/csrfaith_critic_smoke.sh` 或 `scripts/csrfaith_critic_7b_grpo.sh`，把 critic checkpoint 接入 CSR-GRPO。

## 快速开始

```bash
# 1. 安装
git clone https://github.com/nicebro123/CSR-Faith && cd CSR-Faith
pip install -r requirements.txt && pip install -e .

# 2. 配置本机（数据盘 / GPU）
cp scripts/env.local.example.sh scripts/env.local.sh && vim scripts/env.local.sh

# 3. 离线自检（无需 GPU）
python3 -m unittest discover -s tests

# 4. 链路冒烟（需 GPU + vLLM）
bash scripts/csrfaith_smoke.sh

# 5. 正式训练
bash scripts/csrfaith_7b_grpo.sh

# 6. 合并权重供评估
python3 scripts/model_merger.py --local_dir "$DATA_ROOT/ckpts/csrfaith_7B/global_step_75/actor"
```

---

## ✨ 特性

- **零数据改动**：监督信号全部从 batch 已有字段（`problem` / `ground_truth` 内嵌 `<scene>` `<answer>`）派生。
- **Rationale 评分**：coverage / precision / compactness / sufficiency / necessity / overall 六维。
- **步级因果干预**：关系翻转、实体交换、步掩码 → 前缀续写解码 → 归一化答案比对。
- **GRPO + 拉格朗日对偶约束**，乘子状态随 checkpoint 保存与恢复。
- **代码与数据/权重彻底分离**，机器专属配置外置，开箱即用的可复现脚手架。

### 方法一图流（CSR-Faith advantage）

```text
A_task = group_zscore(task_reward)          # 任务奖励（答案对错 + 空间分）
A_rat  = group_zscore(rationale_overall)    # 思维链是否覆盖必要空间证据
A_step = group_zscore(step_cfs_mean)        # 每步在反事实干预下是否真正影响答案

A_base = A_task + λ_cov · A_rat + λ_step · A_step       # 对偶约束注入
A_csr  = A_base · (α + (1-α) · step_cfs_mean)           # 步级因果调制
```

`λ_cov` / `λ_step` 由拉格朗日对偶上升自适应更新；`α` 为调制下界（默认 0.1）。

---

## 📁 目录结构（代码与数据分离）

代码仓库**只放代码**。模型权重、数据集、下载缓存、训练 checkpoint、日志**全部落到仓库之外的大盘目录** `DATA_ROOT`，并已被 `.gitignore` 排除，不会进版本库。

```
CSR-Faith/                      # ← 本仓库（只有代码）
├── verl/                       # 训练框架 + CSR/CIT 核心实现
│   ├── trainer/                #   core_algos.py（CSR advantage） / ray_trainer.py（训练循环集成）
│   └── utils/                  #   causal_rationale.py / step_causal.py / counterfactual.py / answer_normalization.py
├── scripts/                    # 训练 / 调试 / 评估脚本
│   ├── csrfaith_7b_grpo.sh     #   正式训练
│   ├── csrfaith_smoke.sh       #   最小链路冒烟
│   ├── env.local.example.sh    #   机器配置模板（复制为 env.local.sh 使用，后者不入库）
│   ├── debug_csr_batch.py      #   离线检查 CSR 派生监督
│   ├── build_csr_target_cache.py
│   └── model_merger.py         #   FSDP 分片 → HF 权重
├── tests/                      # 单元测试
├── docs/                       # 设计文档 + 执行手册 + 模块详解
├── requirements.txt / setup.py / config 见 scripts/config.yaml
└── README.md

$DATA_ROOT/                     # ← 仓库之外的大盘目录（不入库）
├── hf_cache/                   # HF_HOME：模型 + 数据集下载缓存
└── ckpts/csrfaith_7B/          # 训练 checkpoint
    └── global_step_*/{actor/, dataloader.pt, csrfaith_state.pt}
```

机器专属配置（`DATA_ROOT`、用哪几张 GPU、是否走镜像）写在 **`scripts/env.local.sh`**（已 gitignore），训练脚本启动时自动 `source`。每台机器维护自己的一份，互不干扰，也不污染公共代码。

---

## 🖥️ 硬件要求

| 配置 | 说明 |
| --- | --- |
| 推荐 | 2–4 × 80GB（A100 / H100 / H20）。默认脚本按 2 卡 + FSDP CPU offload 调好 |
| 最低 | 2 × 40GB（开 offload，降 `rollout.n` 与 `max_num_batched_tokens`） |
| 冒烟 | 1 × 24GB 可跑 3B（`MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct N_GPUS=1 bash scripts/csrfaith_smoke.sh`） |

依赖 `vllm==0.8.0` 做策略模型的前缀续写解码（step-level CFS）；无 vLLM 时 rationale 评分仍可用，step CFS 会安全降级为「无调制」。

---

## 🔧 环境安装

```
Python >= 3.9   CUDA 12.x
torch >= 2.1    vllm == 0.8.0   flash-attn >= 2.4.3
ray   transformers >= 4.49   datasets   tensordict
```

```bash
git clone https://github.com/nicebro123/CSR-Faith && cd CSR-Faith
pip install -r requirements.txt
pip install -e .
```

> `flash-attn` 编译慢，若失败可 `pip install flash-attn --no-build-isolation`。

---

## ⚙️ 配置本机环境（关键一步）

```bash
cp scripts/env.local.example.sh scripts/env.local.sh
vim scripts/env.local.sh
```

```bash
# scripts/env.local.sh（本机实际值，已 gitignore）
export DATA_ROOT="../csr_faith_assets"      # 数据/权重/checkpoint 根目录（仓库同级目录）
export HF_HOME="${DATA_ROOT}/hf_cache"      # HuggingFace 下载缓存
export CKPT_ROOT="${DATA_ROOT}/ckpts"       # checkpoint 输出目录
export CUDA_VISIBLE_DEVICES="0,1"           # 用哪几张卡
export N_GPUS=2                             # 卡数（须与上面一致）
export WANDB_MODE=offline                   # 日志离线，不需登录
export HF_ENDPOINT="https://huggingface.co" # 国内无法直连时可改为 https://hf-mirror.com
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export DATA_FILE="hunarbatra/STVQA-7K"
export CAUSAL_CRITIC_PATH="${CKPT_ROOT}/causal_spatial_critic"
```

> 不创建 `env.local.sh` 也能跑：用脚本内置默认值（`DATA_ROOT=../csr_faith_assets`、`CUDA_VISIBLE_DEVICES=1,2`、`N_GPUS=2`）。任何变量都可被 `env.local.sh` 或命令行覆盖。

---

## 📦 数据与模型准备

首次训练会按 `env.local.sh` 设置自动下载到 `$HF_HOME`。更推荐提前统一预取，避免训练时卡在下载：

```bash
bash scripts/prepare_assets.sh
```

该脚本会读取 `scripts/env.local.sh`，创建 `DATA_ROOT/HF_HOME/CKPT_ROOT`，并下载 `MODEL_PATH` 与 `DATA_FILE`。如果只想下载其中一个：

```bash
bash scripts/prepare_assets.sh --model-only
bash scripts/prepare_assets.sh --data-only
```

| 资源 | 名称 | 说明 |
| --- | --- | --- |
| 策略模型 | `Qwen/Qwen2.5-VL-7B-Instruct` | 被训练的多模态模型（3B 版可低显存调试） |
| 数据集 | `hunarbatra/STVQA-7K` | `@train` / `@val` split，`ground_truth` 内嵌 `<scene>` `<answer>` |
| 审查模型 | `Qwen/Qwen2.5-7B-Instruct-AWQ` | **仅 CIT-Faith 需要；CSR-Faith 不用** |

---

## ✅ 离线自检（无需 GPU，强烈建议先做）

```bash
# 单元测试（缺 numpy/torch 时部分自动跳过）
python3 -m unittest discover -s tests
# 预期：OK (skipped=NN)

# 离线检查单条样本的 target / rationale 评分 / step 干预（不启动模型/Ray）
python3 scripts/debug_csr_batch.py \
    --problem "Where is the chair relative to the table?" \
    --ground-truth '<scene>{"objects":[{"id":"chair.1","bbox":[10,20,100,200]},{"id":"table.1","bbox":[120,30,300,260]}],"relationships":[{"subject":"chair.1","predicate":"left of","object":"table.1"}]}</scene><answer>left</answer>' \
    --response '<observe>..</observe><scene>{"objects":[{"id":"chair.1","bbox":[10,20,100,200]},{"id":"table.1","bbox":[120,30,300,260]}]}</scene><think>1. chair.1 is left. 2. table.1 is right.</think><answer>left</answer>' \
    --jsonl
```

---

## 🚀 训练

### 1) 冒烟测试（先确认链路能跑通，需 GPU + Ray + vLLM）

```bash
bash scripts/csrfaith_smoke.sh
```

固定为 `max_steps=2`、`rollout.n=2`、小 batch、关闭验证与保存。**通过标准**：控制台出现 `csr/rationale_*` 指标且有限、`csr/step_interventions_mean > 0`、无 shape mismatch / DataProto 冲突。

### 2) 正式训练

```bash
bash scripts/csrfaith_7b_grpo.sh
```

- 启动机制：脚本内部调用 `python3 -m verl.trainer.main config=scripts/config.yaml ...`；**单机自动初始化本地 Ray，无需手动 `ray start`**。
- **必须在仓库根目录执行**（`-m verl.trainer.main` 与 `config=scripts/config.yaml` 都相对根目录）。
- 命令行可追加任意覆盖：
  ```bash
  bash scripts/csrfaith_7b_grpo.sh trainer.max_steps=150 worker.rollout.n=4
  ```

显存不足时依次降：`worker.rollout.max_num_batched_tokens` → `worker.rollout.n` → `algorithm.csr_max_steps` → `data.max_response_length`。

### 训练监控（关注这些指标的趋势）

| 指标 | 含义 | 期望 |
| --- | --- | --- |
| `csr/rationale_coverage_mean` | 思维链覆盖必要空间证据的比例 | 上升，逼近 `tau_coverage=0.7` |
| `csr/rationale_precision_mean` | 生成事实命中 target 的比例 | 上升（抑制装饰性事实） |
| `csr/rationale_overall_mean` | 六维加权综合分 | 稳定上升 |
| `csr/step_cfs_mean` | 步级反事实下答案改变率（核心因果指标） | 上升 |
| `csr/step_cfs_valid_ratio` | 有效 step CFS 样本比例 | > 0.4 正常；过低说明格式/干预生成脆弱 |
| `csr/lambda_coverage` / `csr/lambda_step_cfs` | 拉格朗日乘子 | batch 低于阈值时上升 |

---

## 💾 输出与 Checkpoint

```
$DATA_ROOT/ckpts/csrfaith_7B/
├── latest_global_step.txt
└── global_step_{25,50,75}/
    ├── actor/                 # FSDP 分片权重
    ├── dataloader.pt
    └── csrfaith_state.pt      # CSR 拉格朗日乘子 (lambda_coverage / lambda_step_cfs)
```

- 默认 `save_freq=25`、`save_limit=3`：第 25/50/75 步各存一次，保留最近 3 个。
- 日志：控制台 + 本地 `wandb/`（默认离线，`wandb sync` 可后传）。
- 断点续训：`bash scripts/csrfaith_7b_grpo.sh trainer.load_checkpoint_path=$DATA_ROOT/ckpts/csrfaith_7B/global_step_50`，CSR 乘子自动恢复。

---

## 🔀 合并权重（推理/评估前）

FSDP 分片需合并成标准 HF 权重：

```bash
python3 scripts/model_merger.py --local_dir $DATA_ROOT/ckpts/csrfaith_7B/global_step_75/actor
```

输出在 `.../actor/huggingface/`，该目录才能被 `transformers` / `vllm` 直接加载。

---

## 📊 评估

评估脚本来自上游 [SpatialThinker](https://github.com/hunarbatra/SpatialThinker)（`evaluation/evals.py`）。**必须在其 `evaluation/` 目录内运行**（脚本用 `from templates import ...`）：

```bash
cd ../SpatialThinker/evaluation
python3 evals.py \
    --dataset blink-spatial \
    --template spatial_thinker \
    --model_path ../../csr_faith_assets/ckpts/csrfaith_7B/global_step_75/actor/huggingface \
    --processor_name Qwen/Qwen2.5-VL-7B-Instruct \
    --cuda 0 --batch_size 4
```

- `--template spatial_thinker` 对齐训练的 observe/scene/think/answer 格式，**必须用它**。
- 推荐基准：`stvqa`(in-domain) / `blink-spatial` / `cv-bench` / `3dsrbench`。

---

## 🧪 消融

| 方法 | 开关 |
| --- | --- |
| baseline GRPO | `algorithm.enable_citfaith=False algorithm.enable_csrfaith=False` |
| CIT-Faith | `algorithm.enable_citfaith=True algorithm.enable_csrfaith=False` |
| CSR rationale only | `algorithm.enable_csrfaith=True algorithm.csr_max_steps=0`（关步级 CFS） |
| Full CSR-Faith | 默认 `scripts/csrfaith_7b_grpo.sh` |
| CIT + CSR | 两个 enable 同时 True |

---

## 🩺 故障排查

| 现象 | 原因 / 解决 |
| --- | --- |
| `ModuleNotFoundError: verl` | 未在仓库根目录运行，或未 `pip install -e .` |
| 训练卡在下载 | 未设 `HF_HOME` / 网络不通；提前预取，或在 `env.local.sh` 设 `HF_ENDPOINT=https://hf-mirror.com` |
| OOM | 依次降 `max_num_batched_tokens` → `rollout.n` → `csr_max_steps` → `max_response_length`；确认 `offload_params/optimizer=true` |
| `csr/step_cfs_valid_ratio` 趋近 0 | 多数 rollout 缺 `<think>`/`<answer>` 或干预无法生成；用 `scripts/debug_csr_batch.py` 检查真实 rollout |
| 启动要求 wandb 登录 | `env.local.sh` 设 `WANDB_MODE=offline`，或 CLI `trainer.logger='["console"]'` |
| 评估 `ModuleNotFoundError: templates` | 未在 `evaluation/` 目录内运行 `evals.py` |
| `from_pretrained` 加载失败 | 评估要指向合并后的 `actor/huggingface/`，而非 `actor/`（后者是 FSDP 分片） |

---

## ☑️ 复现核对清单

- [ ] `pip install -e .` 成功，`python3 -m unittest discover -s tests` 通过
- [ ] 已 `cp scripts/env.local.example.sh scripts/env.local.sh` 并设好 `DATA_ROOT` / GPU
- [ ] `bash scripts/csrfaith_smoke.sh` 跑通，出现 `csr/*` 指标
- [ ] 正式训练后 `$DATA_ROOT/ckpts/csrfaith_7B/global_step_75/` 含 `actor/` + `csrfaith_state.pt`
- [ ] `model_merger.py` 产出 `actor/huggingface/`，可被评估脚本加载

---

## 致谢

- [SpatialThinker](https://github.com/hunarbatra/SpatialThinker) — 3D 空间推理 MLLM 与评估
- [EasyR1](https://github.com/hiyouga/EasyR1) — 多模态 RL 训练框架
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) — 多模态大模型

## 引用

```bibtex
@misc{csrfaith2026,
  title  = {CSR-Faith: Causal Spatial Rationales for Faithful Multimodal Spatial Reasoning},
  year   = {2026}
}
```
