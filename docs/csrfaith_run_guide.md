# CSR-Faith 执行文档（数据 / 权重 / 模型 / 训练 / 评估）

本文件是 CSR-Faith（含 CIT-Faith 基座）从零到训练的端到端操作手册。所有命令里的数据集名、模型名、超参数均取自仓库现有脚本（`scripts/*.sh`、`scripts/config.yaml`），不是占位符。

> 适用代码状态：Phase 1（确定性 CSR 评分 + advantage）+ Phase 2（step-level CFS）已完整实现。Phase 3（图验证器）未实现，不在本手册范围。

---

## 0. 资源清单（先看这张表）

| 类别 | 名称 | 来源 | 大小（约） | 用途 |
| --- | --- | --- | --- | --- |
| 策略模型 | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace | ~16 GB | 被训练的多模态策略模型（7B 主实验） |
| 策略模型（小） | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace | ~7 GB | 3B 调试/低显存（`spatialthinker_3b_grpo.sh`） |
| 审查模型 | `Qwen/Qwen2.5-7B-Instruct-AWQ` | HuggingFace | ~6 GB | **仅 CIT-Faith 需要**；CSR-Faith 不需要 |
| 主数据集 | `hunarbatra/STVQA-7K` | HuggingFace Datasets | 含图像，数 GB | CSR / CIT 训练默认数据（`@train` / `@val`） |
| 备用数据集 | `hunarbatra/Clevr_SAT_3k` | HuggingFace Datasets | 较小 | `config.yaml` 默认值，可用于快速调试 |

**关键结论：跑 CSR-Faith 只需要「策略模型 + 数据集」。审查模型（AWQ）是 CIT-Faith 专用的，CSR 脚本默认 `enable_citfaith=False`，不会加载它。**

---

## 1. 环境准备

### 1.1 硬件 / 系统要求

- GPU：7B 训练建议 ≥ 4×A100/H100（80GB）。Smoke 测试可用 1 张 ≥ 24GB 显存卡跑 3B。
- CUDA：与 `vllm==0.8.0` / `flash-attn` 匹配的 CUDA 12.x。
- Python：`>= 3.9`（仓库 `setup.py` 要求 `>=3.9.0`，ruff target `py39`）。

### 1.2 安装

```bash
# 1) 创建环境
conda create -n csrfaith python=3.10 -y
conda activate csrfaith

# 2) 进入项目
cd /path/to/cit-faith

# 3) 安装依赖 + 本项目（editable）
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` 关键项：`torch`、`vllm==0.8.0`、`flash-attn>=2.4.3`、`ray`、`tensordict`、`transformers>=4.49.0`、`datasets`、`wandb`、`flashinfer-python`。

> `flash-attn` 编译较慢，若失败可单独 `pip install flash-attn --no-build-isolation`。

### 1.3 环境变量

**代码与数据/权重分离（本项目采用）：** 代码仓库只放代码；模型/数据集下载缓存和 checkpoint 默认落到仓库同级目录 `../csr_faith_assets`。布局：

```
CIT-Faith/                    ← 代码（当前仓库）
../csr_faith_assets/          ← 其他（数据 / 权重 / checkpoint）
    ├── hf_cache/             ← HF_HOME：模型 + 数据集下载缓存
    └── ckpts/                ← 训练 checkpoint 输出
```

训练脚本和下载脚本都会读取 `scripts/env.local.sh`，统一使用 `DATA_ROOT`、`HF_HOME`、`CKPT_ROOT`、`MODEL_PATH`、`DATA_FILE`。想换位置只改 `env.local.sh`，或外部 `export DATA_ROOT=...` 覆盖。

```bash
cp -n scripts/env.local.example.sh scripts/env.local.sh
vim scripts/env.local.sh
```

---

## 2. 下载模型权重和数据集

训练脚本里写的是 HF repo id（如 `Qwen/Qwen2.5-VL-7B-Instruct`），首次运行会自动下载到 `HF_HOME`。更推荐提前统一预取，避免训练时卡在下载：

```bash
python3 -m pip install -U huggingface_hub
bash scripts/prepare_assets.sh
```

只下载模型或只下载数据：

```bash
bash scripts/prepare_assets.sh --model-only
bash scripts/prepare_assets.sh --data-only
```

国内网络可在 `scripts/env.local.sh` 里设置：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### 使用本地路径（离线集群）

`MODEL_PATH` 和 `DATA_FILE` 支持改成本地目录：

```bash
export MODEL_PATH="../csr_faith_assets/local_models/Qwen2.5-VL-7B-Instruct"
export DATA_FILE="../csr_faith_assets/local_datasets/STVQA-7K"
```

---

## 3. 数据集字段约定

CSR-Faith 训练**不修改数据集**，监督信号全部从 batch 已有字段（`problem` / `ground_truth` 内嵌 `<scene>`、`<answer>`）派生。

训练脚本里数据写法为 `名称@split`：

```bash
data.train_files="hunarbatra/STVQA-7K@train"
data.val_files="hunarbatra/STVQA-7K@val"
data.answer_key="answer"               # STVQA-7K 的 answer 字段包含 scene graph 与 answer tag
data.image_key="images"
```

> 字段约定：`data.answer_key` 决定哪个列会进入 `ground_truth`。CSR/spatial_sgg 应使用内嵌 `<scene>{json}</scene>` 与 `<answer>...</answer>` 的 `answer` 列；`answer_option_text` 只有纯答案文本，不足以构造 CSR target。如果某条样本解析不到 scene graph，则该样本 CSR target 为空，仅保留任务奖励（不会报错）。

### （可选）预构建 CSR target 缓存

不启动训练，离线把每条样本的 target 证据先算好并落盘，便于检查派生质量：

```bash
# 方式 A：从 HF 数据集直接构建（需本地装好 datasets）
python3 scripts/build_csr_target_cache.py \
    --data hunarbatra/STVQA-7K@train \
    --ground-truth-key answer \
    --output cache/csr_targets/stvqa/train.jsonl

# 方式 B：从本地 json 构建
python3 scripts/build_csr_target_cache.py \
    --input-json sample.json \
    --output cache/csr_targets/stvqa/train.jsonl
```

输出每条含 `target_source` / `target_confidence` / `target_fact_count`，可统计 `target_coverage`（有多少样本成功派生出 target）。

---

## 4. 离线验证（无需 GPU，强烈建议先做）

在烧 GPU 之前，先确认 CSR 派生逻辑与代码本身没问题。这些都在 CPU 上秒级完成：

```bash
# 4.1 单元测试（51 个，本地无训练依赖时 12 个自动跳过）
python3 -m unittest discover -s tests

# 4.2 运行前预检：依赖、关键文件、CSR 配置入口
python3 scripts/check_csrfaith_ready.py --no-fail

# 4.3 离线检查单条样本：target / rationale score / step intervention
python3 scripts/debug_csr_batch.py \
    --problem "Where is the chair relative to the table?" \
    --ground-truth '<scene>{"objects":[{"id":"chair.1","bbox":[10,20,100,200]},{"id":"table.1","bbox":[120,30,300,260]}],"relationships":[{"subject":"chair.1","predicate":"left of","object":"table.1"}]}</scene><answer>left</answer>' \
    --response '<observe>...</observe><scene>{"objects":[{"id":"chair.1","bbox":[10,20,100,200]},{"id":"table.1","bbox":[120,30,300,260]}]}</scene><think>1. chair.1 is left. 2. table.1 is right.</think><answer>left</answer>' \
    --jsonl

# 4.4 批量离线检查（输入 json 列表）
python3 scripts/debug_csr_batch.py --input-json sample.json --jsonl
```

通过标准：测试全绿；预检没有缺关键文件或 CSR 配置入口；debug 输出里 `target.source` 不为 `empty`，`rationale_score` 各项在 [0,1]，`step_interventions` 非空且保留了全部标签。若本地还没安装训练依赖，预检会列出缺失模块，此时先补依赖再跑 GPU smoke。

---

## 5. 训练 Smoke（最小链路，先确认能跑通）

正式训练前用极小配置验证「数据 → rollout → CSR 评分 → advantage → 更新 → checkpoint」整条链路。需要 GPU + Ray + vLLM 环境。

```bash
# 默认单卡、7B；可用 env 覆盖
bash scripts/csrfaith_smoke.sh

# 低显存改 3B + 显式单卡
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct N_GPUS=1 bash scripts/csrfaith_smoke.sh
```

该脚本固定为 `max_steps=2`、`rollout.n=2`、小 batch、`max_response_length=512`、关闭验证、中间保存和 KL/ref policy、`csr_max_steps=2`；关闭验证时不会加载 val dataloader。默认 `trainer.save_freq=-1` 会跳过 checkpoint 保存，如需验证存取链路可显式设置 `trainer.save_freq=1`。

**通过标准（对照 `docs/csrfaith_implementation_status.md`）：**
- 启动无 OmegaConf key 报错
- 控制台出现 `csr/rationale_*` 指标且为有限值
- 对格式良好的 rollout，`csr/step_interventions_mean > 0`
- 无 `DataProto` union 冲突
- `_balance_batch` 之后无 shape mismatch
- `actor_rollout_wg.generate_continuations` 无报错

### checkpoint 存取冒烟

```bash
bash scripts/csrfaith_smoke.sh \
    trainer.save_freq=1 \
    trainer.save_limit=3 \
    trainer.save_checkpoint_path=ckpts/csrfaith_smoke
```

应生成：

```
ckpts/csrfaith_smoke/global_step_1/actor/
ckpts/csrfaith_smoke/global_step_1/dataloader.pt
ckpts/csrfaith_smoke/global_step_1/csrfaith_state.pt   # lambda_coverage / lambda_step_cfs
```

---

## 6. 正式训练

### 6.0 启动命令与机制

启动分两层，日常用顶层即可：

```bash
# ① 顶层（推荐）：先 cd 到项目根，再跑脚本
cd /Users/quanquan/Desktop/cit-faith
bash scripts/csrfaith_7b_grpo.sh
```

脚本内部等价于 ② 底层的 Python 模块调用（`scripts/csrfaith_7b_grpo.sh` 第 8 行起）：

```bash
python3 -m verl.trainer.main \
    config=scripts/config.yaml \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
    ... (其余 overrides) ...
```

机制（来自 `verl/trainer/main.py`）：
- 入口 `main()` 用 `config=scripts/config.yaml` 作基底，命令行参数逐个覆盖它。
- **单机自动 `ray.init()` 拉本地 Ray 集群，无需手动 `ray start`**（`main.py:100-102`）。
- 随后起一个 `@ray.remote` Runner → `trainer.init_workers()` → `trainer.fit()`。

### 6.1 启动前必须满足的 3 个条件

1. **必须在项目根目录执行**：`-m verl.trainer.main`（包路径）和 `config=scripts/config.yaml`（相对路径）都相对项目根 `/Users/quanquan/Desktop/CIT-Faith`。在别处执行会找不到模块或配置。
2. **单机不需要 `ray start`**：代码会自动起本地集群。**只有多机**（`trainer.nnodes>1`）才需要先在各节点手动组 Ray 集群，再用本命令连接。
3. **GPU 默认使用物理卡 1、2（两张 H20）**：脚本顶部默认 `CUDA_VISIBLE_DEVICES=1,2` 且 `N_GPUS=2`，进程内映射为 `cuda:0/cuda:1`。如需换卡，可外部覆盖：`CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 bash scripts/csrfaith_7b_grpo.sh`。

> ⚠️ `scripts/runtime_env.yaml`（设 `TORCH_NCCL_AVOID_RECORD_STREAMS` 的那个）**不会被上面的 bash 启动路径自动加载**——`main.py` 在 `ray.init()` 里硬编码了自己的 env_vars（`TOKENIZERS_PARALLELISM`、`NCCL_DEBUG`）。该 yaml 仅在用 `ray job submit --runtime-env scripts/runtime_env.yaml` 提交时才生效。若需要那个 NCCL 变量，请手动 `export` 或走 `ray job submit`。

### 6.2 脚本核心配置

脚本核心配置（来自 `scripts/csrfaith_7b_grpo.sh`）：

```bash
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
DATA_FILE=${DATA_FILE:-hunarbatra/STVQA-7K}
N_GPUS=${N_GPUS:-2}

worker.reward.score_function=spatial_sgg     # 空间任务奖励
trainer.n_gpus_per_node=${N_GPUS}
trainer.save_freq=25                         # 默认每 25 step 保存一次
trainer.save_limit=3                         # 默认保留 25/50/75 三个 CSR checkpoint
trainer.max_steps=75
trainer.total_episodes=75
worker.rollout.n=6                           # 每个 prompt 采样数
data.max_prompt_length=6144
data.max_response_length=2048

algorithm.enable_citfaith=False              # CSR 单独跑，不开 CIT
algorithm.enable_csrfaith=True
algorithm.csr_target_max_relations=4
algorithm.csr_target_max_objects=6
algorithm.csr_coverage_weight=0.4
algorithm.csr_precision_weight=0.2
algorithm.csr_compactness_weight=0.1
algorithm.csr_sufficiency_weight=0.2
algorithm.csr_necessity_weight=0.1
algorithm.csr_step_cfs_alpha=0.1
algorithm.csr_max_steps=6
algorithm.csr_max_step_interventions=1
algorithm.tau_coverage=0.7
algorithm.tau_step_cfs=0.5
algorithm.dual_lr=0.01
```

### 跑正式训练前必须知道的两点

1. **中间 checkpoint 默认会保存并保留**：`csrfaith_7b_grpo.sh` 显式设置 `trainer.save_freq=25` 和 `trainer.save_limit=3`，因此 75 step 训练默认保留 `global_step_25`、`global_step_50`、`global_step_75`。如果追加 `trainer.save_freq=-1`，训练结束不会额外保存 checkpoint。
   - 默认路径是 `$DATA_ROOT/ckpts/csrfaith_7B/`。
   - 想减少存档可追加 `trainer.save_freq=-1`；想保留更多续训点可追加 `trainer.save_freq=15 trainer.save_limit=3`。
2. **wandb 默认离线**：脚本顶部设置了 `WANDB_MODE=offline`，日志会写到本地 `wandb/` 目录，不要求联网登录。也可以命令行追加 `trainer.logger='["console"]'` 完全关掉 wandb。

### 常用覆盖

```bash
# 改 GPU 数 / 实验名 / 保存路径 / 开启中间存档（命令行追加即可覆盖脚本内值）
bash scripts/csrfaith_7b_grpo.sh \
    trainer.n_gpus_per_node=8 \
    trainer.experiment_name=csrfaith_7B_run2 \
    trainer.save_checkpoint_path=ckpts/csrfaith_7B_run2 \
    trainer.save_freq=15 \
    trainer.save_limit=3 \
    trainer.logger='["console"]'

# 断点续训
bash scripts/csrfaith_7b_grpo.sh \
    trainer.load_checkpoint_path=ckpts/csrfaith_7B/global_step_30
```

> 续训时 `csrfaith_state.pt` 会被自动加载，恢复 `lambda_coverage` / `lambda_step_cfs`，保证约束强度连续（见 `ray_trainer.py` load 逻辑）。

### 显存 / 吞吐调参（资源不足时）

- 降 `worker.rollout.n`（6→4/2）、降 `data.max_response_length`
- 降 `worker.rollout.max_num_batched_tokens`
- `csr_max_steps`（6→3）、`csr_max_step_interventions`（保持 1）降低 step-CFS 续写开销
- `config.yaml` 里开 `worker.actor.offload.offload_params/optimizer=true`

---

## 7. 训练监控

默认 logger 含 console，正式可加 wandb（`config.yaml` 的 `trainer.logger`、`project_name`）。重点关注 CSR 指标：

| 指标 | 含义 | 期望趋势 |
| --- | --- | --- |
| `csr/target_confidence_mean` | 自动 target 派生置信度 | 越高越稳定 |
| `csr/rationale_coverage_mean` | `<think>/<scene>` 覆盖 target facts 比例 | 上升，逼近 `tau_coverage=0.7` |
| `csr/rationale_precision_mean` | 生成 facts 命中 target 比例 | 上升，抑制装饰性事实 |
| `csr/rationale_overall_mean` | 加权综合分 | 稳定上升 |
| `csr/step_cfs_mean` | step 级反事实答案改变率 | 上升（核心因果指标） |
| `csr/step_cfs_valid_ratio` | 有效 step CFS 样本比例 | > 0.4 为正常；偏低说明格式/干预生成脆弱 |
| `csr/step_interventions_mean` | 每 rollout 干预数 | 监控开销 |
| `csr/lambda_coverage` / `csr/lambda_step_cfs` | 拉格朗日乘子 | batch 均值低于阈值时上升 |

排错信号：`step_cfs_valid_ratio` 持续接近 0 → 多数 rollout 缺 `<think>`/`<answer>` 或干预无法生成；先回到第 4 步用 `debug_csr_batch.py` 检查真实 rollout 文本。

---

## 8. Checkpoint 合并与导出

FSDP 分片 checkpoint 需合并成标准 HF 权重才能用于推理/评估：

```bash
python3 scripts/model_merger.py --local_dir $DATA_ROOT/ckpts/csrfaith_7B/global_step_75/actor
```

- 输入 `--local_dir` = actor 目录（内含 `model_world_size_*_rank_*.pt` FSDP 分片）。
- **输出固定写到子目录 `{actor}/huggingface/`**（见 `model_merger.py` 的 `save_pretrained(hf_path)`）。
- 因此**可被 `transformers` / `vllm` 加载的模型路径是 `$DATA_ROOT/ckpts/csrfaith_7B/global_step_75/actor/huggingface/`**。下游评估要用这个，而不是 `actor/` 本身（`actor/` 里只是分片，无法 `from_pretrained`）。

---

## 9. 评估

评估脚本来自上游 **SpatialThinker** 仓库（与 `cit-faith` 同级），路径：
`/Users/quanquan/Desktop/SpatialThinker-main/evaluation/evals.py`。
本仓库不内置 `evaluation/`，用合并后的权重在该脚本里跑基准即可。

### 9.1 前置

```bash
# 评估脚本额外依赖（若未装）
pip install mathruler qwen-vl-utils python-dotenv
# flash-attn 必需：脚本写死 attn_implementation="flash_attention_2"
```

### 9.2 运行（关键：必须在 evaluation/ 目录内执行）

`evals.py` 顶部是 `from templates import SPATIAL_THINKER_TEMPLATE`（非包内相对导入），
因此**必须先 `cd` 进 `evaluation/`**，否则会 `ModuleNotFoundError: templates`。

```bash
cd ../SpatialThinker-main/evaluation

python3 evals.py \
    --dataset blink-spatial \
    --template spatial_thinker \
    --model_path ../../csr_faith_assets/ckpts/csrfaith_7B/global_step_75/actor/huggingface \
    --processor_name Qwen/Qwen2.5-VL-7B-Instruct \
    --cuda 0 \
    --batch_size 4
```

- `--template spatial_thinker`：**必须用这个**，它对齐 CSR/CIT 训练的 observe/scene/think/answer 格式（见 `templates.py` 的 `SPATIAL_THINKER_TEMPLATE`）。用别的模板会偏离训练分布。
- `--model_path`：指向第 8 步合并后的 **`huggingface` 子目录**（`.../actor/huggingface`），不是 `actor/` 本身——后者是 FSDP 分片，`from_pretrained` 无法加载。
- `--processor_name`：合并目录若缺 processor/tokenizer 文件，显式指回基座 `Qwen/Qwen2.5-VL-7B-Instruct` 最稳妥。
- `--cuda N`：单卡推理（脚本用 `device_map=cuda:N`）。
- `--batch_size>1`：走批量生成；`--num_samples K`：只评前 K 条；`--resume`：从已有输出续跑。

输出写到 `./evaluation/outputs/{dataset}_{模型名}.json`，并在日志打印 `accuracy ± std_err`（含分任务子项，如 CV-Bench 的 Count/Relation/Depth/Distance）。

### 9.3 推荐基准

| `--dataset` | 说明 | 备注 |
| --- | --- | --- |
| `stvqa` | **In-domain**：`hunarbatra/STVQA-7K` 的 `val` split | 与训练同源，直接看泛化 |
| `blink-spatial` | BLINK 空间关系 | 论文常用 |
| `blink-depth` | BLINK 相对深度 | |
| `cv-bench` / `cv-bench-2D` / `cv-bench-3D` | CV-Bench 空间 | 分 2D/3D 子任务 |
| `3dsrbench` | 3D 空间推理 | 含 height/location/orientation/multi_object |
| `realworld_qa` / `spatialbench` / `robospatial` / `mmvp` | 其他空间/真实场景 | 按需 |

> 评估按 MCQ 精确匹配 + `mathruler.grade_answer` 双重判定（见 `evals.py` 的 `process_response` / `extract_answer`），对答案大小写/括号/选项字母都做了归一化。

### 9.4 基线对照

把同一组命令分别指向不同 checkpoint（baseline GRPO / CIT-Faith / CSR-Faith 的合并权重），用同一 `--dataset --template` 跑，即可得到论文对照表的 Accuracy / Spatial score 列。

### 论文级消融（来自 dev 文档）

| 方法 | 开关 |
| --- | --- |
| baseline GRPO | `enable_citfaith=False enable_csrfaith=False` |
| CIT-Faith | `enable_citfaith=True enable_csrfaith=False` |
| CSR rationale only | `enable_csrfaith=True csr_max_steps=0`（关 step-CFS） |
| CSR step CFS only | `enable_csrfaith=True` + rationale 权重置 0 |
| Full CSR-Faith | 默认 `csrfaith_7b_grpo.sh` |
| CIT + CSR | 两个 enable 同时 True |

---

## 10. 端到端最短路径（TL;DR）

```bash
# 1. 装环境
conda create -n csrfaith python=3.10 -y && conda activate csrfaith
cd /path/to/CIT-Faith && pip install -r requirements.txt && pip install -e .

# 2. 预取模型 + 数据（可跳过，训练会自动下载）
bash scripts/prepare_assets.sh

# 3. 离线自检（无 GPU）
python3 -m unittest discover -s tests
python3 scripts/debug_csr_batch.py --input-json sample.json --jsonl

# 4. GPU 链路冒烟
bash scripts/csrfaith_smoke.sh

# 5. 正式训练
bash scripts/csrfaith_7b_grpo.sh

# 6. 合并权重
python3 scripts/model_merger.py --local_dir $DATA_ROOT/ckpts/csrfaith_7B/global_step_75/actor
```

---

## 附：常见问题

**Q: CSR-Faith 必须下载 AWQ 审查模型吗？**
不需要。审查模型只服务 CIT-Faith；CSR 脚本 `enable_citfaith=False`。

**Q: 没有 vLLM 能跑吗？**
step-level CFS 依赖策略模型 prefix-continuation decode（经 vLLM）。无 vLLM 时 Phase 1 的 rationale 评分仍可用，但 step CFS 会因无法续写而失效（valid 比例趋 0，模块按"无调制"安全降级，不会崩）。建议安装 `vllm==0.8.0`。

**Q: 数据集解析不到 scene graph 会报错吗？**
不会。该样本 CSR target 为空，仅保留任务奖励，训练继续。

**Q: 改了哪些参数才是真正"关掉"某模块？**
`csr_max_steps=0` 关 step-CFS；把 `csr_*_weight` 调成只剩 coverage 可近似"仅覆盖"；`enable_csrfaith=False` 完全回退到原始 GRPO/CIT 路径。
</content>
</invoke>
