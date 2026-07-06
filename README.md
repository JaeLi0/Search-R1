<div align="center">
  <img src="public/logo.png" alt="Search-R1 logo" width="200"/>

  # Search-R1 (Enhanced)

  **Reinforcement-learning agents that learn to *think → search → answer*, built on [veRL](https://github.com/volcengine/verl).**

  [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#quickstart)
  [![Built on veRL](https://img.shields.io/badge/built%20on-veRL-orange.svg)](VERL_README.md)
  [![PPO%2FGRPO](https://img.shields.io/badge/RL-PPO%20%7C%20GRPO-brightgreen.svg)](#quickstart)
</div>

<p align="right"><a href="#english">English</a> | <a href="#中文">中文</a></p>

---

<a id="english"></a>
## English

### Overview

Search-R1 trains an LLM to interleave reasoning and web/wiki search via reinforcement learning (PPO / GRPO). At each turn the model either issues a query (`<search>query</search>`) or commits to an answer (`<answer>...</answer>`); retrieved passages are injected back as `<information>...</information>` and masked out of the training loss so the model isn't penalized/rewarded for text it didn't generate.

This repo is a second-development build on top of the open-source [Search-R1](https://github.com/PeterGriffinJin/Search-R1) / [veRL](https://github.com/volcengine/verl) stack, adding:

- **Hybrid retrieval (BM25 + Dense, RRF fusion)** — see measured results below
- **Multi-component reward shaping** — decomposes the original single, end-of-sequence EM reward into format / retrieval-hit / efficiency sub-rewards (capped below the correctness reward) to densify the training signal
- **GRPO advantage enhancement** — trajectory-quality weighting + in-group z-score normalization + a FIFO replay buffer of successful trajectories
- **Training data augmentation** — query expansion, difficulty stratification, intent classification, quality filtering, and contrastive sample construction, implemented as an optional pre-processing stage

### Architecture

<img src="public/main.png" alt="architecture" width="720"/>

```
Think → <search> query </search> → retrieval service (BM25 / Dense / RRF) → <information> docs </information> → Think → ... → <answer>
```

The RL loop is driven by `RayPPOTrainer` (Actor/Rollout/Ref/Critic on FSDP + vLLM); the multi-turn agent logic lives in `search_r1/llm_agent/generation.py`; retrieval is served over a FastAPI endpoint (`POST /retrieve`).

### Installation

```bash
# 1. Environment
conda create -n search-r1 python=3.10 -y && conda activate search-r1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e .
pip install pyserini faiss-gpu fastapi uvicorn sentence-transformers huggingface_hub

# 2. Data (corpus + FAISS index + preprocessed NQ/HotpotQA, ~20GB)
python scripts/download_data.py --save_path ./data
```

Hardware: 8× GPU with ≥24GB VRAM recommended (A100/A800 80GB for the default 3B/7B configs); CUDA 12.1, Python 3.10.

### Usage

**1. Start the retrieval service** (pick one)

```bash
# dense (E5 + FAISS)
python search_r1/search/retrieval_server.py \
    --index_path data/index/e5_Flat.index --corpus_path data/corpus/wiki-18.jsonl \
    --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu

# hybrid (BM25 + Dense, RRF fusion)
python search_r1/search/hybrid_retrieval.py \
    --bm25_index_path data/index/bm25 --dense_index_path data/index/e5_Flat.index \
    --corpus_path data/corpus/wiki-18.jsonl --topk 10 --fusion_method rrf --rrf_k 60.0 --faiss_gpu
```

**2. Train**

```bash
bash train_grpo.sh   # GRPO, no critic
bash train_ppo.sh    # PPO, with critic
```

**3. Inference** (with the retrieval service from step 1 still running)

```bash
python infer.py   # edit `model_id` to point at your trained checkpoint
```

### Results

Retrieval quality, measured with `scripts/optimization_benchmark.py` (200 queries / 500 docs, `rrf_k=60`):

| Method | Recall@10 | MRR |
|---|---|---|
| BM25 only | 94.75% | 0.9475 |
| Dense only (E5) | 95.25% | 0.9783 |
| **Hybrid (RRF)** | **99.75%** | 0.9723 |

Hybrid retrieval improves Recall@10 by **+5.0pp** over BM25-only and **+4.5pp** over Dense-only on this benchmark. Reproduce with:

```bash
python scripts/optimization_benchmark.py
```

Reward shaping and data augmentation are implemented and unit-verifiable (see `verl/trainer/main_ppo_format.py::RewardManager` and `scripts/data_optimization.py`), but not yet backed by a full RL training run in this repo — no end-to-end training curves are claimed here.

### Project structure

```
search_r1/
├── llm_agent/          # multi-turn Think-Search-Answer loop (generation.py)
└── search/             # retrieval backends: BM25, Dense (FAISS), hybrid RRF fusion, rerank
verl/                   # RL training engine (Ray + FSDP + vLLM), forked from ByteDance's veRL
scripts/                # data download/preprocessing, retrieval benchmarking
train_grpo.sh / train_ppo.sh   # training entry points
```

### Acknowledgement

Built on [veRL](https://github.com/volcengine/verl) ([HybridFlow paper](https://arxiv.org/abs/2409.19256v2)) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1). See [VERL_README.md](VERL_README.md) for the upstream framework's citation and acknowledgements.

### License

[Apache License 2.0](LICENSE)

---

<a id="中文"></a>
## 中文

### 项目简介

Search-R1 用强化学习(PPO/GRPO)训练大语言模型交替进行"推理"与"搜索"。模型每一轮可以发起搜索(`<search>query</search>`)或直接给出答案(`<answer>...</answer>`),检索到的文档以 `<information>...</information>` 形式注入上下文,并在训练时被 mask 掉(不参与 loss 计算),避免模型把检索结果误当作自己生成的内容来学习。

本仓库基于开源 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) / [veRL](https://github.com/volcengine/verl) 二次开发,新增:

- **混合检索(BM25 + Dense,RRF 融合)** —— 实测数据见下方
- **多组件奖励塑形** —— 把原本"仅序列末尾 1 个 EM 分数"的稀疏奖励,拆分为格式 / 检索命中 / 搜索效率等子奖励(总和严格小于答对奖励,保证正确性始终是最高优先级),用于加密训练信号
- **GRPO 优势估计增强** —— 轨迹质量加权 + 组内 z-score 归一化 + 成功轨迹 FIFO 回放缓存
- **训练数据增强** —— 查询扩展、难度分层、意图分类、质量过滤、对比样本构造,作为可选的预处理阶段接入

### 架构

<img src="public/main.png" alt="architecture" width="720"/>

```
Think → <search> query </search> → 检索服务 (BM25 / Dense / RRF) → <information> 文档 </information> → Think → ... → <answer>
```

RL 训练循环由 `RayPPOTrainer` 驱动(Actor/Rollout/Ref/Critic,基于 FSDP + vLLM);多轮 Agent 逻辑在 `search_r1/llm_agent/generation.py`;检索服务以 FastAPI 提供(`POST /retrieve`)。

### 安装

```bash
# 1. 环境
conda create -n search-r1 python=3.10 -y && conda activate search-r1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e .
pip install pyserini faiss-gpu fastapi uvicorn sentence-transformers huggingface_hub

# 2. 数据(语料 + FAISS 索引 + 预处理好的 NQ/HotpotQA,约20GB)
python scripts/download_data.py --save_path ./data
```

硬件建议:8× 显存 ≥24GB 的 GPU(默认 3B/7B 配置推荐 A100/A800 80GB);CUDA 12.1,Python 3.10。

### 使用方法

**1. 启动检索服务**(三选一)

```bash
# Dense 检索 (E5 + FAISS)
python search_r1/search/retrieval_server.py \
    --index_path data/index/e5_Flat.index --corpus_path data/corpus/wiki-18.jsonl \
    --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu

# 混合检索 (BM25 + Dense,RRF 融合)
python search_r1/search/hybrid_retrieval.py \
    --bm25_index_path data/index/bm25 --dense_index_path data/index/e5_Flat.index \
    --corpus_path data/corpus/wiki-18.jsonl --topk 10 --fusion_method rrf --rrf_k 60.0 --faiss_gpu
```

**2. 训练**

```bash
bash train_grpo.sh   # GRPO,无需 Critic
bash train_ppo.sh    # PPO,需要 Critic
```

**3. 推理**(保持第 1 步的检索服务运行中)

```bash
python infer.py   # 把 model_id 改成你训练出的 checkpoint 路径
```

### 实测结果

用 `scripts/optimization_benchmark.py` 实测(200 条 query / 500 篇文档,`rrf_k=60`):

| 方法 | Recall@10 | MRR |
|---|---|---|
| BM25 单路 | 94.75% | 0.9475 |
| Dense 单路 (E5) | 95.25% | 0.9783 |
| **混合 (RRF)** | **99.75%** | 0.9723 |

在此基准上,混合检索 Recall@10 较 BM25 单路提升 **+5.0pp**,较 Dense 单路提升 **+4.5pp**。复现:

```bash
python scripts/optimization_benchmark.py
```

奖励塑形和数据增强的代码均已实现、可独立验证(见 `verl/trainer/main_ppo_format.py::RewardManager` 与 `scripts/data_optimization.py`),但本仓库尚未跑完整的 RL 训练来产出端到端训练曲线,这里不对训练效果做数字宣称。

### 项目结构

```
search_r1/
├── llm_agent/          # 多轮 Think-Search-Answer 循环 (generation.py)
└── search/             # 检索后端:BM25、Dense (FAISS)、混合 RRF 融合、重排序
verl/                   # RL 训练引擎 (Ray + FSDP + vLLM),fork 自字节跳动 veRL
scripts/                # 数据下载/预处理、检索基准测试
train_grpo.sh / train_ppo.sh   # 训练入口脚本
```

### 致谢

基于 [veRL](https://github.com/volcengine/verl)([HybridFlow 论文](https://arxiv.org/abs/2409.19256v2))与 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 二次开发。上游框架的引用与致谢见 [VERL_README.md](VERL_README.md)。

### License

[Apache License 2.0](LICENSE)
