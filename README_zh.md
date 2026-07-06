<div align="center">
  <img src="public/logo.png" alt="Search-R1 logo" width="200"/>

  # Search-R1(二次开发增强版)

  **通过强化学习训练"推理→搜索→回答"的 Agent,基于 [veRL](https://github.com/volcengine/verl) 构建。**

  [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#安装)
  [![Built on veRL](https://img.shields.io/badge/built%20on-veRL-orange.svg)](VERL_README.md)
  [![PPO%2FGRPO](https://img.shields.io/badge/RL-PPO%20%7C%20GRPO-brightgreen.svg)](#安装)

  <p align="right"><a href="#english">English</a> | <a href="#中文">中文</a></p>
</div>

---

## 项目简介

Search-R1 用强化学习(PPO/GRPO)训练大语言模型交替进行"推理"与"搜索"。模型每一轮可以发起搜索(`<search>query</search>`)或直接给出答案(`<answer>...</answer>`),检索到的文档以 `<information>...</information>` 形式注入上下文,并在训练时通过 `info_mask` 被排除在 loss 计算之外,避免模型把检索结果误当作自己生成的内容来学习。

本仓库基于开源 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) / [veRL](https://github.com/volcengine/verl) 二次开发,扩展了三个方向:

### 1. 混合检索(BM25 + Dense,RRF 融合)

原项目只支持单一检索器(BM25 或 Dense 二选一),存在覆盖盲区:关键词密集的查询对 Dense 不友好,语义/改写类查询对 BM25 不友好。`search_r1/search/hybrid_retrieval.py`(`HybridRetriever`)让两路检索器并行召回(各取 `topk × 2` 候选),再用 **RRF(Reciprocal Rank Fusion)** 融合:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))      (k = 60)
```

选 RRF 而不是直接分数融合,是因为 BM25 和 Dense 的分数量纲不可比(BM25 大约 10-100,余弦相似度大约 0-1)——直接相加会被 BM25 主导,归一化后加权又需要调权重且对离群值敏感;RRF 只依赖排名,天然规避了这个问题。另外还实现了两种备选融合策略:`ScoreWeightedFusion`(已知两路检索器质量有差异时,Min-Max 归一化后加权)和 `ConvexCombinationFusion`(两路质量相近、分数分布可比时)。服务接口与原 `retrieval_server.py` 完全兼容(同样的 `POST /retrieve` 请求/响应格式,已在 `scripts/api_compatibility_verifier.py` 中验证),`generation.py` 无需任何改动即可切换。

### 2. 多组件奖励塑形

原方案的奖励是放在序列**最后一个 token** 上的单一 EM(Exact Match)分数——其余所有 token 都是零奖励,这会让 PPO/GRPO 的信用分配(credit assignment)非常困难(仅约 1% 的 token 携带奖励信号)。`verl/trainer/main_ppo_format.py::RewardManager` 把它拆分为附着在行为发生位置的多个子奖励:**格式奖励**(`<search>`/`<answer>` 标签是否规范)、**检索命中奖励**(搜索是否真的召回了 ground-truth 文档)、**搜索效率奖励**(惩罚冗余/不必要的搜索)。这些辅助奖励的总和被严格限制在低于"答对"奖励之下,保证模型永远不会学到"看起来努力但答错"优先于"直接答对"。

### 3. GRPO 优势估计增强

对每个 prompt 采样的 `n_agent` 条轨迹,在做组内标准优势归一化之前先按结果加权(成功 ×1.5 / 失败 ×0.6 / 部分正确 ×1.0),并用 5σ 裁剪排除离群值。同时维护一个容量 1000 的 FIFO 成功轨迹回放缓存,每个 batch 混入约 10% 的历史成功样本,缓解 on-policy 强化学习容易出现的"遗忘"问题。

### 4. 训练数据增强

`scripts/data_optimization.py` 在原有数据流程之上新增一个可选的预处理阶段:查询扩展(同义词替换 + 句式改写)、难度分层为 easy/medium/hard 三级(基于问题长度、多跳指示词、时间/否定限定词、专有名词密度五个特征打分)、6 类查询意图分类、基于规则的质量过滤,以及对比样本构造(数字 ±1 扰动 + 实体替换,构造难负例/正例对)。输出格式与原 parquet 数据兼容,属于可选启用,不影响原有数据流程。

## 项目结构

```
search_r1/
├── llm_agent/          # 多轮 Think-Search-Answer 循环 (generation.py, tensor_helper.py)
└── search/             # 检索后端:BM25、Dense (FAISS)、混合 RRF 融合、重排序
verl/                   # RL 训练引擎 (Ray + FSDP + vLLM),fork 自字节跳动 veRL
scripts/                # 数据下载/预处理、检索基准测试、数据增强
train_grpo.sh / train_ppo.sh   # 训练入口脚本
```



## 架构

<img src="public/main.png" alt="architecture" width="720"/>

```
Think → <search> query </search> → 检索服务 (BM25 / Dense / RRF) → <information> 文档 </information> → Think → ... → <answer>
```

RL 训练循环由 `RayPPOTrainer` 驱动(Ray 编排的 Actor/Rollout/Ref/Critic,基于 FSDP + vLLM);多轮 Agent 循环(生成 → 解析 `<search>`/`<answer>` → 调用检索器 → 拼接 `<information>` → 重复直到 `<answer>` 或达到 `max_turns`)实现在 `search_r1/llm_agent/generation.py::LLMGenerationManager.run_llm_loop`;检索服务以 FastAPI 提供(`POST /retrieve`)。

## 安装

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

## 使用方法

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

关键训练参数(`train_grpo.sh` / `train_ppo.sh`):

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR` | `data/nq_search` | 训练数据目录 |
| `BASE_MODEL` | `Qwen/Qwen2.5-3B` | 基础模型 |
| `max_turns` | 2 | 每条轨迹最大搜索轮数 |
| `retriever.url` | `http://127.0.0.1:8000/retrieve` | 检索服务地址 |
| `retriever.topk` | 3 | 每次搜索返回文档数 |
| `actor_rollout_ref.rollout.n_agent` | 5(GRPO)/ 1(PPO) | 每个 prompt 采样的轨迹数 |
| `trainer.total_training_steps` | 1005 | 总训练步数 |

**3. 推理**(保持第 1 步的检索服务运行中)

```bash
python infer.py   # 把 model_id 改成你训练出的 checkpoint 路径
```

## 实测结果

用 `scripts/optimization_benchmark.py` 实测(200 条 query / 500 篇文档,`rrf_k=60`):

| 方法 | Hit@1 | Hit@3 | Hit@5 | Recall@10 | MRR | 延迟 |
|---|---|---|---|---|---|---|
| BM25 单路 | 94.0% | 95.0% | 95.5% | 94.75% | 0.9475 | 1.3ms |
| Dense 单路 (E5) | 97.5% | 98.0% | 98.0% | 95.25% | 0.9783 | 1.4ms |
| **混合 (RRF)** | 96.0% | 98.0% | 100.0% | **99.75%** | 0.9723 | 2.6ms |

混合检索 Recall@10 较 BM25 单路提升 **+5.0pp**,较 Dense 单路提升 **+4.5pp**,延迟约为单路检索的 2 倍(两路召回后融合)。同时对 `rrf_k` 做了敏感性扫描(Recall@10):

| `rrf_k` | 10 | 30 | 60 | 100 | 200 |
|---|---|---|---|---|---|
| Recall@10 | 98.79% | 98.20% | 98.42% | 98.04% | 97.66% |

复现:

```bash
python scripts/optimization_benchmark.py
```

奖励塑形和数据增强的代码均已实现、可独立验证(见 `verl/trainer/main_ppo_format.py::RewardManager` 与 `scripts/data_optimization.py`),但本仓库尚未跑完整的 RL 训练来产出端到端训练曲线,这里不对训练效果做数字宣称。


## 致谢

基于 [veRL](https://github.com/volcengine/verl)([HybridFlow 论文](https://arxiv.org/abs/2409.19256v2))与 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 二次开发。上游框架的引用与致谢见 [VERL_README.md](VERL_README.md)。

## License

[Apache License 2.0](LICENSE)
