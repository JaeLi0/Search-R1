# Search-R1 数据准备

## 概览

训练需要三类数据：

| 数据 | 来源 | 产物 |
|------|------|------|
| QA 问答对 | `RUC-NLPIR/FlashRAG_datasets` (HuggingFace) | `train.parquet` / `test.parquet` |
| Wikipedia 语料库 | `PeterJinGo/wiki-18-corpus` (HuggingFace) | `wiki-18.jsonl` (~13GB) |
| FAISS 向量索引 | `PeterJinGo/wiki-18-e5-index` (HuggingFace) | `e5_Flat.index` (~4GB) |

`train.parquet` 和 `test.parquet` 的每一条样本包含：问题文本、prompt 模板（含 `<think>/<search>/<answer>` 指令）、golden answer（用于 rule-based reward 计算）。

训练脚本（`train_grpo.sh` / `train_ppo.sh`）直接读取 `.parquet` 文件。

---

## 方式一：一键下载（推荐）

```bash
python scripts/download_data.py --save_path ./data
```

这个脚本会自动下载以上三类数据。跳过不需要的部分：

```bash
python scripts/download_data.py --save_path ./data --skip_corpus   # 只下 QA 数据
python scripts/download_data.py --save_path ./data --skip_dataset  # 只下语料库和索引
```

### 说明：预处理好的 `.parquet` 从哪里来？

`download_data.py` 的第三步会从 HuggingFace 下载 **已经预处理好的** `train.parquet` 和 `test.parquet`：

```
https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
```

这个仓库是项目作者用下面的 **方式二** 脚本处理完后上传的，跟你自己跑一遍 `data_process.sh` 得到的结果完全一致。如果网络没问题，直接用这个就行，省去自己跑预处理的时间。

---

## 方式二：从原始数据自己生成 `.parquet`

如果你想了解每一条样本是怎么构建的，或者需要修改 prompt 模板、增减数据集，可以自己跑预处理。

### 1. 下载原始 QA 数据

原始数据来自 HuggingFace 上的 `RUC-NLPIR/FlashRAG_datasets`，包含 NQ、HotpotQA、TriviaQA、PopQA、2WikiMultihopQA、MuSiQue、Bamboogle 共 7 个数据集。

不需要手动下载，预处理脚本运行时会通过 `datasets.load_dataset()` 自动拉取。

### 2. 生成 train.parquet

```bash
python scripts/data_process/qa_search_train_merge.py \
  --local_dir ./data/nq_hotpotqa_train \
  --data_sources nq,hotpotqa
```

这一步会：
1. 从 `FlashRAG_datasets` 加载 `nq` 和 `hotpotqa` 两个数据集的 `train` 切分
2. 把每个问题包装成 Search-R1 的 prompt 模板（包含 `<think>`、`<search>`、`<answer>` 指令结构）
3. 附加 `golden_answers` 作为 ground truth（用于训练时做 rule-based reward）
4. 合并后输出为 `train.parquet`

### 3. 生成 test.parquet

```bash
python scripts/data_process/qa_search_test_merge.py \
  --local_dir ./data/nq_hotpotqa_train \
  --data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
```

逻辑同上，但覆盖 7 个数据集，使用各自数据集的 `test`/`dev` 切分，输出为 `test.parquet`。

### 4. 一键执行

以上步骤已经写在 `data_process.sh` 里，直接运行即可：

```bash
WORK_DIR=/your/work/dir bash scripts/nq_hotpotqa/data_process.sh
```

---

## 完整的数据流

```
┌─────────────────────────────────────────────────────────────────┐
│  语料库 & 索引（供检索服务使用）                                   │
│                                                                 │
│  HuggingFace                         本地产物                    │
│  ──────────                          ────────                    │
│  PeterJinGo/wiki-18-e5-index  ──►  e5_Flat.index                │
│  PeterJinGo/wiki-18-corpus    ──►  wiki-18.jsonl                 │
│                                                                 │
│  下载方式：download_data.py 或 scripts/download.py               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  QA 数据集（供训练/评估读取）                                      │
│                                                                 │
│  HuggingFace                         本地产物                    │
│  ──────────                          ────────                    │
│  RUC-NLPIR/FlashRAG_datasets  ──►  train.parquet                │
│    (nq, hotpotqa, ...)          ──►  test.parquet                │
│                                                                 │
│  生成方式：                                                      │
│    - qa_search_train_merge.py（train）                           │
│    - qa_search_test_merge.py（test）                             │
│                                                                 │
│  或直接下载预处理好的：                                           │
│    - PeterJinGo/nq_hotpotqa_train（已包含 train/test.parquet）   │
└─────────────────────────────────────────────────────────────────┘
```

训练脚本的 `data.train_files` 和 `data.val_files` 参数直接指向这些 `.parquet` 路径。

---

## 下载语料库和索引（不含 QA 数据）

如果你只需要语料库和检索索引：

```bash
python scripts/download.py --save_path ./data
cat ./data/part_* > ./data/e5_Flat.index
gzip -d ./data/wiki-18.jsonl.gz
```

---

## 启动检索服务

数据准备好、需要在启动训练**之前**把检索服务先跑起来：

```bash
conda activate retriever
bash retrieval_launch.sh
```

检索服务是独立进程，训练时通过 HTTP 调用。如果检索服务挂了或没启动，训练不会报错，但 observation 会全是空字符串——训练前最好 `curl` 一下确认服务存活。

---

## 启动训练

```bash
# PPO
bash scripts/nq_hotpotqa/v0.2/train_ppo.sh

# GRPO
bash scripts/nq_hotpotqa/v0.2/train_grpo.sh
```

---

## 启动评估

```bash
bash scripts/nq_hotpotqa/evaluate.sh
```

可以通过修改脚本中的 `$BASE_MODEL` 变量来指定要评估的模型路径。
