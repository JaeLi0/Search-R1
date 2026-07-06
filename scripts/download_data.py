#!/usr/bin/env python3
"""
Search-R1 数据下载脚本
=======================

下载 Search-R1 项目所需的全部数据：
  1. Wikipedia 语料库 (wiki-18.jsonl) - 检索源
  2. FAISS Dense 索引 (e5_Flat.index) - 向量检索
  3. QA 训练/测试数据集 (NQ, HotpotQA, TriviaQA, PopQA, etc.)
  4. 预处理合并数据集 (nq_hotpotqa_train)

用法:
  python scripts/download_data.py --save_path ./data

参数:
  --save_path     数据保存目录 (默认: ./data)
  --download_corpus    下载语料库和索引 (默认: True)
  --download_dataset   下载 QA 数据集 (默认: True)
  --download_merged    下载预处理合并数据 (默认: True)
  --skip_corpus        跳过语料库下载 (文件较大 ~5GB)
  --skip_dataset       跳过数据集下载
  --skip_merged        跳过预处理数据下载
"""

import argparse
import os
import sys
import subprocess
import gzip
import shutil
from pathlib import Path


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: str, description: str = ""):
    """安全执行 shell 命令"""
    label = f" [{description}]" if description else ""
    print(f"  $ {cmd}{label}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [!] 失败: {result.stderr.strip()}")
        return False
    return True


# ============================================================
# 1. Wikipedia 语料库 + FAISS 索引
# ============================================================

def download_corpus_and_index(save_path: str):
    """
    下载 Wikipedia 语料库和 E5 Dense 索引

    来源:
      - https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus
      - https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index

    文件:
      - wiki-18.jsonl (解压后 ~13GB)
      - e5_Flat.index (FAISS 向量索引 ~4GB)
    """
    print("\n" + "=" * 60)
    print("  1. 下载 Wikipedia 语料库和 FAISS 索引")
    print("=" * 60)

    index_dir = os.path.join(save_path, "index")
    corpus_dir = os.path.join(save_path, "corpus")
    ensure_dir(index_dir)
    ensure_dir(corpus_dir)

    # 1a. 下载 E5 索引分片
    print("\n  [1a] 下载 E5 FAISS 索引...")
    index_file = os.path.join(index_dir, "e5_Flat.index")
    part_aa = os.path.join(index_dir, "part_aa")
    part_ab = os.path.join(index_dir, "part_ab")

    if os.path.exists(index_file):
        print(f"    [skip] 索引已存在: {index_file}")
    elif os.path.exists(part_aa) and os.path.exists(part_ab):
        print("    [合并] 分片已下载，正在合并...")
        with open(index_file, 'wb') as out_f:
            for part in [part_aa, part_ab]:
                with open(part, 'rb') as in_f:
                    shutil.copyfileobj(in_f, out_f)
        print(f"    [完成] 索引合并到: {index_file}")
    else:
        print(f"    从 HuggingFace 下载索引分片...")
        print(f"    仓库: PeterJinGo/wiki-18-e5-index")
        from huggingface_hub import hf_hub_download
        for part in ["part_aa", "part_ab"]:
            hf_hub_download(
                repo_id="PeterJinGo/wiki-18-e5-index",
                filename=part,
                repo_type="dataset",
                local_dir=index_dir,
            )
        # 合并
        print("    合并分片...")
        with open(index_file, 'wb') as out_f:
            for part in [part_aa, part_ab]:
                with open(os.path.join(index_dir, part), 'rb') as in_f:
                    shutil.copyfileobj(in_f, out_f)
        print(f"    索引已保存: {index_file}")

    # 1b. 下载并解压语料库
    print("\n  [1b] 下载 Wikipedia 语料库...")
    corpus_file = os.path.join(corpus_dir, "wiki-18.jsonl")
    corpus_gz = os.path.join(corpus_dir, "wiki-18.jsonl.gz")

    if os.path.exists(corpus_file):
        print(f"    [skip] 语料库已存在: {corpus_file}")
    elif os.path.exists(corpus_gz):
        print(f"    [解压] gz 文件已下载，正在解压...")
        with gzip.open(corpus_gz, 'rb') as f_in:
            with open(corpus_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"    [完成] 语料库: {corpus_file}")
    else:
        print(f"    从 HuggingFace 下载语料库...")
        print(f"    仓库: PeterJinGo/wiki-18-corpus")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="PeterJinGo/wiki-18-corpus",
            filename="wiki-18.jsonl.gz",
            repo_type="dataset",
            local_dir=corpus_dir,
        )
        print(f"    解压中...")
        with gzip.open(corpus_gz, 'rb') as f_in:
            with open(corpus_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"    语料库已保存: {corpus_file}")

    # 统计信息
    if os.path.exists(corpus_file):
        size_gb = os.path.getsize(corpus_file) / (1024**3)
        print(f"\n    语料库大小: {size_gb:.1f} GB")
    if os.path.exists(index_file):
        size_gb = os.path.getsize(index_file) / (1024**3)
        print(f"    索引大小: {size_gb:.1f} GB")

    return {
        "index_path": index_file if os.path.exists(index_file) else None,
        "corpus_path": corpus_file if os.path.exists(corpus_file) else None,
    }


# ============================================================
# 2. QA 数据集 (从 FlashRAG)
# ============================================================

def download_qa_datasets(save_path: str):
    """
    下载 QA 数据集

    来源: RUC-NLPIR/FlashRAG_datasets on HuggingFace

    包含:
      - NQ (Natural Questions)
      - HotpotQA
      - TriviaQA
      - PopQA
      - 2WikiMultihopQA
      - MuSiQue
      - Bamboogle
    """
    print("\n" + "=" * 60)
    print("  2. 下载 QA 数据集")
    print("=" * 60)

    dataset_dir = os.path.join(save_path, "datasets")
    ensure_dir(dataset_dir)

    dataset_names = [
        ("nq", "Natural Questions"),
        ("hotpotqa", "HotpotQA (多跳推理)"),
        ("triviaqa", "TriviaQA"),
        ("popqa", "PopQA"),
        ("2wikimultihopqa", "2WikiMultihopQA (多跳)"),
        ("musique", "MuSiQue (多跳组合)"),
        ("bamboogle", "Bamboogle"),
    ]

    results = {}
    for name, desc in dataset_names:
        print(f"\n  [{name}] {desc}")
        try:
            from datasets import load_dataset
            dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", name)
            splits = list(dataset.keys())
            for split in splits:
                num = len(dataset[split])
                print(f"    {split}: {num} samples")
            results[name] = {"splits": splits, "loaded": True}
        except Exception as e:
            print(f"    [警告] 加载失败: {e}")
            results[name] = {"splits": [], "loaded": False}

    return results


# ============================================================
# 3. 预处理合并数据集
# ============================================================

def download_merged_dataset(save_path: str):
    """
    下载预处理合并数据集

    来源: https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train

    包含:
      - nq_hotpotqa_train/train.parquet
      - nq_hotpotqa_train/test.parquet
    """
    print("\n" + "=" * 60)
    print("  3. 下载预处理合并数据集 (NQ + HotpotQA)")
    print("=" * 60)

    merged_dir = os.path.join(save_path, "nq_hotpotqa_train")
    ensure_dir(merged_dir)

    train_file = os.path.join(merged_dir, "train.parquet")
    test_file = os.path.join(merged_dir, "test.parquet")

    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"  [skip] 数据已存在: {merged_dir}")
        return merged_dir

    print(f"  从 HuggingFace 下载...")
    print(f"  仓库: PeterJinGo/nq_hotpotqa_train")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="PeterJinGo/nq_hotpotqa_train",
            repo_type="dataset",
            local_dir=merged_dir,
        )
        print(f"  数据已保存: {merged_dir}")
        return merged_dir
    except Exception as e:
        print(f"  [警告] 下载失败: {e}")
        print(f"  备选方案: 运行数据预处理脚本生成")
        print(f"    bash scripts/nq_hotpotqa/data_process.sh")
        return None


# ============================================================
# 4. 下载状态总览
# ============================================================

def print_summary(save_path: str, results: dict):
    """打印下载状态总览"""
    print("\n" + "=" * 60)
    print("  数据下载状态总览")
    print("=" * 60)

    checks = [
        ("FAISS 索引", os.path.join(save_path, "index", "e5_Flat.index")),
        ("Wikipedia 语料库", os.path.join(save_path, "corpus", "wiki-18.jsonl")),
        ("NQ+HotpotQA 训练集", os.path.join(save_path, "nq_hotpotqa_train", "train.parquet")),
        ("NQ+HotpotQA 测试集", os.path.join(save_path, "nq_hotpotqa_train", "test.parquet")),
    ]

    for name, path in checks:
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        size = ""
        if exists:
            size_gb = os.path.getsize(path) / (1024**3)
            size = f" ({size_gb:.1f} GB)"
        print(f"  {status} {name}{size}")

    # 磁盘使用
    total_size = 0
    for _, path in checks:
        if os.path.exists(path):
            total_size += os.path.getsize(path)
    print(f"\n  总计磁盘使用: {total_size/(1024**3):.1f} GB")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Search-R1 数据下载脚本")
    parser.add_argument("--save_path", type=str, default="./data",
                        help="数据保存根目录 (默认: ./data)")
    parser.add_argument("--skip_corpus", action="store_true",
                        help="跳过语料库和索引下载")
    parser.add_argument("--skip_dataset", action="store_true",
                        help="跳过 QA 数据集下载")
    parser.add_argument("--skip_merged", action="store_true",
                        help="跳过预处理合并数据下载")
    args = parser.parse_args()

    print("=" * 60)
    print("  Search-R1 数据下载")
    print("=" * 60)
    print(f"  保存路径: {args.save_path}")
    print(f"  预计总大小: ~20GB (语料 ~13GB, 索引 ~4GB, 数据集 ~3GB)")
    print()

    ensure_dir(args.save_path)

    # 1. 语料库和索引
    if not args.skip_corpus:
        download_corpus_and_index(args.save_path)

    # 2. QA 数据集
    if not args.skip_dataset:
        download_qa_datasets(args.save_path)

    # 3. 预处理合并数据
    if not args.skip_merged:
        download_merged_dataset(args.save_path)

    # 4. 状态总览
    print_summary(args.save_path, {})

    # 5. 下一步提示
    print("""
  下一步:
    1. 预处理数据集 (生成 search-format parquet):
       bash scripts/nq_hotpotqa/data_process.sh

    2. 启动检索服务:
       bash retrieval_launch.sh

    3. 开始训练:
       bash train_grpo.sh
    """)


if __name__ == "__main__":
    main()
