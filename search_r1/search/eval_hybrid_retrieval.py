"""
Hybrid Retrieval Evaluation - 混合检索召回评估

评估 BM25、Dense、以及混合检索在不同查询上的表现。
用于验证混合检索策略的优化效果。

指标：
1. Hit@K - 前 K 个结果中是否包含正确答案
2. MRR - Mean Reciprocal Rank
3. NDCG - Normalized Discounted Cumulative Gain

用法：
    python search_r1/search/eval_hybrid_retrieval.py \
        --dataset_path ./data/nq-dev.jsonl \
        --bm25_index ./index/bm25 \
        --dense_index ./index/e5_Flat.index \
        --corpus_path ./data/corpus.jsonl
"""

import argparse
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from hybrid_retrieval import (
    HybridRetriever, BM25Retriever, DenseRetriever,
    HybridRetrievalConfig, get_fusion_strategy
)


@dataclass
class RetrievalResult:
    """单条检索结果"""
    query: str
    bm25_results: List[dict]
    dense_results: List[dict]
    hybrid_results: List[dict]
    bm25_scores: List[float]
    dense_scores: List[float]
    hybrid_scores: List[float]
    ground_truth: List[str]


@dataclass
class EvalMetrics:
    """评估指标"""
    method: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    latency_ms: float


def load_dataset(dataset_path: str) -> List[dict]:
    """加载评估数据集"""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_hit_at_k(retrieved_docs: List[dict], ground_truth: List[str], k: int) -> float:
    """计算 Hit@K"""
    if not ground_truth:
        return 0.0

    top_k = retrieved_docs[:k]
    for doc in top_k:
        doc_title = doc.get('title', '').lower()
        doc_text = doc.get('text', '').lower()
        for gt in ground_truth:
            if gt.lower() in doc_title or gt.lower() in doc_text:
                return 1.0
    return 0.0


def compute_mrr(retrieved_docs: List[dict], ground_truth: List[str]) -> float:
    """计算 MRR (Mean Reciprocal Rank)"""
    if not ground_truth:
        return 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_title = doc.get('title', '').lower()
        doc_text = doc.get('text', '').lower()
        for gt in ground_truth:
            if gt.lower() in doc_title or gt.lower() in doc_text:
                return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved_docs: List[dict], ground_truth: List[str], k: int = 10) -> float:
    """计算 NDCG@K"""
    if not ground_truth:
        return 0.0

    dcg = 0.0
    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        doc_title = doc.get('title', '').lower()
        doc_text = doc.get('text', '').lower()
        relevance = 0
        for gt in ground_truth:
            if gt.lower() in doc_title or gt.lower() in doc_text:
                relevance = 1
                break
        dcg += relevance / np.log2(rank + 1)

    # IDCG: 理想情况下的 DCG（所有相关文档都在前排）
    num_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retriever(
    retriever,
    queries: List[str],
    ground_truths: List[List[str]],
    topk: int = 10,
    description: str = ""
) -> EvalMetrics:
    """评估单个检索器"""
    hits_at_1, hits_at_3, hits_at_5, hits_at_10 = 0, 0, 0, 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    total = len(queries)

    start_time = time.time()

    results = retriever.batch_search(queries, num=topk, return_score=True)[0] if hasattr(retriever, 'batch_search') else []
    results_with_scores = retriever.batch_search(queries, num=topk, return_score=True)

    for i, (query, gt) in enumerate(tqdm(zip(queries, ground_truths),
                                          desc=description, total=total)):
        docs = results[i] if results else []
        scores = results_with_scores[1][i] if results_with_scores else []

        hits_at_1 += compute_hit_at_k(docs, gt, 1)
        hits_at_3 += compute_hit_at_k(docs, gt, 3)
        hits_at_5 += compute_hit_at_k(docs, gt, 5)
        hits_at_10 += compute_hit_at_k(docs, gt, 10)
        mrr_sum += compute_mrr(docs, gt)
        ndcg_sum += compute_ndcg(docs, gt, topk)

    elapsed = (time.time() - start_time) * 1000 / total  # ms per query

    return EvalMetrics(
        method=description,
        hit_at_1=hits_at_1 / total,
        hit_at_3=hits_at_3 / total,
        hit_at_5=hits_at_5 / total,
        hit_at_10=hits_at_10 / total,
        mrr=mrr_sum / total,
        latency_ms=elapsed
    )


def print_results(results: List[EvalMetrics]):
    """打印评估结果"""
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'Hit@1':<8} {'Hit@3':<8} {'Hit@5':<8} {'Hit@10':<8} {'MRR':<8} {'Latency(ms)':<12}")
    print("=" * 80)

    for r in results:
        print(f"{r.method:<20} {r.hit_at_1:.4f}    {r.hit_at_3:.4f}    {r.hit_at_5:.4f}    "
              f"{r.hit_at_10:.4f}    {r.mrr:.4f}    {r.latency_ms:.2f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Retrieval")
    # 数据配置
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="评估数据集路径 (jsonl 格式，需包含 question 和 golden_answers 字段)")
    parser.add_argument("--corpus_path", type=str, required=True,
                       help="语料库路径")
    # 索引路径
    parser.add_argument("--bm25_index_path", type=str, required=True,
                       help="BM25 索引路径")
    parser.add_argument("--dense_index_path", type=str, required=True,
                       help="Dense 索引路径")
    # 模型配置
    parser.add_argument("--retrieval_model_path", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--retrieval_pooling_method", type=str, default="mean")
    parser.add_argument("--retrieval_query_max_length", type=int, default=256)
    parser.add_argument("--retrieval_use_fp16", action='store_true', default=True)
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--faiss_gpu", action='store_true', default=True)
    # 评估配置
    parser.add_argument("--topk", type=int, default=10, help="评估的 topk")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="最多评估的样本数 (用于快速验证)")

    args = parser.parse_args()

    # 加载数据
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path)[:args.max_samples]
    queries = [d['question'] for d in dataset]
    ground_truths = [d.get('golden_answers', d.get('answer', [''])) for d in dataset]
    if isinstance(ground_truths[0], str):
        ground_truths = [[gt] for gt in ground_truths]

    print(f"Loaded {len(queries)} queries for evaluation")

    # 构建配置
    config = HybridRetrievalConfig(
        bm25_index_path=args.bm25_index_path,
        dense_index_path=args.dense_index_path,
        corpus_path=args.corpus_path,
        topk=args.topk,
        fusion_method="rrf",
        rrf_k=60.0,
        dense_weight=0.5,
        retrieval_model_path=args.retrieval_model_path,
        retrieval_pooling_method=args.retrieval_pooling_method,
        retrieval_query_max_length=args.retrieval_query_max_length,
        retrieval_use_fp16=args.retrieval_use_fp16,
        retrieval_batch_size=args.retrieval_batch_size,
        faiss_gpu=args.faiss_gpu,
    )

    # 初始化检索器
    print("\nInitializing retrievers...")
    bm25_retriever = BM25Retriever(config)
    dense_retriever = DenseRetriever(config)
    hybrid_retriever = HybridRetriever(config)

    # 评估
    results = []

    print("\nEvaluating BM25...")
    results.append(evaluate_retriever(
        bm25_retriever, queries, ground_truths, args.topk,
        description="BM25 Only"
    ))

    print("\nEvaluating Dense (E5)...")
    results.append(evaluate_retriever(
        dense_retriever, queries, ground_truths, args.topk,
        description="Dense (E5)"
    ))

    print("\nEvaluating Hybrid (RRF)...")
    results.append(evaluate_retriever(
        hybrid_retriever, queries, ground_truths, args.topk,
        description="Hybrid (RRF)"
    ))

    # 打印结果
    print_results(results)

    # 保存结果
    output_path = "hybrid_retrieval_eval_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "topk": args.topk,
                "max_samples": len(queries),
                "rrf_k": config.rrf_k,
                "dense_weight": config.dense_weight,
            },
            "results": [
                {
                    "method": r.method,
                    "hit_at_1": r.hit_at_1,
                    "hit_at_3": r.hit_at_3,
                    "hit_at_5": r.hit_at_5,
                    "hit_at_10": r.hit_at_10,
                    "mrr": r.mrr,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ]
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
