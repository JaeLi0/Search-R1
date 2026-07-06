"""
Hybrid Retrieval Fusion - 混合检索融合策略

增强 Search-R1 的召回阶段，通过 RRF (Reciprocal Rank Fusion) 算法
融合 BM25 稀疏检索和 Dense 向量检索的结果，提升召回质量。

设计思路：
1. 保持与原有 retrieval_server.py 的接口兼容
2. 新增 HybridRetriever 类，支持双路召回 + RRF 融合
3. 可通过配置选择不同的融合策略：RRF、Score-based Weighted、Convex Combination

论文参考：
- "Reciprocal Rank Fusion for Multiple Retrieval Modalities" (IRNLP, 2023)
- "Sparse, Dense, and Learned Retrieval for RAG" (Google Research, 2024)
"""

import json
import os
import argparse
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import faiss
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm


# ===================== 基础工具函数 =====================

def load_corpus(corpus_path: str):
    """加载语料库"""
    import datasets
    corpus = datasets.load_dataset('json', data_files=corpus_path, split="train", num_proc=4)
    return corpus


def read_jsonl(file_path: str) -> List[dict]:
    """读取 jsonl 文件"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs: List[int]) -> List[dict]:
    """根据索引加载文档"""
    return [corpus[int(idx)] for idx in doc_idxs]


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    """池化操作"""
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError(f"Pooling method {pooling_method} not implemented!")


def load_model(model_path: str, use_fp16: bool = False):
    """加载编码器模型"""
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


# ===================== 配置类 =====================

@dataclass
class HybridRetrievalConfig:
    """混合检索配置"""
    # BM25 配置
    bm25_index_path: str = "./index/bm25"
    # Dense 配置
    dense_index_path: str = "./index/e5_Flat.index"
    # 共享配置
    corpus_path: str = "./data/corpus.jsonl"
    topk: int = 10
    # 融合配置
    fusion_method: str = "rrf"  # "rrf" | "weighted" | "convex"
    rrf_k: float = 60.0  # RRF 算法参数，越大越平滑
    dense_weight: float = 0.5  # Dense 检索权重 (0-1)
    dense_weight: float = 0.5
    # Dense 编码器配置
    retrieval_model_path: str = "intfloat/e5-base-v2"
    retrieval_pooling_method: str = "mean"
    retrieval_query_max_length: int = 256
    retrieval_use_fp16: bool = True
    retrieval_batch_size: int = 128
    faiss_gpu: bool = True


# ===================== Encoder =====================

class Encoder:
    """文本编码器"""
    def __init__(self, model_name: str, model_path: str, pooling_method: str,
                 max_length: int, use_fp16: bool):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model, self.tokenizer = load_model(model_path, use_fp16)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        # E5 模型需要添加前缀
        if "e5" in self.model_name.lower():
            query_list = [f"query: {q}" for q in query_list] if is_query else [f"passage: {q}" for q in query_list]

        # BGE 模型需要添加前缀
        if "bge" in self.model_name.lower() and is_query:
            query_list = [f"Represent this sentence for searching relevant passages: {q}" for q in query_list]

        inputs = self.tokenizer(query_list, max_length=self.max_length, padding=True,
                                truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs['input_ids'].shape[0], 1), dtype=torch.long).to(inputs['input_ids'].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state,
                               inputs['attention_mask'], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        del inputs, output
        torch.cuda.empty_cache()
        return query_emb


# ===================== BM25 Retriever =====================

class BM25Retriever:
    """BM25 稀疏检索器"""
    def __init__(self, config: HybridRetrievalConfig):
        from pyserini.search.lucene import LuceneSearcher
        self.config = config
        self.searcher = LuceneSearcher(config.bm25_index_path)
        self.corpus = load_corpus(config.corpus_path)

    def search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.config.topk

        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            return ([], []) if return_score else []

        scores = [hit.score for hit in hits]
        hits = hits[:num]

        all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
        results = [{
            'title': content.split("\n")[0].strip('"'),
            'text': "\n".join(content.split("\n")[1:]),
            'contents': content,
            'docid': hit.docid
        } for content in all_contents]

        return (results, scores) if return_score else results

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results, scores = [], []
        for query in query_list:
            r, s = self.search(query, num, True)
            results.append(r)
            scores.append(s)
        return (results, scores) if return_score else results


# ===================== Dense Retriever =====================

class DenseRetriever:
    """Dense 向量检索器"""
    def __init__(self, config: HybridRetrievalConfig):
        self.config = config
        self.index = faiss.read_index(config.dense_index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(config.corpus_path)
        self.encoder = Encoder(
            model_name="e5",
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )
        self.batch_size = config.retrieval_batch_size

    def search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.config.topk

        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        results = load_docs(self.corpus, idxs[0])
        for i, r in enumerate(results):
            r['docid'] = int(idxs[0][i])

        return (results, scores[0].tolist()) if return_score else results

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.config.topk

        results, scores = [], []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Dense retrieval"):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            for i, r in enumerate(batch_results):
                r['docid'] = int(flat_idxs[i])

            batch_results = [batch_results[i*num:(i+1)*num] for i in range(len(batch_idxs))]
            scores.extend(batch_scores)
            results.extend(batch_results)

            del batch_emb, batch_scores, batch_idxs
            torch.cuda.empty_cache()

        return (results, scores) if return_score else results


# ===================== 融合策略 =====================

class FusionStrategy:
    """检索结果融合策略基类"""

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        """
        融合多路检索结果
        Args:
            retrieval_results: List of (documents, scores) tuples from different retrievers
        Returns:
            Fused list of (document, fused_score) sorted by score descending
        """
        raise NotImplementedError


class RRFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) - 倒数排序融合

    核心思想：对每个检索器返回的结果，按排名赋分 (1/rank)，最后累加。
    优势：对单个检索器的排序质量不敏感，泛化能力强。

    公式: RRF(d) = Σ 1/(k + rank_i(d))

    参考：F.以北等, "Reciprocal Rank Fusion for Multiple Retrieval Modalities", 2023
    """

    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        doc_scores = defaultdict(float)

        for docs, _ in retrieval_results:
            for rank, doc in enumerate(docs, start=1):
                doc_key = doc.get('docid', doc.get('contents', str(doc)))
                doc_scores[doc_key] += 1.0 / (self.k + rank)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for doc_key, score in sorted_docs:
            doc = next((d for d_list, _ in retrieval_results for d in d_list
                       if d.get('docid') == doc_key or d.get('contents') == doc_key), None)
            if doc:
                result.append((doc, score))

        return result[:kwargs.get('topk', 10)]


class ScoreWeightedFusion(FusionStrategy):
    """
    Score-based Weighted Fusion - 基于分数的加权融合

    核心思想：将不同检索器的分数归一化后加权求和。
    适用于已知各检索器质量差异的场景。

    公式: score(d) = Σ w_i * norm(score_i(d))
    """

    def __init__(self, weights: List[float] = None):
        self.weights = weights or [0.5, 0.5]

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        assert len(self.weights) == len(retrieval_results), "权重数量必须与检索器数量一致"

        doc_scores = defaultdict(float)
        doc_info = {}

        for retriever_idx, (docs, scores) in enumerate(retrieval_results):
            if not scores:
                continue

            # Min-Max 归一化
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s != min_s else 1.0

            for doc, score in zip(docs, scores):
                norm_score = (score - min_s) / range_s
                doc_key = doc.get('docid', doc.get('contents', str(doc)))
                doc_scores[doc_key] += self.weights[retriever_idx] * norm_score
                doc_info[doc_key] = doc

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_info.get(k), s) for k, s in sorted_items[:kwargs.get('topk', 10)]]


class ConvexCombinationFusion(FusionStrategy):
    """
    Convex Combination Fusion - 凸组合融合

    核心思想：在分数空间中做线性插值，假设不同检索器的分数分布相似。
    适用于检索器质量相近、且分数分布可比的场景。
    """

    def __init__(self, weights: List[float] = None):
        self.weights = weights or [0.5, 0.5]

    def fuse(self, retrieval_results: List[Tuple[List[dict], List[float]]], **kwargs) -> List[Tuple[dict, float]]:
        assert len(self.weights) == len(retrieval_results), "权重数量必须与检索器数量一致"

        doc_scores = defaultdict(float)
        doc_info = {}

        for retriever_idx, (docs, scores) in enumerate(retrieval_results):
            if not scores:
                continue

            for doc, score in zip(docs, scores):
                doc_key = doc.get('docid', doc.get('contents', str(doc)))
                doc_scores[doc_key] += self.weights[retriever_idx] * score
                doc_info[doc_key] = doc

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_info.get(k), s) for k, s in sorted_items[:kwargs.get('topk', 10)]]


def get_fusion_strategy(method: str, **kwargs) -> FusionStrategy:
    """获取融合策略实例"""
    strategies = {
        "rrf": RRFusion(k=kwargs.get("rrf_k", 60.0)),
        "weighted": ScoreWeightedFusion(weights=kwargs.get("weights")),
        "convex": ConvexCombinationFusion(weights=kwargs.get("weights")),
    }
    if method not in strategies:
        raise ValueError(f"Unknown fusion method: {method}. Available: {list(strategies.keys())}")
    return strategies[method]


# ===================== Hybrid Retriever =====================

class HybridRetriever:
    """
    混合检索器 - 融合 BM25 和 Dense 双路召回

    相比单一检索器的优势：
    1. BM25 对关键词匹配敏感，擅长实体、术语检索
    2. Dense 对语义相似度敏感，擅长同义词、语义扩展
    3. 融合后可覆盖更多查询类型

    使用方式：
        retriever = HybridRetriever(config)
        results = retriever.search("Python 教程", topk=5)
    """

    def __init__(self, config: HybridRetrievalConfig):
        self.config = config
        self.bm25_retriever = BM25Retriever(config)
        self.dense_retriever = DenseRetriever(config)
        self.fusion_strategy = get_fusion_strategy(
            config.fusion_method,
            rrf_k=config.rrf_k,
            weights=[1 - config.dense_weight, config.dense_weight]
        )

    def search(self, query: str, num: int = None, return_score: bool = False):
        """单查询混合检索"""
        if num is None:
            num = self.config.topk

        # 并行双路召回
        bm25_results, bm25_scores = self.bm25_retriever.search(query, num * 2, True)
        dense_results, dense_scores = self.dense_retriever.search(query, num * 2, True)

        # 融合
        fused = self.fusion_strategy.fuse(
            [(bm25_results, bm25_scores), (dense_results, dense_scores)],
            topk=num
        )

        if return_score:
            return [d for d, _ in fused], [s for _, s in fused]
        else:
            return [d for d, _ in fused]

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        """批量查询混合检索"""
        if num is None:
            num = self.config.topk

        # 并行双路批量召回
        bm25_results, bm25_scores = self.bm25_retriever.batch_search(query_list, num * 2, True)
        dense_results, dense_scores = self.dense_retriever.batch_search(query_list, num * 2, True)

        # 逐条融合
        fused_results, fused_scores = [], []
        for i in range(len(query_list)):
            fused = self.fusion_strategy.fuse(
                [(bm25_results[i], bm25_scores[i]), (dense_results[i], dense_scores[i])],
                topk=num
            )
            fused_results.append([d for d, _ in fused])
            fused_scores.append([s for _, s in fused])

        return (fused_results, fused_scores) if return_score else fused_results


# ===================== FastAPI 服务 =====================

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI(title="Hybrid Retrieval Fusion API")


def _passages2string(retrieval_result: List[dict]) -> str:
    """将检索结果格式化为字符串"""
    format_reference = ''
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item.get('contents', '')
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
    return format_reference


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    混合检索接口，融合 BM25 和 Dense 双路召回

    请求格式：
    {
        "queries": ["query1", "query2"],
        "topk": 3,
        "return_scores": true
    }

    返回格式：
    {
        "result": [
            [{"document": {...}, "score": 0.95}, ...],
            [...]
        ]
    }
    """
    if not request.topk:
        request.topk = config.topk

    results, scores = hybrid_retriever.batch_search(
        request.queries,
        num=request.topk,
        return_score=True
    )

    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = [{"document": doc, "score": scores[i][j]}
                        for j, doc in enumerate(single_result)]
            resp.append(combined)
        else:
            resp.append(single_result)

    return {"result": resp}


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok", "fusion_method": config.fusion_method}


@app.get("/info")
def retrieval_info():
    """获取检索器配置信息"""
    return {
        "fusion_method": config.fusion_method,
        "rrf_k": config.rrf_k,
        "topk": config.topk,
        "bm25_index": config.bm25_index_path,
        "dense_index": config.dense_index_path,
    }


# ===================== 入口 =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Fusion Server")
    # 路径配置
    parser.add_argument("--bm25_index_path", type=str, default="./index/bm25")
    parser.add_argument("--dense_index_path", type=str, default="./index/e5_Flat.index")
    parser.add_argument("--corpus_path", type=str, default="./data/corpus.jsonl")
    # 检索配置
    parser.add_argument("--topk", type=int, default=10, help="返回的 topk 结果数")
    # 融合配置
    parser.add_argument("--fusion_method", type=str, default="rrf",
                       choices=["rrf", "weighted", "convex"],
                       help="融合策略: rrf(倒数排序融合) / weighted(加权融合) / convex(凸组合融合)")
    parser.add_argument("--rrf_k", type=float, default=60.0,
                       help="RRF 算法参数，建议范围 30-100，越大越平滑")
    parser.add_argument("--dense_weight", type=float, default=0.5,
                       help="Dense 检索权重 (0-1)，BM25 权重自动为 1-dense_weight")
    # Dense 模型配置
    parser.add_argument("--retrieval_model_path", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--retrieval_pooling_method", type=str, default="mean")
    parser.add_argument("--retrieval_query_max_length", type=int, default=256)
    parser.add_argument("--retrieval_use_fp16", action='store_true', default=True)
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--faiss_gpu", action='store_true', default=True)
    # 服务配置
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    args = parser.parse_args()

    config = HybridRetrievalConfig(
        bm25_index_path=args.bm25_index_path,
        dense_index_path=args.dense_index_path,
        corpus_path=args.corpus_path,
        topk=args.topk,
        fusion_method=args.fusion_method,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        retrieval_model_path=args.retrieval_model_path,
        retrieval_pooling_method=args.retrieval_pooling_method,
        retrieval_query_max_length=args.retrieval_query_max_length,
        retrieval_use_fp16=args.retrieval_use_fp16,
        retrieval_batch_size=args.retrieval_batch_size,
        faiss_gpu=args.faiss_gpu,
    )

    hybrid_retriever = HybridRetriever(config)

    print(f"=" * 60)
    print(f"Hybrid Retrieval Fusion Server")
    print(f"=" * 60)
    print(f"Fusion Method: {config.fusion_method}")
    print(f"RRF K: {config.rrf_k}")
    print(f"Dense Weight: {config.dense_weight}")
    print(f"TopK: {config.topk}")
    print(f"=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
