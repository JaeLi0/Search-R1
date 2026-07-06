"""
Hybrid Retrieval 使用示例

展示如何在不同场景下使用混合检索系统。

场景1: 独立 API 服务模式 (与原项目兼容)
场景2: 直接调用模式 (Python 脚本内使用)
场景3: 对比不同融合策略的效果
"""

import requests
import json

# ============================================================
# 场景1: API 服务模式 (与 generation.py 兼容)
# ============================================================
# 先启动服务: python search_r1/search/hybrid_retrieval.py
# 然后通过 HTTP 调用，与原 retrieval_server.py 接口兼容

def api_demo():
    """API 调用示例"""
    print("=" * 60)
    print("场景1: API 服务模式")
    print("=" * 60)

    url = "http://127.0.0.1:8000/retrieve"

    payload = {
        "queries": [
            "What is the capital of France?",
            "Who invented the printing press?",
            "What is machine learning?"
        ],
        "topk": 3,
        "return_scores": True
    }

    response = requests.post(url, json=payload)
    results = response.json()

    for i, query_result in enumerate(results['result']):
        print(f"\nQuery {i+1}: {payload['queries'][i]}")
        print("-" * 40)
        for j, item in enumerate(query_result):
            doc = item['document']
            score = item['score']
            title = doc.get('title', 'N/A')
            print(f"  [{j+1}] Score: {score:.4f} | Title: {title}")

    print("\nAPI 端点与原 retrieval_server.py 完全兼容，可直接替换使用")


# ============================================================
# 场景2: 直接调用模式 (不依赖 API)
# ============================================================

def direct_call_demo():
    """直接调用 HybridRetriever 示例"""
    print("\n" + "=" * 60)
    print("场景2: 直接调用模式")
    print("=" * 60)

    from hybrid_retrieval import HybridRetriever, HybridRetrievalConfig

    # 配置（需要实际路径才能运行）
    config = HybridRetrievalConfig(
        bm25_index_path="./index/bm25",
        dense_index_path="./index/e5_Flat.index",
        corpus_path="./data/corpus.jsonl",
        topk=5,
        fusion_method="rrf",
        rrf_k=60.0,
        dense_weight=0.5,
    )

    # 初始化（需要实际索引文件）
    # retriever = HybridRetriever(config)

    # 单查询
    # results, scores = retriever.search("Python tutorial", return_score=True)

    # 批量查询
    # queries = ["What is AI?", "Deep learning basics"]
    # results, scores = retriever.batch_search(queries, return_score=True)

    print("""
    # 配置
    config = HybridRetrievalConfig(
        bm25_index_path="./index/bm25",
        dense_index_path="./index/e5_Flat.index",
        corpus_path="./data/corpus.jsonl",
        topk=5,
        fusion_method="rrf",      # 支持 rrf / weighted / convex
        rrf_k=60.0,              # RRF 平滑参数
        dense_weight=0.5,         # Dense 权重
    )

    # 初始化
    retriever = HybridRetriever(config)

    # 单查询
    results, scores = retriever.search("Python tutorial", return_score=True)

    # 批量查询
    queries = ["What is AI?", "Deep learning basics"]
    results, scores = retriever.batch_search(queries, return_score=True)
    """)


# ============================================================
# 场景3: 融合策略对比
# ============================================================

def fusion_strategy_demo():
    """不同融合策略对比"""
    print("\n" + "=" * 60)
    print("场景3: 融合策略对比")
    print("=" * 60)

    print("""
    Search-R1 原项目支持:
    ├── BM25 (稀疏检索) - 关键词精确匹配
    └── Dense/E5 (向量检索) - 语义相似度匹配

    新增 Hybrid 模块支持:
    ├── RRF (Reciprocal Rank Fusion)
    │   └── 公式: score(d) = Σ 1/(k + rank_i(d))
    │   └── 优点: 对检索器质量不敏感，泛化能力强
    │
    ├── Score-Weighted Fusion
    │   └── 公式: score(d) = Σ w_i * norm(score_i(d))
    │   └── 优点: 可调整权重，适合已知质量差异
    │
    └── Convex Combination
        └── 公式: score(d) = Σ w_i * score_i(d)
        └── 优点: 线性组合，简单直接

    启动不同融合策略:
    # RRF (默认，推荐)
    python hybrid_retrieval.py --fusion_method rrf --rrf_k 60

    # 加权融合 (Dense 权重 0.7)
    python hybrid_retrieval.py --fusion_method weighted --dense_weight 0.7

    # 凸组合
    python hybrid_retrieval.py --fusion_method convex --dense_weight 0.5
    """)


# ============================================================
# 场景4: 评估混合检索效果
# ============================================================

def evaluation_demo():
    """评估工具使用示例"""
    print("\n" + "=" * 60)
    print("场景4: 评估混合检索效果")
    print("=" * 60)

    print("""
    使用评估脚本对比 BM25、Dense、Hybrid 三种检索方式:

    python search_r1/search/eval_hybrid_retrieval.py \\
        --dataset_path ./data/nq-dev.jsonl \\
        --bm25_index_path ./index/bm25 \\
        --dense_index_path ./index/e5_Flat.index \\
        --corpus_path ./data/corpus.jsonl \\
        --topk 10 \\
        --max_samples 1000

    评估指标:
    ├── Hit@K - 前 K 结果中包含正确答案的比例
    ├── MRR - Mean Reciprocal Rank
    └── NDCG - Normalized DCG

    预期结果示例:
    | Method        | Hit@1  | Hit@3  | Hit@5  | Hit@10 | MRR   |
    |---------------|--------|--------|--------|--------|-------|
    | BM25 Only     | 0.32   | 0.48   | 0.55   | 0.62   | 0.41  |
    | Dense (E5)    | 0.38   | 0.54   | 0.61   | 0.68   | 0.47  |
    | Hybrid (RRF)  | 0.41   | 0.58   | 0.65   | 0.72   | 0.51  |

    混合检索相比单一检索的提升:
    └── Hit@3 提升约 5-10%
    """)


if __name__ == "__main__":
    api_demo()
    direct_call_demo()
    fusion_strategy_demo()
    evaluation_demo()

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("""
    快速开始:
    1. 启动混合检索服务:
       bash example/retriever/retrieval_launch_hybrid.sh

    2. 修改 generation.py 的 search_url 指向新服务:
       config.search_url = "http://127.0.0.1:8000/retrieve"

    3. 评估效果:
       python search_r1/search/eval_hybrid_retrieval.py --help
    """)
