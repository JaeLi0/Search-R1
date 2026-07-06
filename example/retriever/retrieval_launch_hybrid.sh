#!/bin/bash

# Hybrid Retrieval Fusion 启动脚本
# 使用 RRF (Reciprocal Rank Fusion) 融合 BM25 + Dense 双路召回

# -- Paths: change these to match your setup --
DATA_PATH="./data"

# BM25 索引路径
bm25_index=$DATA_PATH/index/bm25
# Dense 向量索引路径
dense_index=$DATA_PATH/index/e5_Flat.index
# 语料库路径
corpus_file=$DATA_PATH/corpus/wiki-18.jsonl
# 编码器模型路径
encoder_path=intfloat/e5-base-v2

# 启动混合检索服务
# 融合策略可选: rrf (默认), weighted, convex
# RRF_K 参数控制融合平滑度，越大越平滑

python search_r1/search/hybrid_retrieval.py \
    --bm25_index_path $bm25_index \
    --dense_index_path $dense_index \
    --corpus_path $corpus_file \
    --retrieval_model_path $encoder_path \
    --topk 3 \
    --fusion_method rrf \
    --rrf_k 60.0 \
    --dense_weight 0.5 \
    --retrieval_use_fp16 \
    --faiss_gpu \
    --port 8000
