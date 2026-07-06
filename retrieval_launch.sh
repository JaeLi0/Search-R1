#!/bin/bash
# ============================================
# Launch the E5 retrieval server (FastAPI)
# ============================================

# -- Paths: change these to match your setup --
INDEX_PATH="./data/index/e5_Flat.index"
CORPUS_PATH="./data/corpus/wiki-18.jsonl"
RETRIEVER_NAME="e5"
RETRIEVER_MODEL="intfloat/e5-base-v2"

python search_r1/search/retrieval_server.py \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --topk 3 \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu
