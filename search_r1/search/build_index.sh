#!/bin/bash
# ============================================
# Build E5 / BM25 retrieval index
# ============================================

# -- Paths: change these to match your setup --
CORPUS_PATH="./data/corpus/wiki-18.jsonl"     # jsonl corpus file
SAVE_DIR="./data/index"                       # where to save the index
RETRIEVER_NAME="e5"                           # "e5" for dense, "bm25" for BM25
RETRIEVER_MODEL="intfloat/e5-base-v2"

# faiss_type: Flat=exact, HNSW32/64/128=approximate (faster, more memory)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python search_r1/search/index_builder.py \
    --retrieval_method $RETRIEVER_NAME \
    --model_path $RETRIEVER_MODEL \
    --corpus_path $CORPUS_PATH \
    --save_dir $SAVE_DIR \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
