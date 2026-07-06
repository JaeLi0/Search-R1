
# ============================================
# E5 Retrieval Cache Generation Script
# Generates retrieval cache for nq_rag.py
# ============================================

DATA_NAME=nq
TOPK=3

# -- Paths: change these to match your setup --
DATASET_PATH="./data/$DATA_NAME"
INDEX_PATH="./data/index"
CORPUS_PATH="./data/corpus/wiki-18.jsonl"
SAVE_CACHE_PATH="./data/retrieval_cache/${DATA_NAME}/e5_${SPLIT}_cache_top${TOPK}.json"
RETRIEVAL_MODEL="intfloat/e5-base-v2"

SPLIT='test'   # or 'train'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python search_r1/search/retrieval.py \
    --retrieval_method e5 \
    --retrieval_topk $TOPK \
    --index_path $INDEX_PATH \
    --corpus_path $CORPUS_PATH \
    --dataset_path $DATASET_PATH \
    --data_split $SPLIT \
    --retrieval_model_path $RETRIEVAL_MODEL \
    --retrieval_pooling_method mean \
    --retrieval_batch_size 512 \
    --save_cache_path $SAVE_CACHE_PATH
