#!/bin/bash
# BM25 retrieval server

DATA_PATH="./data"
index_file=$DATA_PATH/index/bm25
corpus_file=$DATA_PATH/corpus/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name
