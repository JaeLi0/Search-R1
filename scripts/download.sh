#!/bin/bash
# Download E5 FAISS index and Wikipedia corpus
# Usage: bash scripts/download.sh

save_path="./data"

python scripts/download.py --save_path $save_path

# Merge FAISS index parts
cat $save_path/part_aa $save_path/part_ab > $save_path/index/e5_Flat.index
