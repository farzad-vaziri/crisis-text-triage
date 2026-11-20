#!/usr/bin/env bash
# Full pipeline, in order. About 75 minutes on a 2-vCPU machine without a GPU.
set -e
cd "$(dirname "$0")"
python download.py
python prepare.py
python train_tfidf.py
python train_transformer.py small
python zeroshot_llm.py Qwen/Qwen2.5-0.5B-Instruct
python train_transformer.py large
python evaluate.py
