# SemFuse: LLM-Guided Semantic Edge Refinement for Graph Fusion

## 📰 News

- 🚩 Updates (2026-04-07) Initial open-source README with full end-to-end pipeline.

## 🤔 What is SemFuse?

**SemFuse** is a semantic graph enhancement pipeline for node classification.

It uses an LLM to:
- score existing edges and remove low-relevance connections,
- mine potential new edges from cosine-similar node pairs,
- re-score candidate edges and add high-confidence edges back.

Finally, SemFuse trains a **multi-view GNN** that fuses original and refined graphs (addition fusion or attention fusion) for robust performance.

![SemFuse](images/SemFuse.png)

## 👋🏻 Method Overview

The workflow has two stages:

1. **LLM-based edge refinement**
- `construct_edge_json.py`: build prompts from original edges.
- `llm_infer.py`: run batched LLM inference.
- `clean_json.py`: parse LLM JSON outputs.
- `del.py`: drop low-score edges to get a filtered graph.
- `get_similar.py`: retrieve top-k non-edge cosine-similar node pairs.
- `llm_infer.py` (second pass): score candidate edges.
- `merge_json.py`: merge candidate metadata and LLM scores.
- `add.py`: add high-score candidate edges to build final graph.

2. **Graph model training**
- `model/gnns_fuse_add.py` (or `gnns_fuse_attention.py`) trains a two-view GNN on original + refined graphs.

## 🏃‍♂️‍➡️ Quick Start

### 📁 1) Data preparation

Put dataset `.npz` files into `dataset/`.

Expected keys in each `.npz`:
- `node_texts`
- `node_features`
- `node_labels`
- `edges` (shape `[2, E]`)
- `train_masks`, `val_masks`, `test_masks`

Example layout:

```text
SemFuse
├── dataset
│   ├── Cornell.npz
│   ├── Wisconsin.npz
│   ├── Texas.npz
│   ├── Actor.npz
│   └── Amazon.npz
└── src
```

### 🧩 2) Environment setup

```bash
conda create -n semfuse python=3.10 -y
conda activate semfuse
pip install torch numpy scikit-learn tqdm vllm peft
```

For LLM fine-tuning, install dependencies in `src/llm/llama_factory`:

```bash
cd src/llm/llama_factory
pip install -e .
cd ../../..
```

### 🔑 3) Configure model paths

Edit these files before running:

- `src/llm/train.sh`
  - set `MODEL_PATH`
  - set `OUTPUT_DIR`
- `src/script/llm_infer.py`
  - set `base_model_path`
  - set `lora_path`
  - set `input_path` / `output_path` for each phase

### 🐹 4) Run the SemFuse pipeline

From repo root (`SemFuse/`) execute:

```bash
# Step 1: LoRA SFT training
cd src/llm
bash train.sh
cd ../..

# Step 2: Build prompts from original edges
cd src/script
python construct_edge_json.py

# Step 3: LLM inference on original edges
python llm_infer.py

# Step 4: Clean LLM JSON outputs
python clean_json.py

# Step 5: Delete low-score edges -> *_filtered.npz
python del.py

# Step 6: Build cosine-similarity candidate edge prompts
python get_similar.py

# Step 7: LLM inference on candidate edges (edit llm_infer.py input/output first)
python llm_infer.py

# Step 8: lean LLM JSON outputs
python clean_json.py

# Step 9: Merge candidate metadata and LLM outputs
python merge_json.py

# Step 10: Add high-score candidate edges -> *_final.npz
python add.py
cd ../..
```

Then train fusion GNN:

```bash
python src/model/gnns_fuse_add.py \
  --data-orig dataset/Cornell.npz \
  --data-enhanced dataset/Cornell_final.npz \
  --model sage \
  --epochs 400 \
  --device cuda
```

## ⚠️ Notes

- In your message, `cpmstrict_edge_json.py` appears to be a typo; the repo file is `construct_edge_json.py`.
- Your sequence has duplicate step number `6`; the second one should be step `7`.
- Several scripts currently contain hard-coded dataset names/paths (e.g., `Amazon`, `Cornell`). Please edit those constants for your target dataset before running.

## 🙏 Acknowledgement

- LLM training/inference stack is built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- Thanks to open-source PyTorch and graph-learning communities.
