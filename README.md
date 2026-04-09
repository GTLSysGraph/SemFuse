# SemFuse: Aligning LLM Semantics with Graph Topology for Heterophilic Learning

## 💡 What is SemFuse?

**SemFuse** is a semantic graph enhancement pipeline for **heterophilic graph learning** and node classification.

It uses an LLM to:
- score existing edges and remove low-relevance connections,
- mine potential new edges from cosine-similar node pairs,
- re-score candidate edges and add high-confidence edges back.

Finally, SemFuse trains a **multi-view GNN** that fuses original and refined graphs (addition fusion or attention fusion) for robust performance.

![SemFuse](images/SemFuse.png)

## 🧠 Method Overview

The workflow has two stages:

1. **LLM-based edge refinement**
- `construct_edge_json.py`: build prompts from original edges.
- `llm_infer.py`: run batched LLM inference.
- `clean_json.py`: parse LLM JSON outputs.
- `del.py`: drop low-score edges to get a filtered graph.
- `get_similar.py`: retrieve top-k non-edge cosine-similar node pairs.
- `merge_json.py`: merge candidate metadata and LLM scores.
- `add.py`: add high-score candidate edges to build final graph.

2. **Graph model training**
- `model/gnns_fuse_add.py` (or `gnns_fuse_attention.py`) trains a two-view GNN on original + refined graphs.

Main project structure:

```text
SemFuse
├── README.md
├── requirements.txt
├── dataset/
├── images/
└── src
    ├── llm
    │   ├── train.sh
    │   └── llama_factory/
    ├── model
    │   ├── gnns_fuse_add.py
    │   ├── gnns_fuse_attention.py
    │   └── gnns_original.py
    └── script
        ├── construct_edge_json.py
        ├── llm_infer.py
        ├── clean_json.py
        ├── del.py
        ├── get_similar.py
        ├── merge_json.py
        └── add.py
```

## ⚙️ Quick Start

### 📁 1) Data preparation

Pull dataset files with Git LFS:

```bash
git lfs install
git lfs pull
```

Then ensure dataset `.npz` files are in `dataset/`.

Expected keys in each `.npz`:
- `node_texts`
- `node_features`
- `node_labels`
- `edges` (shape `[2, E]`)
- `train_masks`, `val_masks`, `test_masks`

### 🧭 2) Environment setup

```bash
conda create -n semfuse python=3.10 -y
conda activate semfuse
pip install -r requirements.txt
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

### 🚀 4) Run the SemFuse pipeline

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

# Step 8: Clean LLM JSON outputs
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


## 🙏 Acknowledgement

- LLM training/inference stack is built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- Our datasets are from [LLM4HeG](https://github.com/honey0219/LLM4HeG).
- Our prompts and method design reference [LLM4RGNN](https://github.com/zhongjian-zhang/LLM4RGNN).
