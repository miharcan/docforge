# DocForge — Product Docs RAG + Evaluation Bench (GitLab CI + Ubuntu)

DocForge is a lightweight, end-to-end **retrieval + RAG answering** project built on a **real product documentation corpus** (GitLab) plus an **operating system documentation corpus** (Ubuntu).

It is designed to be both:

- **Product-oriented**: a usable local “docs assistant” CLI you can run and demo
- **Research-oriented**: a reproducible benchmark setup for evaluating retrieval, reranking, and generation

---

## What DocForge supports

- **Corpus building** from GitLab Markdown + Ubuntu HTML snapshots
- **FAISS indexing** with SentenceTransformers embeddings
- **Retrieval CLI** (FAISS + optional cross-encoder reranking)
- **Answering CLI** using a locally served LLM (vLLM)
- **Citation enforcement**: answers must reference retrieved sources

---

## Data sources

- **GitLab docs** (repository content under `doc/` at a fixed ref/tag)
- **Ubuntu docs** (downloaded HTML pages, each with `meta.json` containing the canonical URL)

> Snapshots are explicit and versioned so results are reproducible.

---

## Pipeline overview

1. Fetch raw docs → `data/raw/...`
2. Build chunk corpus → `data/corpus/chunks.jsonl`
3. Build FAISS index → `data/index/bench/{faiss.index,meta.jsonl}`
4. Retrieve top-k passages → CLI output
5. Generate grounded answer with citations → CLI output

---

## Smart pipeline (recommended)

Instead of running each fetch/build step manually, use:

```bash
python scripts/run_pipeline.py \
  --gitlab-ref v17.0.0-ee \
  --ubuntu-snapshot 2026-01-06 \
  --ubuntu-urls data/ubuntu_urls.txt
```

This command:
- checks whether raw GitLab docs already exist
- checks whether Ubuntu snapshot already exists
- checks whether corpus/index already exist
- runs only the missing steps

If everything is already present, it skips fetch/build and finishes quickly.

To prepare missing data and run the app in one command:

```bash
python scripts/run_pipeline.py \
  --gitlab-ref v17.0.0-ee \
  --ubuntu-snapshot 2026-01-06 \
  --ubuntu-urls data/ubuntu_urls.txt \
  --run-app \
  --app-mode retrieve \
  --query "How do I use cache in .gitlab-ci.yml?"
```

For a full RAG answer (requires vLLM server running):

```bash
python scripts/run_pipeline.py \
  --gitlab-ref v17.0.0-ee \
  --ubuntu-snapshot 2026-01-06 \
  --ubuntu-urls data/ubuntu_urls.txt \
  --run-app \
  --app-mode answer \
  --query "How do I use cache in .gitlab-ci.yml?" \
  --device cpu \
  --rerank \
  --rerank-device cpu \
  --llm-base-url http://localhost:8000 \
  --llm-model Qwen/Qwen2.5-3B-Instruct
```

---

## Setup

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install .
```
**(Optional) Install ML dependencies**
Required only if you want cross-encoder reranking.
```bash
pip install .[ml]
```



### 2) Fetch GitLab docs
```bash 
python scripts/fetch_gitlab_docs.py --ref v17.0.0-ee 
```

### 3) Fetch Ubuntu docs snapshot
```bash 
python scripts/fetch_ubuntu_docs.py --snapshot 2026-01-06 --urls data/ubuntu_urls.txt 
```

### 4) Build chunk corpus JSONL
```bash 
python scripts/build_corpus.py --gitlab-ref v17.0.0-ee --ubuntu-snapshot 2026-01-06 
```

Output: data/corpus/chunks.jsonl

## Build FAISS index
### 5) Create the vector index
```bash 
python scripts/build_faiss_index.py \
  --corpus data/corpus/chunks.jsonl \
  --out-dir data/index/bench \
  --embed-model sentence-transformers/all-MiniLM-L6-v2
```

Output:
- data/index/bench/faiss.index
- data/index/bench/meta.jsonl

## Retrieval-only CLI (product mode)
The retrieval command prints the top matching passages (with optional reranking).

### 6) Retrieve top docs
```bash 
python -m docforge.cli retrieve "How do I use cache in .gitlab-ci.yml?" --k 5 --rerank --rerank-device cuda 
```

What you’ll see:
- Query
- Ranked hits with score
- Section title + source URL
- Snippet preview

## Answering (RAG) with citations via vLLM
DocForge can also generate answers using a locally hosted LLM (served like an API).

### 7) Install vLLM (one-time)

If `vllm` is not found on your machine, install it in your active DocForge virtual environment:

```bash
source .venv/bin/activate
pip install "vllm>=0.6.0"
```

Verify:

```bash
vllm --help
```

If install fails due to GPU/CUDA compatibility, you can still use retrieval mode (`retrieve`) without vLLM.

### 8) Start a local LLM server (vLLM)
This configuration is stable on single-GPU systems (16GB VRAM) and avoids GPU OOM by explicitly limiting context size and disabling CUDA graph capture.

```bash 
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key local-token
```

Why these flags matter:
```bash
--max-model-len 4096
```
Prevents vLLM from reserving excessive KV cache (critical for GPU stability)

```bash
--max-num-seqs 1
```
Caps concurrent requests to avoid memory pressure

```bash
--gpu-memory-utilization 0.80
```
Leaves headroom to reduce runtime OOM risk.

```bash
--enforce-eager
```

Disables CUDA graphs and torch.compile, trading peak throughput for reliability

This setup is ideal for:
- Local development
- Single-user RAG / agent workflows
- Reproducible demos and benchmarks

### 9) Ask for a cited answer

```bash 
python -m docforge.cli answer "How do I use cache in .gitlab-ci.yml?" \
  --k 2 \
  --rerank \
  --rerank-device cpu \
  --llm-base-url http://localhost:8000 \
  --llm-model Qwen/Qwen2.5-3B-Instruct
```

If you get `Connection refused` for `localhost:8000`, start the vLLM server first.
If you get CUDA OOM during retrieval/reranking, run with `--device cpu` and `--rerank-device cpu`.
If vLLM throws CUDA OOM / returns HTTP 500, restart the vLLM server and retry with lower context (`--k 2`).

DocForge will:
- Retrieve passages
- (Optionally) rerank with a cross-encoder
- Generate an answer grounded in passages
- If citations are missing, it forces a rewrite to add [1] [2] ...
- Prints a “Sources used” panel mapping [i] → URL


## Quick command summary
### Build
```bash 
python scripts/fetch_gitlab_docs.py --ref v17.0.0-ee
python scripts/fetch_ubuntu_docs.py --snapshot 2026-01-06 --urls data/ubuntu_urls.txt
python scripts/build_corpus.py --gitlab-ref v17.0.0-ee --ubuntu-snapshot 2026-01-06
python scripts/build_faiss_index.py --corpus data/corpus/chunks.jsonl --out-dir data/index/bench
```

### Retrieve
```bash 
python -m docforge.cli retrieve "..." --k 5 --rerank --rerank-device cuda
```

### Answer (RAG)
```bash 
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --port 8000 \
  --api-key local-token

python -m docforge.cli answer "..." --k 2 --rerank --rerank-device cpu
```
