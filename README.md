# DocForge — Product Docs RAG + Evaluation Bench (GitLab CI + Ubuntu)

DocForge is a lightweight, end-to-end **retrieval + RAG answering** project built on a **real product documentation corpus** (GitLab) plus an **operating system doc corpus** (Ubuntu). It’s designed to be:

- **Product-oriented**: a usable local “docs assistant” CLI you can run and demo
- **Research-oriented**: a reproducible benchmark setup for evaluating retrieval + reranking + generation

It currently supports:
- **Corpus building** from GitLab Markdown + Ubuntu HTML snapshots
- **FAISS indexing** with SentenceTransformers embeddings
- **Retrieval CLI** (FAISS + optional cross-encoder reranking)
- **Answer CLI** via a locally served LLM (vLLM) with **citation enforcement**

---

## What’s inside

### Data sources
- **GitLab docs** (repo content under `doc/` at a specific ref/tag)
- **Ubuntu docs** (downloaded HTML pages saved with `meta.json` holding canonical URL)

> You control the snapshot/ref so results are reproducible.

### Pipeline
1. Fetch raw docs → `data/raw/...`
2. Build chunk corpus → `data/corpus/chunks.jsonl`
3. Build FAISS index → `data/index/bench/{faiss.index,meta.jsonl}`
4. Retrieve (top-k passages) → CLI output
5. Answer using vLLM + citations → CLI output


---

## Setup

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt```
