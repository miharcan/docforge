#!/usr/bin/env python3
"""
Build a FAISS vector index + metadata from a JSONL chunk corpus.

Input JSONL lines must include:
- chunk_id, domain, version, doc_id, section, anchor, text, source_url

Outputs:
- <outdir>/faiss.index
- <outdir>/meta.jsonl

Run:
  python scripts/build_faiss_index.py --corpus data/corpus/chunks_bench.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --outdir data/index/bench --batch 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to chunks JSONL")
    ap.add_argument("--model", required=True, help="Sentence-Transformers model name or local path")
    ap.add_argument("--outdir", required=True, help="Output directory for FAISS + meta")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--device", default=None, help="e.g. 'cpu' or 'cuda' (optional)")
    ap.add_argument("--index", choices=["flatip", "hnsw"], default="flatip", help="FAISS index type")
    ap.add_argument("--hnsw_m", type=int, default=32, help="HNSW M (only if --index hnsw)")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(corpus_path)
    if not items:
        raise ValueError(f"No items found in corpus: {corpus_path}")

    texts = [(it.get("text") or "").strip() for it in items]
    if any(not t for t in texts):
        raise ValueError("Found empty 'text' fields in corpus. Fix corpus before indexing.")

    # Load embedder
    if args.device:
        model = SentenceTransformer(args.model, device=args.device)
    else:
        model = SentenceTransformer(args.model)

    # Encode (normalize so cosine similarity == inner product)
    embs = model.encode(
        texts,
        batch_size=args.batch,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embs = np.asarray(embs, dtype="float32")

    dim = embs.shape[1]
    n = embs.shape[0]

    # Build FAISS index
    if args.index == "flatip":
        index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    else:
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)

    index.add(embs)

    # Save artifacts
    faiss.write_index(index, str(outdir / "faiss.index"))

    with (outdir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            # meta can store everything you need for citations
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Save a small manifest for reproducibility
    manifest = {
        "corpus": str(corpus_path),
        "count": n,
        "dim": dim,
        "model": args.model,
        "index_type": args.index,
    }
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Indexed {n} chunks (dim={dim})")
    print(f"FAISS index: {outdir / 'faiss.index'}")
    print(f"Metadata:    {outdir / 'meta.jsonl'}")
    print(f"Manifest:    {outdir / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
