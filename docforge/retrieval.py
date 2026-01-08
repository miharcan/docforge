from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class Retrieved:
    score: float
    meta: Dict[str, Any]
    text: str


def load_meta(meta_path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def _path_heuristic_boost(meta: Dict[str, Any]) -> float:
    """
    Product-oriented boost so canonical docs rank above migration pages.
    Keep it small and report ablations later.
    """
    doc_id = meta.get("doc_id", "")  # e.g. "gitlab:ci/migration/github_actions.md"
    if not doc_id.startswith("gitlab:ci/"):
        return 0.0

    rel = doc_id[len("gitlab:"):]  # "ci/..."
    boost = 0.0

    # prefer canonical reference & core docs
    if rel.startswith("ci/caching/"):
        boost += 0.15
    if rel.startswith("ci/yaml/"):
        boost += 0.12
    if rel.startswith("ci/variables/"):
        boost += 0.10

    # downweight migration guides for general “how do I …” queries
    if rel.startswith("ci/migration/"):
        boost -= 0.10

    return boost

def _content_boost(meta: Dict[str, Any]) -> float:
    t = (meta.get("text") or "").lower()
    s = (meta.get("section") or "").lower()
    boost = 0.0
    if "gitlab-ci.yml" in t or "gitlab-ci.yml" in s:
        boost += 0.05
    if "define the cache" in t or "cache reference" in t:
        boost += 0.08
    if "clear the cache" in t:
        boost -= 0.05
    return boost


class Retriever:
    def __init__(
        self,
        index_path: Path,
        meta_path: Path,
        embed_model: str,
        device: Optional[str] = None,
    ) -> None:
        self.index = load_index(index_path)
        self.meta = load_meta(meta_path)
        self.embedder = SentenceTransformer(embed_model, device=device) if device else SentenceTransformer(embed_model)

        # sanity: index size should match meta
        if self.index.ntotal != len(self.meta):
            raise ValueError(f"Index has {self.index.ntotal} vectors but meta has {len(self.meta)} rows")

    def search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 30,
        apply_boosts: bool = True,
    ) -> List[Retrieved]:
        qv = self.embedder.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")

        D, I = self.index.search(qv, fetch_k)
        results: List[Retrieved] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            m = self.meta[idx]
            s = float(score)
            if apply_boosts:
                s += _path_heuristic_boost(m)
                s += _content_boost(m)
            results.append(Retrieved(score=s, meta=m, text=m.get("text", "")))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def rerank(
        self,
        query: str,
        candidates: List[Retrieved],
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Retrieved]:
        """
        Cross-encoder reranking. Call this after FAISS fetch.
        """
        if not candidates:
            return candidates

        ce = CrossEncoder(rerank_model, device=device) if device else CrossEncoder(rerank_model)
        pairs = [(query, c.text) for c in candidates]
        scores = ce.predict(pairs)

        reranked = []
        for c, s in zip(candidates, scores):
            reranked.append(Retrieved(score=float(s), meta=c.meta, text=c.text))

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k] if top_k else reranked
    
    

