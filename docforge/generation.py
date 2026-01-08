from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os, re
import requests

@dataclass
class Citation:
    source_url: str
    doc_id: str
    section: str

def call_vllm_chat(
    messages: List[Dict[str, str]],
    model: str,
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> str:
    api_key = api_key or os.getenv("VLLM_API_KEY", "local-token")
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_CITE_RE = re.compile(r"\[\d+\]")

def has_citations(text: str) -> bool:
    return bool(_CITE_RE.search(text))


def build_rag_messages(query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_blocks = []
    for i, p in enumerate(passages, 1):
        context_blocks.append(
            f"[{i}] doc_id: {p.get('doc_id')}\n"
            f"    section: {p.get('section')}\n"
            f"    source_url: {p.get('source_url')}\n"
            f"    text:\n{p.get('text')}\n"
        )
    context = "\n\n".join(context_blocks)

    system = (
        "You are a documentation assistant.\n"
        "You MUST answer using ONLY the provided Sources.\n\n"
        "Strict rules:\n"
        "1) Every paragraph MUST end with citations like [1] or [2][3].\n"
        "2) Do NOT invent YAML/examples. If an example is not explicitly in Sources, do not write it.\n"
        "3) If Sources do not contain enough info, say exactly: \"I don't know from the provided sources.\" "
        "and ask one targeted follow-up question.\n"
        "4) Prefer GitLab sources for GitLab questions.\n"
        "5) Output plain markdown. No preamble.\n"
    )

    user = (
        f"Question: {query}\n\n"
        f"Sources:\n{context}\n\n"
        "Write the answer now, following the rules."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]