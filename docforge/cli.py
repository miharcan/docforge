from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .retrieval import Retriever
from .generation import has_citations

app = typer.Typer(add_completion=False)
console = Console()


def _default_index_dir() -> Path:
    return Path("data/index/bench")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="User question"),
    k: int = typer.Option(5, help="How many results to show"),
    fetch_k: int = typer.Option(40, help="How many candidates to fetch from FAISS before trimming"),
    index_dir: Path = typer.Option(_default_index_dir(), help="Directory with faiss.index and meta.jsonl"),
    embed_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Embedding model"),
    device: Optional[str] = typer.Option(None, help="Embedding device: e.g. 'cpu' or 'cuda'"),
    boosts: bool = typer.Option(True, help="Apply simple path boosts (product mode)"),
    rerank: bool = typer.Option(False, help="Rerank with a cross-encoder"),
    rerank_model: str = typer.Option("cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder reranker"),
    rerank_device: Optional[str] = typer.Option(None, help="Reranker device: e.g. 'cpu' or 'cuda'"),
) -> None:
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise typer.BadParameter(f"Missing faiss.index/meta.jsonl in {index_dir}")

    r = Retriever(index_path=index_path, meta_path=meta_path, embed_model=embed_model, device=device)

    candidates = r.search(query=query, k=k if not rerank else fetch_k, fetch_k=fetch_k, apply_boosts=boosts)

    if rerank:
        candidates = r.rerank(
            query=query,
            candidates=candidates,
            rerank_model=rerank_model,
            device=rerank_device,
            top_k=k,
        )
    else:
        candidates = candidates[:k]

    console.print(Panel.fit(Text(query, style="bold"), title="Query"))

    for i, hit in enumerate(candidates, 1):
        m = hit.meta
        title = f"#{i}  score={hit.score:.4f}  [{m.get('domain')}] {m.get('doc_id')}"
        section = m.get("section", "")
        url = m.get("source_url", "")

        snippet = (hit.text or "").strip().replace("\n", " ")
        if len(snippet) > 450:
            snippet = snippet[:450] + "…"

        body = f"[bold]Section:[/bold] {section}\n[bold]Source:[/bold] {url}\n\n{snippet}"
        console.print(Panel(body, title=title, expand=False))


@app.command()
def answer(
    query: str = typer.Argument(..., help="User question"),
    k: int = typer.Option(5, help="How many final sources to use"),
    fetch_k: int = typer.Option(40, help="How many candidates to fetch from FAISS before trimming"),
    index_dir: Path = typer.Option(_default_index_dir(), help="Directory with faiss.index and meta.jsonl"),
    embed_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Embedding model"),
    device: Optional[str] = typer.Option(None, help="Embedding device: e.g. 'cpu' or 'cuda'"),
    boosts: bool = typer.Option(True, help="Apply simple path boosts (product mode)"),
    rerank: bool = typer.Option(True, help="Rerank with a cross-encoder before generation"),
    rerank_model: str = typer.Option("cross-encoder/ms-marco-MiniLM-L-6-v2", help="Cross-encoder reranker"),
    rerank_device: Optional[str] = typer.Option(None, help="Reranker device: e.g. 'cpu' or 'cuda'"),
    llm_model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", help="vLLM served model name"),
    llm_base_url: str = typer.Option("http://localhost:8000", help="vLLM base URL"),
) -> None:
    from .generation import call_vllm_chat, build_rag_messages

    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"
    if not index_path.exists() or not meta_path.exists():
        raise typer.BadParameter(f"Missing faiss.index/meta.jsonl in {index_dir}")

    r = Retriever(index_path=index_path, meta_path=meta_path, embed_model=embed_model, device=device)

    candidates = r.search(query=query, k=k if not rerank else fetch_k, fetch_k=fetch_k, apply_boosts=boosts)
    if rerank:
        candidates = r.rerank(query=query, candidates=candidates, rerank_model=rerank_model, device=rerank_device, top_k=k)
    else:
        candidates = candidates[:k]

    # Build passages for the prompt
    passages = []
    for hit in candidates:
        m = hit.meta
        passages.append(
            {
                "doc_id": m.get("doc_id"),
                "section": m.get("section"),
                "source_url": m.get("source_url"),
                "text": hit.text,
            }
        )

    msgs = build_rag_messages(query=query, passages=passages)
    answer_text = call_vllm_chat(messages=msgs, model=llm_model, base_url=llm_base_url)

    

    if not has_citations(answer_text):
        # Force a rewrite with citations, without changing the meaning.
        msgs2 = msgs + [{
            "role": "user",
            "content": (
                "You forgot citations. Rewrite your answer with inline citations "
                "at the end of EVERY paragraph using [1], [2], etc. "
                "Do NOT add any new facts. If you cannot cite something, remove it."
            )
        }]
        answer_text = call_vllm_chat(messages=msgs2, model=llm_model, base_url=llm_base_url)

    
    console.print(Panel.fit(Text(query, style="bold"), title="Query"))
    console.print(Panel(answer_text, title="Answer", expand=False))
    # Print sources mapping so you can visually verify citations
    src_lines = []
    for i, p in enumerate(passages, 1):
        src_lines.append(f"[{i}] {p.get('doc_id')} — {p.get('section')}\n    {p.get('source_url')}")
    console.print(Panel("\n".join(src_lines), title="Sources used", expand=False))


if __name__ == "__main__":
    app()