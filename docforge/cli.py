from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .retrieval import Retriever

app = typer.Typer(add_completion=False)
console = Console()


def _default_index_dir() -> Path:
    return Path("data/index/bench")


@app.command()
def main(
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


if __name__ == "__main__":
    app()