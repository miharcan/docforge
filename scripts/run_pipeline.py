#!/usr/bin/env python3
"""
Smart DocForge pipeline runner.

Behavior:
- If raw GitLab docs are missing, fetch them.
- If raw Ubuntu snapshot is missing, fetch it.
- If corpus is missing, build it.
- If FAISS index is missing, build it.
- If requested, run DocForge app command (retrieve/answer).

This avoids expensive refetch/rebuild work when artifacts already exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def has_gitlab_raw(ref: str) -> bool:
    root = REPO_ROOT / "data" / "raw" / "gitlab" / ref
    doc_root = root / "doc"
    return root.exists() and (root / "META.txt").exists() and doc_root.exists() and any(doc_root.rglob("*.md"))


def has_ubuntu_raw(snapshot: str) -> bool:
    root = REPO_ROOT / "data" / "raw" / "ubuntu" / snapshot
    return root.exists() and (root / "MANIFEST.json").exists() and any(root.rglob("page.html"))


def has_corpus(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def has_index(index_dir: Path) -> bool:
    return (index_dir / "faiss.index").exists() and (index_dir / "meta.jsonl").exists()


def maybe_fetch_gitlab(ref: str, force: bool) -> None:
    if not force and has_gitlab_raw(ref):
        print(f"[SKIP] GitLab raw docs already present for ref={ref}")
        return

    run(
        [
            sys.executable,
            "scripts/fetch_gitlab_docs.py",
            "--ref",
            ref,
            "--workdir",
            "data/_work/gitlab_repo",
            "--outdir",
            "data/raw/gitlab",
        ]
    )


def maybe_fetch_ubuntu(snapshot: str, urls: Path, force: bool) -> None:
    if not force and has_ubuntu_raw(snapshot):
        print(f"[SKIP] Ubuntu raw snapshot already present for snapshot={snapshot}")
        return

    run(
        [
            sys.executable,
            "scripts/fetch_ubuntu_docs.py",
            "--snapshot",
            snapshot,
            "--urls",
            str(urls),
            "--outdir",
            "data/raw/ubuntu",
        ]
    )


def maybe_build_corpus(gitlab_ref: str, ubuntu_snapshot: str, out_corpus: Path, force: bool) -> None:
    if not force and has_corpus(out_corpus):
        print(f"[SKIP] Corpus already present: {out_corpus}")
        return

    run(
        [
            sys.executable,
            "scripts/build_corpus.py",
            "--gitlab-ref",
            gitlab_ref,
            "--ubuntu-snapshot",
            ubuntu_snapshot,
            "--out",
            str(out_corpus),
        ]
    )


def maybe_build_index(corpus: Path, index_dir: Path, embed_model: str, force: bool) -> None:
    if not force and has_index(index_dir):
        print(f"[SKIP] FAISS index already present: {index_dir}")
        return

    run(
        [
            sys.executable,
            "scripts/build_faiss_index.py",
            "--corpus",
            str(corpus),
            "--model",
            embed_model,
            "--outdir",
            str(index_dir),
        ]
    )


def run_app_command(
    mode: str,
    query: str,
    k: int,
    index_dir: Path,
    device: str | None,
    rerank: bool,
    rerank_device: str | None,
    llm_base_url: str,
    llm_model: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "docforge.cli",
        mode,
        query,
        "--k",
        str(k),
        "--index-dir",
        str(index_dir),
    ]

    if device:
        cmd.extend(["--device", device])

    if rerank:
        cmd.append("--rerank")
        if rerank_device:
            cmd.extend(["--rerank-device", rerank_device])

    if mode == "answer":
        cmd.extend(["--llm-base-url", llm_base_url, "--llm-model", llm_model])

    try:
        run(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "App command failed. Artifacts are ready, but runtime dependencies "
            "(for example embedding model availability/network cache, reranker, or LLM server) "
            "may be missing."
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare DocForge artifacts if missing, then optionally run the app.")
    ap.add_argument("--gitlab-ref", default="v17.0.0-ee")
    ap.add_argument("--ubuntu-snapshot", default="2026-01-06")
    ap.add_argument("--ubuntu-urls", default="data/ubuntu_urls.txt")

    ap.add_argument("--corpus", default="data/corpus/chunks.jsonl")
    ap.add_argument("--index-dir", default="data/index/bench")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")

    ap.add_argument("--force-fetch-gitlab", action="store_true")
    ap.add_argument("--force-fetch-ubuntu", action="store_true")
    ap.add_argument("--force-corpus", action="store_true")
    ap.add_argument("--force-index", action="store_true")

    ap.add_argument("--run-app", action="store_true", help="Run DocForge app command after preparing artifacts")
    ap.add_argument("--app-mode", choices=["retrieve", "answer"], default="retrieve")
    ap.add_argument("--query", default="How do I use cache in .gitlab-ci.yml?")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None, help="Embedding device for app run: e.g. cpu or cuda")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-device", default=None)
    ap.add_argument("--llm-base-url", default="http://localhost:8000")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    ubuntu_urls = Path(args.ubuntu_urls)
    corpus = Path(args.corpus)
    index_dir = Path(args.index_dir)

    maybe_fetch_gitlab(args.gitlab_ref, force=args.force_fetch_gitlab)
    maybe_fetch_ubuntu(args.ubuntu_snapshot, urls=ubuntu_urls, force=args.force_fetch_ubuntu)
    maybe_build_corpus(args.gitlab_ref, args.ubuntu_snapshot, out_corpus=corpus, force=args.force_corpus)
    maybe_build_index(corpus=corpus, index_dir=index_dir, embed_model=args.embed_model, force=args.force_index)

    print("\n[OK] Pipeline artifacts are ready.")
    if args.run_app:
        try:
            run_app_command(
                mode=args.app_mode,
                query=args.query,
                k=args.k,
                index_dir=index_dir,
                device=args.device,
                rerank=args.rerank,
                rerank_device=args.rerank_device,
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}")
            raise SystemExit(1)
    else:
        print("[INFO] App run skipped. Use --run-app to execute retrieve/answer.")


if __name__ == "__main__":
    main()
