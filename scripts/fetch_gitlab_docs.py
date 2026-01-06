#!/usr/bin/env python3
"""
Fetch GitLab docs at a pinned ref (tag/branch/commit) with a sparse checkout of only `doc/`.

Usage:
  python scripts/fetch_gitlab_docs.py --ref v17.0.0
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


GITLAB_REPO_URL = "https://gitlab.com/gitlab-org/gitlab.git"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command with nice error output."""
    print(f"→ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_docs(src_repo_dir: Path, dst_docs_dir: Path) -> None:
    src_docs = src_repo_dir / "doc"
    if not src_docs.exists():
        raise FileNotFoundError(f"Expected {src_docs} to exist, but it does not.")

    dst_docs_dir.parent.mkdir(parents=True, exist_ok=True)
    safe_rmtree(dst_docs_dir)

    # Copy tree (Python 3.8+)
    shutil.copytree(src_docs, dst_docs_dir)
    print(f"Copied docs to: {dst_docs_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Git ref: tag (e.g. v17.0.0), branch, or commit SHA")
    parser.add_argument(
        "--repo-url",
        default=GITLAB_REPO_URL,
        help="GitLab repo URL (default: gitlab-org/gitlab)",
    )
    parser.add_argument(
        "--workdir",
        default="../data/_work/gitlab_repo",
        help="Temporary working directory for the sparse repo",
    )
    parser.add_argument(
        "--outdir",
        default="../data/raw/gitlab",
        help="Output root where docs will be copied",
    )
    args = parser.parse_args()

    ref = args.ref
    repo_url = args.repo_url
    workdir = Path(args.workdir)
    outroot = Path(args.outdir)
    out_docs = outroot / ref / "doc"

    # Clean workdir each run so you get a reproducible state
    safe_rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) Init empty repo
    run(["git", "init"], cwd=workdir)

    # 2) Add remote
    run(["git", "remote", "add", "origin", repo_url], cwd=workdir)

    # 3) Enable sparse checkout and select only doc/
    run(["git", "config", "core.sparseCheckout", "true"], cwd=workdir)
    run(["git", "sparse-checkout", "init", "--cone"], cwd=workdir)
    run(["git", "sparse-checkout", "set", "doc"], cwd=workdir)

    # 4) Fetch only what we need for that ref (depth=1 keeps it light)
    # If ref is a tag/branch name, this works well.
    # If ref is a commit SHA, you may need a deeper fetch.
    try:
        run(["git", "fetch", "--depth", "1", "origin", ref], cwd=workdir)
        run(["git", "checkout", "FETCH_HEAD"], cwd=workdir)
    except subprocess.CalledProcessError:
        print("Fetch failed (maybe a commit SHA). Retrying with a deeper fetch...")
        run(["git", "fetch", "origin", ref], cwd=workdir)
        run(["git", "checkout", "FETCH_HEAD"], cwd=workdir)

    # 5) Copy doc/ out to a clean data location
    copy_docs(workdir, out_docs)

    # 6) Write a small metadata file for reproducibility
    meta_path = outroot / ref / "META.txt"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(workdir)).decode().strip()
    meta_path.write_text(
        f"repo_url={repo_url}\nref={ref}\nresolved_sha={sha}\n",
        encoding="utf-8",
    )
    print(f"Wrote metadata: {meta_path}")
    print(f"Resolved SHA: {sha}")


if __name__ == "__main__":
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    main()
