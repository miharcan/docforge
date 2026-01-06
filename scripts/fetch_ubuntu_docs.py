#!/usr/bin/env python3
"""
Fetch a pinned snapshot of Ubuntu documentation pages (HTML) into data/raw/ubuntu/<snapshot>/...

Usage:
  python scripts/fetch_ubuntu_docs.py --snapshot 2026-01-06 --urls data/ubuntu_urls.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTROOT = REPO_ROOT / "data/raw/ubuntu"
DEFAULT_URLS_FILE = REPO_ROOT / "data/ubuntu_urls.txt"


@dataclass
class FetchResult:
    url: str
    status_code: int
    content_type: str | None
    etag: str | None
    last_modified: str | None
    sha256: str
    bytes: int
    fetched_at_utc: str


def slugify_path(path: str) -> str:
    """
    Turn a URL path into a safe file path.
    Example: "/community/SSH/OpenSSH/Configuring" -> "community/SSH/OpenSSH/Configuring"
    """
    path = path.strip("/")
    if not path:
        return "index"
    # Remove suspicious stuff
    path = re.sub(r"[^a-zA-Z0-9/_\-.]", "_", path)
    return path


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def read_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def fetch_one(session: requests.Session, url: str, timeout_s: int = 30, tries: int = 4) -> tuple[bytes, FetchResult]:
    last_err: Exception | None = None
    for i in range(tries):
        try:
            r = session.get(url, timeout=timeout_s, allow_redirects=True)
            content = r.content
            fetched_at = datetime.now(timezone.utc).isoformat()

            result = FetchResult(
                url=r.url,  # final URL after redirects
                status_code=r.status_code,
                content_type=r.headers.get("Content-Type"),
                etag=r.headers.get("ETag"),
                last_modified=r.headers.get("Last-Modified"),
                sha256=sha256_bytes(content),
                bytes=len(content),
                fetched_at_utc=fetched_at,
            )

            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} for {url}")

            return content, result

        except Exception as e:
            last_err = e
            wait = 2 ** i
            print(f"Fetch failed ({i+1}/{tries}) for {url}: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {url} after {tries} tries: {last_err}")


def write_artifacts(outroot: Path, snapshot: str, url: str, content: bytes, meta: FetchResult) -> None:
    parsed = urlparse(meta.url)
    domain = parsed.netloc

    rel_path = slugify_path(parsed.path)
    page_dir = outroot / snapshot / domain / rel_path

    page_dir.mkdir(parents=True, exist_ok=True)

    # Save HTML
    html_path = page_dir / "page.html"
    html_path.write_bytes(content)

    # Save metadata JSON
    meta_path = page_dir / "meta.json"
    meta_path.write_text(json.dumps(meta.__dict__, indent=2), encoding="utf-8")

    print(f"{meta.status_code} {meta.bytes:>7}B  {meta.sha256[:10]}  -> {html_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="Snapshot label (e.g., 2026-01-06)")
    parser.add_argument("--urls", default=str(DEFAULT_URLS_FILE), help="Path to a text file of URLs (one per line)")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTROOT), help="Output root directory")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args()

    outroot = Path(args.outdir)
    urls_file = Path(args.urls)

    if not urls_file.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}")

    urls = read_urls(urls_file)
    if not urls:
        raise ValueError(f"No URLs found in {urls_file}")

    # Session + polite headers
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "DocForgeBenchFetcher/0.1 (+https://github.com/yourname/docforge)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    # Write a snapshot manifest for reproducibility
    manifest_dir = outroot / args.snapshot
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "MANIFEST.json").write_text(
        json.dumps(
            {
                "snapshot": args.snapshot,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "urls_file": str(urls_file),
                "urls": urls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Fetch each URL
    for u in urls:
        content, meta = fetch_one(session, u, timeout_s=args.timeout)
        write_artifacts(outroot, args.snapshot, u, content, meta)

    print(f"\nRaw Ubuntu corpus at: {manifest_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
