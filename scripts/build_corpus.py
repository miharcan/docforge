#!/usr/bin/env python3
"""
Build a unified chunk corpus JSONL from:
- GitLab Markdown docs: data/raw/gitlab/<ref>/doc/**/*.md
- Ubuntu HTML pages:   data/raw/ubuntu/<snapshot>/**/page.html

Output:
- data/corpus/chunks.jsonl

Run:
  python scripts/build_corpus.py --gitlab-ref v17.0.0-ee --ubuntu-snapshot 2026-01-06
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator, Dict, Any, List, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[1]


def stable_id(*parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def slug(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"\s+", "-", t)
    return t[:80] if t else "section"


FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
HTML_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAGS = re.compile(r"<[^>]+>")

def clean_gitlab_markdown(s: str) -> str:
    s = FRONT_MATTER.sub("", s)
    s = HTML_COMMENTS.sub("", s)
    # optional: remove HTML tags (keeps plain text)
    s = HTML_TAGS.sub("", s)
    # normalize
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


MIN_CHARS = 120
MAX_CHARS = 4000

def size_and_split_item(item: Dict[str, Any], min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> List[Dict[str, Any]]:
    """
    - Drop items whose text is too short.
    - Split items whose text is too long into multiple parts.
    Splitting is done on paragraph boundaries (blank lines) first; falls back to hard splits.
    """
    text = (item.get("text") or "").strip()
    if len(text) < min_chars:
        return []

    if len(text) <= max_chars:
        return [item]

    # Split on blank lines (paragraphs)
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    subtexts: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            subtexts.append(buf.strip())
        buf = ""

    for p in parts:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf = buf + "\n\n" + p
        else:
            flush()
            # if a single paragraph is still too big, hard-split it
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    subtexts.append(p[i:i+max_chars].strip())
            else:
                buf = p
    flush()

    out: List[Dict[str, Any]] = []
    base_anchor = item.get("anchor", "top")
    base_section = item.get("section", "")

    for i, t in enumerate(subtexts, 1):
        sub = dict(item)
        sub["text"] = t
        # Make anchor unique + stable across splits
        sub_anchor = f"{base_anchor}-p{i}" if i > 1 else base_anchor
        sub["anchor"] = sub_anchor
        # Make chunk_id unique and deterministic
        sub["chunk_id"] = stable_id(sub["domain"], sub["version"], sub["doc_id"], base_section, sub_anchor)
        out.append(sub)

    out = [o for o in out if len((o.get("text") or "").strip()) >= min_chars]
    return out


# ---------------------------
# GitLab Markdown -> chunks
# ---------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")


def iter_gitlab_markdown_chunks(md_path: Path, domain: str, version: str) -> Iterator[Dict[str, Any]]:
    rel_doc_id = str(md_path).split(f"{version}/doc/", 1)[-1] if f"{version}/doc/" in str(md_path) else str(md_path)

    text = md_path.read_text(encoding="utf-8", errors="replace")
    text = clean_gitlab_markdown(text)
    lines = text.splitlines()


    heading_stack: List[Tuple[int, str]] = []
    buf: List[str] = []
    current_heading = "Document"

    anchor_counts: dict[str, int] = {}

    def flush() -> Iterator[Dict[str, Any]]:
        nonlocal buf, current_heading
        chunk_text = "\n".join(buf).strip()
        if not chunk_text:
            return iter([])

        section_path = " > ".join([h for _, h in heading_stack]) if heading_stack else current_heading

        # anchor = slug(heading_stack[-1][1]) if heading_stack else "top"
        base = slug(heading_stack[-1][1]) if heading_stack else "top"
        anchor_counts[base] = anchor_counts.get(base, 0) + 1
        anchor = base if anchor_counts[base] == 1 else f"{base}-{anchor_counts[base]}"


        chunk_id = stable_id(domain, version, rel_doc_id, section_path, anchor)

        # best-effort source URL (optional)
        source_url = f"https://gitlab.com/gitlab-org/gitlab/-/blob/{version}/doc/{rel_doc_id}"

        item = {
            "chunk_id": chunk_id,
            "domain": domain,
            "version": version,
            "doc_id": f"gitlab:{rel_doc_id}",
            "section": section_path,
            "anchor": anchor,
            "text": chunk_text,
            "source_url": source_url,
        }
        buf = []
        return iter([item])

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            # new heading => flush prior section
            yield from flush()

            level = len(m.group(1))
            title = m.group(2).strip()
            current_heading = title

            # maintain stack by level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buf.append(line)

    yield from flush()


# ---------------------------
# Ubuntu HTML -> chunks
# ---------------------------

def iter_ubuntu_html_chunks(html_path: Path, domain: str, version: str, page_url: str | None) -> Iterator[Dict[str, Any]]:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    title = (soup.title.get_text(" ", strip=True) if soup.title else "Page").strip() or "Page"

    # Grab headings in order; chunk per heading block
    headings = soup.find_all(["h1", "h2", "h3", "h4"])
    if not headings:
        # fallback: whole page text
        body_text = soup.get_text("\n", strip=True)
        if body_text:
            parsed = urlparse(page_url or "")
            doc_id = f"{parsed.netloc}{parsed.path}".strip("/")
            chunk_id = stable_id(domain, version, doc_id, "Page", "top")
            yield {
                "chunk_id": chunk_id,
                "domain": domain,
                "version": version,
                "doc_id": f"ubuntu:{doc_id}",
                "section": title,
                "anchor": "top",
                "text": body_text,
                "source_url": page_url,
            }
        return

    def collect_until_next_heading(start):
        parts = []
        for sib in start.next_siblings:
            if getattr(sib, "name", None) in ["h1", "h2", "h3", "h4"]:
                break
            if getattr(sib, "get_text", None):
                t = sib.get_text(" ", strip=True)
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()

    parsed = urlparse(page_url or "")
    doc_id = f"{parsed.netloc}{parsed.path}".strip("/") or html_path.parent.name
    slug_counts = {}  # per-page


    for h in headings:
        sec_title = h.get_text(" ", strip=True)
        sec_text = collect_until_next_heading(h)
        if not sec_text:
            continue

        # If a section is extremely large (common on wiki pages with few subheadings),
        # pre-split into smaller blocks on blank lines / list/table rows.
        if len(sec_text) > MAX_CHARS:
            blocks = [b.strip() for b in re.split(r"\n\s*\n", sec_text) if b.strip()]
            if len(blocks) == 1:
                # fallback: split on line boundaries that usually separate items
                blocks = [b.strip() for b in re.split(r"\n(?=\* |\n?- )|\n(?=\|)", sec_text) if b.strip()]
        else:
            blocks = [sec_text]

        # Prefer real HTML id if present
        base = h.get("id") or slug(sec_title)
        slug_counts[base] = slug_counts.get(base, 0) + 1
        anchor0 = base if slug_counts[base] == 1 else f"{base}-{slug_counts[base]}"

        for j, block in enumerate(blocks, 1):
            anchor = f"{anchor0}-s{j}" if len(blocks) > 1 else anchor0
            chunk_id = stable_id(domain, version, doc_id, sec_title, anchor)

            yield {
                "chunk_id": chunk_id,
                "domain": domain,
                "version": version,
                "doc_id": f"ubuntu:{doc_id}",
                "section": sec_title,
                "anchor": anchor,
                "text": block,
                "source_url": page_url,
            }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gitlab-ref", required=True)
    ap.add_argument("--ubuntu-snapshot", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "data/corpus/chunks.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # GitLab inputs
    gitlab_root = REPO_ROOT / "data/raw/gitlab" / args.gitlab_ref / "doc"
    if not gitlab_root.exists():
        raise FileNotFoundError(f"GitLab doc root not found: {gitlab_root}")

    # Ubuntu inputs
    ubuntu_root = REPO_ROOT / "data/raw/ubuntu" / args.ubuntu_snapshot
    if not ubuntu_root.exists():
        raise FileNotFoundError(f"Ubuntu snapshot root not found: {ubuntu_root}")
    
    ALLOW_PREFIXES = ("ci/",)   # v1: CI only
    DROP_FILENAMES = {"index.md"}

    
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        # GitLab markdown
        for md in gitlab_root.rglob("*.md"):
            rel = md.relative_to(gitlab_root).as_posix()  # path inside doc/
            if md.name.lower() in DROP_FILENAMES:
                continue
            if not rel.startswith(ALLOW_PREFIXES):
                continue

            for item in iter_gitlab_markdown_chunks(md, domain="gitlab", version=args.gitlab_ref):
                for out_item in size_and_split_item(item):
                    f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    n += 1


        # Ubuntu html (read meta.json to recover canonical URL if present)
        for html in ubuntu_root.rglob("page.html"):
            meta = html.parent / "meta.json"
            page_url = None
            if meta.exists():
                try:
                    page_url = json.loads(meta.read_text(encoding="utf-8")).get("url")
                except Exception:
                    page_url = None

            for item in iter_ubuntu_html_chunks(html, domain="ubuntu", version=args.ubuntu_snapshot, page_url=page_url):
                for out_item in size_and_split_item(item):
                    f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
                    n += 1


    print(f"Wrote {n} chunks to {out_path}")


if __name__ == "__main__":
    main()
