#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urljoin, urlparse
import shutil

VERSION = "2025.2"
BASE_URL = "https://docs.openstack.org"
OUT_ROOT = Path("data/raw/openstack") / VERSION

SERVICES = [
    "nova",
    "neutron",
    "keystone",
    "cinder",
    "glance",
    "horizon",
    "placement",
]

ALLOWED_KEYWORDS = [
    "admin",
    "configuration",
    "operations",
    "troubleshooting",
    "install",
    "architecture",
]

HEADERS = {
    "User-Agent": "docforge-openstack-fetcher/1.0"
}


import shutil

def ensure_pandoc():
    if shutil.which("pandoc") is None:
        raise RuntimeError(
            "pandoc not found. Please install it:\n"
            "  sudo apt install pandoc"
        )



def is_allowed(url: str) -> bool:
    return any(k in url for k in ALLOWED_KEYWORDS)


def html_to_markdown(html_path: Path, md_path: Path):
    subprocess.run(
        [
            "pandoc",
            str(html_path),
            "-o",
            str(md_path),
            "--wrap=none",
        ],
        check=True,
    )


def fetch_service(service: str):
    print(f"\n==> Fetching {service} ({VERSION})")

    service_root = OUT_ROOT / service
    service_root.mkdir(parents=True, exist_ok=True)

    index_url = f"{BASE_URL}/{service}/{VERSION}/"
    r = requests.get(index_url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    links = set()

    for a in soup.select("a[href]"):
        href = urljoin(index_url, a["href"])
        if (
            href.startswith(index_url)
            and is_allowed(href)
            and href.endswith(".html")
        ):
            links.add(href)

    print(f"  Found {len(links)} candidate pages")

    for url in sorted(links):
        rel = url.replace(index_url, "").strip("/")
        out_dir = service_root / Path(rel).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        html_file = out_dir / (Path(rel).stem + ".html")
        md_file = out_dir / (Path(rel).stem + ".md")

        if md_file.exists():
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            html_file.write_text(resp.text, encoding="utf-8")

            html_to_markdown(html_file, md_file)
            html_file.unlink(missing_ok=True)

            print(f"    ✔ {rel}")
            time.sleep(0.2)

        except Exception as e:
            print(f"    ✖ Failed {url}: {e}", file=sys.stderr)


def main():
    ensure_pandoc()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for service in SERVICES:
        fetch_service(service)


if __name__ == "__main__":
    main()
