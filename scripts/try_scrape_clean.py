"""Thử scrape 1 URL bằng Tavily (giống hệt web_search) rồi dọn bằng clean_wiki_markdown.

Dùng cùng endpoint / API key / extract_depth / proxy như tool web_news_search;
mặc định lấy ``format=markdown`` vì clean_wiki_markdown là bộ dọn markdown
(dùng ``--format text`` nếu muốn thử đúng đầu ra text của web_search).

Chạy:
    uv run python scripts/try_scrape_clean.py <URL>
    uv run python scripts/try_scrape_clean.py <URL> --format text
    uv run python scripts/try_scrape_clean.py <URL> --save out.md
"""
import argparse
import os
import sys

import requests

# cho phép import package `app` khi chạy trực tiếp scripts/try_scrape_clean.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import nhẹ — content_cleaner chỉ phụ thuộc `re`, không kéo theo litellm
from app.config import settings
from app.soccer_agent.services.content_cleaner import clean_wiki_markdown, extract_summary

_TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
_proxy = os.environ.get("WEB_SEARCH_PROXY")
_PROXIES = {"http": _proxy, "https": _proxy} if _proxy else None


def tavily_scrape(url: str, fmt: str) -> str:
    """Y hệt web_search._tavily_scrape, chỉ thêm tham số format để thử."""
    key = settings.TAVILY_API_KEYS[0] if settings.TAVILY_API_KEYS else ""
    resp = requests.post(
        _TAVILY_EXTRACT_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"urls": [url], "extract_depth": "advanced", "format": fmt},
        proxies=_PROXIES,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results") or []
    if results:
        return results[0].get("raw_content") or ""
    for f in body.get("failed_results") or []:
        print(f"[FAILED] {f.get('url')}: {f.get('error')}", file=sys.stderr)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--format", default="markdown", choices=["markdown", "text"])
    ap.add_argument("--save", help="ghi kết quả đã dọn ra file")
    ap.add_argument("--full", action="store_true", help="in toàn bộ (thay vì 2000 ký tự đầu)")
    ap.add_argument("--no-proxy", action="store_true", help="bỏ qua WEB_SEARCH_PROXY (khi tunnel tắt)")
    args = ap.parse_args()

    if args.no_proxy:
        global _PROXIES
        _PROXIES = None

    raw = tavily_scrape(args.url, args.format)
    cleaned = clean_wiki_markdown(raw)
    summary = extract_summary(cleaned)

    reduce = f"giảm {100 * (1 - len(cleaned) / len(raw)):.0f}%" if raw else "RAW rỗng"
    print("=" * 72)
    print(f"URL   : {args.url}")
    print(f"format={args.format}  proxy={'on' if _PROXIES else 'off'}")
    print(f"RAW {len(raw)} ký tự  ->  CLEANED {len(cleaned)} ký tự  ({reduce})")
    print("=" * 72)

    print("\n----- CLEANED -----\n")
    print(cleaned if args.full else cleaned[:2000] + ("\n… [cắt bớt]" if len(cleaned) > 2000 else ""))
    print("\n----- SUMMARY (extract_summary) -----\n")
    print(summary)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"\n[saved] {args.save}")


if __name__ == "__main__":
    main()
