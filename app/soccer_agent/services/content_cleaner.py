"""
Markdown cleaning helpers for Wikipedia-style content extracted via Tavily.

The cleaning pipeline mirrors the reference implementation iterated in
``kinh_dev.ipynb``: strip Wikipedia chrome (sidebar / toolbar noise, citation
references, edit markers, naked URLs), trim "See also" / "References" /
"External links" tail sections, and run the strip+line-clean pair twice so a
second pass can remove leftovers exposed after the first.

``extract_summary`` returns the intro paragraph block (text right after the
top-level ``# Title`` heading and before the first ``##`` sub-section / table
row) — that's what we persist as the ``SUMMARY`` field on each entity.
"""
from __future__ import annotations

import re
from typing import List

# Section headings that should be cut, along with everything below them.
_META_SECTIONS: frozenset[str] = frozenset({
    "see also",
    "notes",
    "references",
    "external links",
    "further reading",
    "citations",
    "bibliography",
    "footnotes",
})


# ── Public API ────────────────────────────────────────────────────────────────


def clean_wiki_markdown(text: str) -> str:
    """Strip Wikipedia chrome from a markdown blob extracted by Tavily.

    Two-pass: ``strip_links_and_refs`` → ``clean_lines`` → ``strip_links_and_refs``
    → ``clean_lines``. The repeat pass catches noise that only becomes visible
    after the first round of stripping (e.g. orphan text fragments left
    behind when a link wrapper is removed).
    """
    if not text:
        return ""

    text = _strip_links_and_refs(text)
    text = _clean_lines(text, is_first_pass=True)
    text = _strip_links_and_refs(text)
    text = _clean_lines(text, is_first_pass=False)
    return text


def extract_summary(md: str, max_chars: int = 1500) -> str:
    """Return the intro paragraph block of a cleaned Wikipedia markdown doc.

    Heuristic: starts after the top-level ``# Title`` heading (if any),
    stops at the first ``##`` heading, the first markdown table row, or
    after ``max_chars`` of accumulated text. Skips short orphan label
    lines that Wikipedia sometimes emits between the title and the lead.
    """
    if not md:
        return ""

    lines = md.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            start = i + 1
            break

    collected: List[str] = []
    total = 0
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("|"):
            break
        if not stripped:
            if collected:
                break
            continue
        if not collected and _looks_like_orphan_label(stripped):
            continue
        collected.append(stripped)
        total += len(stripped) + 1
        if total >= max_chars:
            break

    return " ".join(collected).strip()


def detect_sections(text: str) -> list[tuple[int, str, int]]:
    """Debug helper: list every ``#``-``###`` heading as ``(level, title, line)``."""
    sections: list[tuple[int, str, int]] = []
    for i, line in enumerate(text.splitlines()):
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            title = re.sub(r"\[.*?\]\(.*?\)", "", m.group(2)).strip()
            sections.append((level, title, i))
    return sections


# ── Internals ─────────────────────────────────────────────────────────────────


def _find_content_end_line(text: str) -> int:
    """Return the index of the first heading whose title is in ``_META_SECTIONS``."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^#{1,2}\s+(.*)", line)
        if m:
            title = re.sub(r"\[.*?\]\(.*?\)", "", m.group(1)).strip().lower()
            if title in _META_SECTIONS:
                return i
    return len(lines)


def _find_content_start_line(lines: list[str]) -> int:
    """Skip up to and including the ``# Title - Wikipedia`` chrome heading.

    Returns the line index of the next ``#`` heading (the actual article
    title) — or 0 when no Wikipedia chrome heading is present.
    """
    saw_wiki_title = False
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(.*)", line)
        if m:
            title = re.sub(r"\[.*?\]\(.*?\)", "", m.group(1)).strip()
            if title.lower().endswith("- wikipedia"):
                saw_wiki_title = True
            elif saw_wiki_title:
                return i
    return 0


def _strip_links_and_refs(text: str) -> str:
    """Strip markdown links, references, citations, and HTML chrome."""
    # Image links: ![alt](url) → drop entirely.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Anchor links: [text](#anchor) → keep text.
    text = re.sub(r"\[([^\]]*)\]\(#[^)]*\)", r"\1", text)
    # General markdown links: [text](url) → keep text.
    text = re.sub(r"\[([^\]]*)\]\([^)\n|]*\)", r"\1", text)
    # Dangling links: [text](url   (no closing paren)
    text = re.sub(r"\[[^\]]*\]\([^)\n|]*", "", text)
    # Empty parentheses left behind after stripping.
    text = re.sub(r"\(\s*\)", "", text)
    # Naked URLs.
    text = re.sub(r"(?:https?:)?//\S+", "", text)
    # Footnote references: [1], [note 1], [a].
    text = re.sub(r"\[\d+\]|\[note \d+\]|\[[a-z]\]", "", text)
    # Trailing extension fragments: ".png)", ".jpg)".
    text = re.sub(r"\.\w{2,4}\)", "", text)
    # Quoted spans before ): "foo").
    text = re.sub(r'"[^"]*"\)', "", text)
    # HTML tags.
    text = re.sub(r"<[^>]+>", "", text)
    # Edit markers.
    text = re.sub(r"\[edit\]", "", text)
    # Wikipedia bold-caret reference markers: **^**.
    text = re.sub(r"\*\*\^\*\*", "", text)
    return text


def _clean_lines(text: str, is_first_pass: bool) -> str:
    """Drop chrome, sidebar/toolbar noise, decorative lines, and short bullets."""
    lines = text.splitlines()

    # Trim everything from the first meta section onward.
    end = _find_content_end_line(text)
    lines = lines[:end]

    # On the first pass, skip the leading Wikipedia chrome header.
    if is_first_pass:
        start = _find_content_start_line(lines)
        lines = lines[start:]

    found_first_para = False
    cleaned: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Pure decoration: ``--- *** ===``, etc.
        if re.fullmatch(r"[\*\-\|\+#=\s\^]+", s):
            continue
        # Short bullet lines (<80 chars) — sidebar noise.
        if re.match(r"^[\*\-\+]", s) and len(s) < 80:
            continue
        # Numbered footnote lines: "1. ^ Foo" / "1. **^** Foo".
        if re.match(r"^\d+\.\s*(\^|\*\*\^\*\*)", s):
            continue

        is_heading = s.startswith("#")
        is_table_row = s.startswith("|")
        is_long = len(s) >= 60

        # Drop short non-heading / non-table lines that appear BEFORE the
        # first real paragraph — these are sidebar / toolbar fragments.
        if not found_first_para and not is_heading and not is_table_row and not is_long:
            continue

        if not is_heading:
            found_first_para = True

        cleaned.append(s)

    result = "\n".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _looks_like_orphan_label(line: str) -> bool:
    """True if ``line`` is a short label-only line with no sentence content."""
    if len(line) > 60:
        return False
    if any(ch in line for ch in ".!?"):
        return False
    return len(line.split()) <= 4
