"""
Tavily integration for the Soccer Agent.

Wraps the async Tavily client behind two helpers:

- ``TavilyKeyPool`` rotates between multiple API keys (configured via
  ``TAVILY_API_KEYS``) and applies a per-key cooldown when Tavily returns a
  rate-limit error. When every key is cooling down the caller is suspended
  until the earliest cooldown expires.

- ``TavilyService`` exposes the high-level operations the toolbox /
  migration script need: probe whether a Wikipedia URL belongs to a given
  entity, find a Wikipedia URL for an entity, extract Wikipedia markdown,
  search news, and a single-domain general fallback search.

All calls go through ``_call_with_failover`` so a 429 on one key transparently
retries on the next available key. Each call instantiates a fresh
``AsyncTavilyClient`` because the client is cheap and binding the chosen key
makes failover semantics straightforward.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, List, Optional, TypeVar
from urllib.parse import unquote, urlparse

from app.config import settings

logger = logging.getLogger(__name__)


_DEFAULT_COOLDOWN_SECONDS = 60.0
_MAX_RETRY_AFTER_SECONDS = 600.0
# BM25 score (slug as document, entity name as query) divided by the score the
# query would get against itself. ``0.0`` = no token overlap, ``1.0`` = every
# query token is in the slug. ``0.4`` accepts disambiguator slugs and
# truncated/abbreviated slugs (``Unai_Arieta`` for ``Unai Arietaleanizbeaskoa``)
# while rejecting unrelated club / season / rivalry pages.
_PERSONAL_WIKI_THRESHOLD = 0.4

# Wikipedia disambiguator words we strip before tokenisation so they don't
# dilute the BM25 score (e.g. ``Igor_Silva_(footballer)`` shouldn't lose
# points just because it has the suffix ``footballer``).
_WIKI_STOPWORDS: frozenset[str] = frozenset({
    "footballer",
    "footballers",
    "soccer",
    "player",
    "players",
    "referee",
    "coach",
    "manager",
    "born",
    "the",
})

# Synthetic filler documents used to anchor BM25 IDF when the real corpus is
# only ``[slug, name]``. Without them, terms appearing in both real docs get a
# ``df = N`` and BM25Okapi produces negative IDF (and therefore negative match
# scores). Twelve unique sentinel tokens push every real query term back into
# the ``df < N`` regime so IDF stays comfortably positive.
_BM25_FILLER_DOCS: list[list[str]] = [
    [f"__bm25filler{i}__"] for i in range(12)
]

# Default soccer-news domains for ``search_news`` — overridable per call.

T = TypeVar("T")


@dataclass
class _KeyState:
    """Per-key state inside :class:`TavilyKeyPool`."""

    api_key: str
    cooldown_until: float = 0.0  # unix timestamp; 0 means available now


@dataclass
class TavilyExtractResult:
    """Result of an extract call. ``markdown`` is None when extraction failed."""

    url: str
    markdown: Optional[str]
    images: List[str] = field(default_factory=list)


class TavilyKeyPool:
    """Round-robin pool with per-key cooldowns triggered by 429-style errors."""

    def __init__(self, keys: Iterable[str]):
        cleaned = [k for k in (k.strip() for k in keys) if k]
        if not cleaned:
            raise ValueError(
                "TavilyKeyPool requires at least one API key. "
                "Set TAVILY_API_KEYS (comma-separated) or TAVILY_API_KEY in the environment."
            )
        self._states: List[_KeyState] = [_KeyState(k) for k in cleaned]
        self._idx: int = -1  # advance to 0 on first acquire
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._states)

    async def acquire(self) -> str:
        """Return the next available API key.

        If every key is cooling down, sleep until the earliest cooldown
        expires and try again.
        """
        while True:
            async with self._lock:
                now = time.time()
                for _ in range(len(self._states)):
                    self._idx = (self._idx + 1) % len(self._states)
                    state = self._states[self._idx]
                    if state.cooldown_until <= now:
                        return state.api_key
                wait = max(s.cooldown_until for s in self._states) - now
                wait = min(max(wait, 1.0), _MAX_RETRY_AFTER_SECONDS)
            logger.info(f"[TavilyKeyPool] All keys cooling down — sleeping {wait:.1f}s")
            await asyncio.sleep(wait)

    def mark_rate_limited(self, api_key: str, retry_after: float = _DEFAULT_COOLDOWN_SECONDS) -> None:
        """Mark ``api_key`` as cooling down for ``retry_after`` seconds."""
        retry_after = min(max(retry_after, 1.0), _MAX_RETRY_AFTER_SECONDS)
        for state in self._states:
            if state.api_key == api_key:
                state.cooldown_until = time.time() + retry_after
                logger.warning(
                    f"[TavilyKeyPool] Key …{api_key[-4:]} cooling down for {retry_after:.0f}s"
                )
                return


class TavilyService:
    """High-level façade for Tavily search + extract operations.

    Construct once and share across the process. Every public method routes
    its underlying Tavily call through :meth:`_call_with_failover` so a single
    rate-limited key never breaks the request — the next healthy key picks up.
    """

    def __init__(
        self,
        keys: Optional[Iterable[str]] = None,
        cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
    ):
        api_keys = list(keys) if keys is not None else list(settings.TAVILY_API_KEYS)
        self._pool = TavilyKeyPool(api_keys)
        self._cooldown_seconds = cooldown_seconds

    # ── Public API ────────────────────────────────────────────────────────

    @staticmethod
    def is_personal_wiki(url: str, entity_name: str) -> bool:
        """Heuristic: does this Wikipedia URL belong to ``entity_name``?

        Uses BM25 token scoring (via ``rank_bm25``) on the URL slug as the
        document and the entity name as the query. The raw BM25 score is
        divided by the score the query gets against itself so we can compare
        it to a fixed threshold in ``[0, 1]``. Stopwords like ``footballer``
        / ``born`` (Wikipedia disambiguator chrome) are stripped before
        tokenisation so they don't pollute the score.
        """
        if not url or not entity_name:
            return False
        slug = TavilyService._wiki_slug(url)
        if not slug:
            return False
        slug_tokens = TavilyService._tokenise(slug)
        name_tokens = TavilyService._tokenise(entity_name)
        if not slug_tokens or not name_tokens:
            return False

        from rank_bm25 import BM25Okapi

        # Build the BM25 corpus from the slug + name + a fixed set of
        # synthetic fillers; the fillers are essential so that terms shared
        # between the slug and the name don't end up with ``df == N`` (which
        # would push their IDF negative under BM25Okapi).
        corpus = [slug_tokens, name_tokens] + _BM25_FILLER_DOCS
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(name_tokens)
        slug_score, max_score = float(scores[0]), float(scores[1])
        if max_score <= 0:
            return False
        return (slug_score / max_score) >= _PERSONAL_WIKI_THRESHOLD

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lowercase, transliterate non-Latin → Latin, split on non-alphanumeric, drop wiki stopwords.

        Wikipedia URLs sometimes use the local-language slug (e.g. Russian
        Wikipedia exposes ``/wiki/Когут,_Игорь_Романович`` for Igor Kogut).
        After ``unquote`` we transliterate via :mod:`unidecode` so the
        Cyrillic / Greek / Vietnamese / etc. tokens land on the same Latin
        alphabet as the entity name we're matching against.
        """
        if not text:
            return []
        try:
            from unidecode import unidecode

            normalised = unidecode(text)
        except Exception:
            normalised = text
        tokens = re.split(r"[^a-zA-Z0-9]+", normalised.lower())
        return [t for t in tokens if t and t not in _WIKI_STOPWORDS and not t.isdigit()]

    async def find_wiki_url(
        self,
        entity_name: str,
        max_results: int = 5,
        score_threshold: float = 0.8,
        prefer_lang: str = "en",
    ) -> Optional[str]:
        """Return the most likely Wikipedia URL for ``entity_name``.

        Searches across all Wikipedia language editions (``wikipedia.org``)
        and applies three filters in order:

        1. Tavily ``score`` strictly greater than ``score_threshold`` (default
           0.9 — only confident hits qualify).
        2. Prefer URLs on the requested language Wikipedia
           (``{prefer_lang}.wikipedia.org``); within that subset return the
           highest-score URL.
        3. If no URL on the preferred language passes, fall back to the
           highest-score URL among the remaining filtered results.

        Returns ``None`` when no result clears the score filter.
        """

        async def _call(client):
            return await client.search(
                query=entity_name,
                include_domains=["en.wikipedia.org","wikipedia.org"],
                max_results=max_results,
                search_depth="advanced",
                exact_match = True,
                chunks_per_source=1,
                topic="general"
                
            )

        try:
            response = await self._call_with_failover(_call)
        except Exception as e:
            logger.warning(f"[TavilyService] find_wiki_url failed for {entity_name!r}: {e}")
            return None

        results = response.get("results", []) or []
        filtered = [r for r in results if r.get("url") and (r.get("score") or 0.0) > score_threshold]
        if not filtered:
            return None

        # BM25 check: keep only URLs whose slug matches the entity name.
        validated = [r for r in filtered if self.is_personal_wiki(r["url"], entity_name)]
        pool_all = validated or filtered  # fall back to score-only if BM25 rejects everything

        preferred_marker = f"//{prefer_lang}.wikipedia.org/"
        preferred = [r for r in pool_all if preferred_marker in r["url"]]
        pool = preferred or pool_all
        best = max(pool, key=lambda r: r.get("score") or 0.0)
        return best["url"]

    async def extract_wiki(self, url: str) -> Optional[TavilyExtractResult]:
        """Extract a Wikipedia page as full markdown, retrying with basic depth on failure."""
        for depth in ("advanced", "basic"):
            kwargs = {
                "urls": url,
                "extract_depth": depth,
                "format": "markdown",
                "include_images": False,
            }

            async def _call(client, _kwargs=kwargs):
                return await client.extract(**_kwargs)

            try:
                response = await self._call_with_failover(_call)
            except Exception as e:
                logger.warning(
                    f"[TavilyService] extract_wiki(depth={depth}) failed for {url}: {e}"
                )
                continue

            extracted = self._first_extract_result(response)
            if extracted is not None and extracted.markdown:
                return extracted
            logger.info(f"[TavilyService] extract_wiki(depth={depth}) empty for {url}")
        return None

    async def extract_batch(
        self,
        urls: List[str],
        query: str = "",
        urls_per_call: int = 5,
        max_concurrency: int = 3,
    ) -> List[Optional[TavilyExtractResult]]:
        """Extract many Wikipedia URLs efficiently.

        Tavily's extract endpoint accepts a list of URLs per call and bills
        one credit per 5 successful URLs (basic) — so batching is ~5× cheaper
        than issuing one call per URL. We chunk ``urls`` into groups of
        ``urls_per_call`` (default 5, matching the pricing block) and dispatch
        up to ``max_concurrency`` chunks in parallel. URLs that fail at
        ``advanced`` depth are retried within the same chunk at ``basic``.

        Returns a list aligned to the input ``urls`` (``None`` for URLs that
        couldn't be extracted at either depth).
        """
        if not urls:
            return []

        results_map: dict[str, Optional[TavilyExtractResult]] = {u: None for u in urls}
        sem = asyncio.Semaphore(max_concurrency)

        async def _do_chunk(chunk: list[str]) -> None:
            async with sem:
                for depth in ("advanced", "basic"):
                    pending = [u for u in chunk if results_map.get(u) is None]
                    if not pending:
                        return
                    kwargs: dict = {
                        "urls": pending,
                        "extract_depth": depth,
                        "format": "markdown",
                        "include_images": True,
                    }
                    if query:
                        kwargs["query"] = query
                        kwargs["chunks_per_source"] = 5

                    async def _call(client, _kwargs=kwargs):
                        return await client.extract(**_kwargs)

                    try:
                        response = await self._call_with_failover(_call)
                    except Exception as e:
                        logger.warning(
                            f"[TavilyService] extract_batch chunk failed at depth={depth} "
                            f"({len(pending)} URL(s)): {e}"
                        )
                        continue

                    for item in response.get("results", []) or []:
                        url = item.get("url")
                        markdown = item.get("raw_content") or item.get("content")
                        if not url or not markdown:
                            continue
                        # Tavily may canonicalize URLs (e.g. trailing slash);
                        # match against the original input.
                        original = url if url in results_map else next(
                            (u for u in pending if u == url or u.rstrip("/") == url.rstrip("/")),
                            None,
                        )
                        if original is None:
                            continue
                        results_map[original] = TavilyService._build_extract_item(item)

        chunks = [urls[i:i + urls_per_call] for i in range(0, len(urls), urls_per_call)]
        await asyncio.gather(*(_do_chunk(chunk) for chunk in chunks))
        return [results_map[u] for u in urls]

    @staticmethod
    def _build_extract_item(item: dict) -> TavilyExtractResult:
        """Convert a single ``results[]`` entry from a Tavily extract response."""
        url = item.get("url", "")
        markdown = item.get("raw_content") or item.get("content")
        images = item.get("images") or []
        if isinstance(images, list):
            image_urls = [img if isinstance(img, str) else img.get("url", "") for img in images]
            image_urls = [u for u in image_urls if u]
        else:
            image_urls = []
        return TavilyExtractResult(url=url, markdown=markdown, images=image_urls)

    async def search_news(
        self,
        query: str,
        time_range: str = "week",
        start_date: Optional[str] = None,
        exact_match: bool = False,
        max_results: int = 10,
        score_threshold: float = 0.5,
        search_depth: str = "basic",
        include_answer: str = "basic",
    ) -> tuple[Optional[str], List[dict]]:
        """Search recent soccer news via Tavily ``topic="news"``.

        Returns ``(answer, results)`` where ``answer`` is Tavily's synthesised
        summary and ``results`` is the list of matching articles.

        When start_date (YYYY-MM-DD) is provided, time_range is ignored per Tavily API rules.
        When exact_match is True, the query is wrapped in quotes for exact phrase matching.
        """
        search_query = f'"{query}"' if exact_match else query
        kwargs: dict = dict(
            query=search_query,
            topic="news",
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            chunks_per_source=5,
        )
        if start_date:
            kwargs["start_published_date"] = start_date
        else:
            kwargs["time_range"] = time_range

        async def _call(client):
            return await client.search(**kwargs)

        try:
            response = await self._call_with_failover(_call)
        except Exception as e:
            logger.warning(f"[TavilyService] search_news failed for {query!r}: {e}")
            return None, []

        answer: Optional[str] = response.get("answer") or None
        results = [
            self._slim_result(r)
            for r in (response.get("results", []) or [])
            if (r.get("score") or 0.0) >= score_threshold
        ]
        return answer, results

    async def search_general(
        self,
        query: str,
        max_results: int = 20,
        score_threshold: float = 0.5,
        search_depth: str = "basic",
        include_answer: str = "basic",
        time_range: str = "month",
    ) -> tuple[Optional[str], List[dict]]:
        """Domain-less fallback search — used when extraction fails entirely."""
        kwargs: dict = dict(
            query=query,
            topic="general",
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            chunks_per_source=5,
            time_range=time_range,
        )

        async def _call(client):
            return await client.search(**kwargs)

        try:
            response = await self._call_with_failover(_call)
        except Exception as e:
            logger.warning(f"[TavilyService] search_general failed for {query!r}: {e}")
            return None, []

        answer: Optional[str] = response.get("answer") or None
        results = [
            self._slim_result(r)
            for r in (response.get("results", []) or [])
            if (r.get("score") or 0.0) >= score_threshold
        ]
        return answer, results

    async def search_combine(
        self,
        query: str,
        time_range: str = "week",
        start_date: Optional[str] = None,
        exact_match: bool = False,
        max_results: int = 5,
        score_threshold: float = 0.5,
        search_depth: str = "basic",
        include_answer: str = "basic",
    ) -> tuple[Optional[str], List[dict]]:
        """Run search_news and search_general in parallel and merge results.

        Returns ``(answer, results)`` with deduplication by URL.
        news results take priority over general on URL collision.
        """
        (news_answer, news_results), (general_answer, general_results) = await asyncio.gather(
            self.search_news(
                query=query,
                time_range=time_range,
                start_date=start_date,
                exact_match=exact_match,
                max_results=max_results,
                score_threshold=score_threshold,
                search_depth=search_depth,
                include_answer=include_answer,
            ),
            self.search_general(
                query=query,
                max_results=max_results,
                score_threshold=score_threshold,
                search_depth=search_depth,
                include_answer=include_answer,
            ),
        )

        # Dedup by title (URL is stripped by _slim_result); news takes priority
        seen: dict[str, dict] = {}
        for r in general_results:
            key = r.get("title") or ""
            if key:
                seen[key] = r
        for r in news_results:
            key = r.get("title") or ""
            if key:
                seen[key] = r
        results = sorted(seen.values(), key=lambda r: r.get("score") or 0.0, reverse=True)

        parts = [a for a in (news_answer, general_answer) if a]
        answer = "\n\n".join(parts) if parts else None
        return answer, results

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _slim_result(r: dict) -> dict:
        """Keep only title, content, and score — drop url, published_date, raw_content, etc."""
        return {
            "title": r.get("title") or "",
            "content": r.get("content") or "",
            "score": r.get("score") or 0.0,
        }

    async def _call_with_failover(
        self,
        fn: Callable[..., Awaitable[T]],
    ) -> T:
        """Run ``fn(client)`` with the next healthy key, retrying on 429."""
        # Imported lazily so importing the module doesn't require the package
        # to be installed (e.g. while typing tests offline).
        from tavily import AsyncTavilyClient

        last_err: Optional[Exception] = None
        attempts = max(self._pool.size, 1)
        for _ in range(attempts):
            api_key = await self._pool.acquire()
            client = AsyncTavilyClient(api_key=api_key)
            try:
                return await fn(client)
            except Exception as e:  # noqa: BLE001 — narrow below
                if self._is_rate_limit(e):
                    cooldown = self._extract_retry_after(e) or self._cooldown_seconds
                    self._pool.mark_rate_limited(api_key, retry_after=cooldown)
                    last_err = e
                    continue
                raise
        if last_err is not None:
            raise last_err
        raise RuntimeError("TavilyService failover exhausted with no error captured")

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """Detect 429 / quota errors across tavily-python and HTTP wrappers."""
        try:
            from tavily import errors as tavily_errors  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover — older tavily without errors module
            tavily_errors = None

        if tavily_errors is not None:
            usage_cls = getattr(tavily_errors, "UsageLimitExceededError", None)
            if usage_cls is not None and isinstance(exc, usage_cls):
                return True

        message = str(exc).lower()
        if "429" in message:
            return True
        return any(
            marker in message
            for marker in ("rate limit", "rate-limit", "too many requests", "usage limit")
        )

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        """Best-effort parse of Retry-After / similar hints from the exception."""
        for attr in ("retry_after", "response"):
            value = getattr(exc, attr, None)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if hasattr(value, "headers"):
                ra = value.headers.get("Retry-After") if value.headers else None
                if ra:
                    try:
                        return float(ra)
                    except ValueError:
                        return None
        return None

    @staticmethod
    def _first_extract_result(response: dict) -> Optional[TavilyExtractResult]:
        results = response.get("results", []) or []
        if not results:
            return None
        return TavilyService._build_extract_item(results[0])

    @staticmethod
    def _wiki_slug(url: str) -> str:
        """Extract the slug after ``/wiki/`` (URL-decoded). Returns "" if not a wiki URL."""
        try:
            parsed = urlparse(url)
        except Exception:
            return ""
        path = unquote(parsed.path or "")
        marker = "/wiki/"
        idx = path.find(marker)
        if idx == -1:
            return ""
        slug = path[idx + len(marker):]
        # Strip section anchors / trailing slashes
        slug = slug.split("#", 1)[0].rstrip("/")
        return slug

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip diacritics/punctuation/separators for fuzzy comparison."""
        if not text:
            return ""
        # Collapse separators (spaces, dashes, underscores) before stripping
        # diacritics so "Ángel-Di_Maria" matches "Angel Di Maria".
        cleaned = re.sub(r"[\s_\-]+", "", text)
        try:
            import unicodedata

            cleaned = unicodedata.normalize("NFKD", cleaned)
            cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
        except Exception:
            pass
        return re.sub(r"[^a-z0-9]+", "", cleaned.lower())
