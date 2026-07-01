import asyncio
import logging
import os
from typing import Annotated, List, Literal, Optional, Tuple, Type

import requests
from langchain.tools import BaseTool, InjectedState
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings
from app.soccer_agent.factory.llm_provider import get_llm
from app.soccer_agent.toolbox._config_loader import tool_description

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://google.serper.dev/search"
_SCRAPE_URL = "https://scrape.serper.dev"
_TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# Optional outbound proxy — set WEB_SEARCH_PROXY in .env to route web_search through a US VPS.
# e.g. WEB_SEARCH_PROXY=socks5h://localhost:1080  (requires: uv add 'requests[socks]')
# Intentionally NOT reading HTTP_PROXY/HTTPS_PROXY to avoid routing LiteLLM/Qdrant/Mongo through the proxy.
_proxy_url = os.environ.get("WEB_SEARCH_PROXY")
_PROXIES: dict | None = {"http": _proxy_url, "https": _proxy_url} if _proxy_url else None
# Serper /search always returns ~10 organic hits; we keep the top N in [5, 7].
_MIN_RESULTS = 5
_MAX_RESULTS = 7
# Per-document char cap — keeps the lead of each article (where the key facts
# usually are) without blowing up prompt size when a page scrapes to 100k+ chars.
_MAX_DOC_CHARS = 8000

_llm = None
_SYNTHESIS_SYSTEM = (
    "You are a soccer news analyst. Synthesize web articles into a concise factual answer.\n"
    "Follow this MANDATORY 3-step process before writing anything.\n\n"
    "## STEP 1 — Establish NOW\n"
    "The user message begins with: 'Current date/time (ICT/UTC+7): <timestamp>'\n"
    "Parse that timestamp as NOW. All date comparisons below use this value.\n"
    "If no timestamp is given, report facts without filtering and note the uncertainty.\n\n"
    "## STEP 2 — Classify query temporal intent\n"
    "Read the query and determine which time window the user is asking about:\n"
    "  FUTURE — user wants events that have NOT yet happened as of NOW:\n"
    "    Signals: 'next', 'upcoming', 'schedule', 'fixture', 'when will', 'sắp tới',\n"
    "    'tiếp theo', 'lịch thi đấu', 'khi nào', 'trận tới'\n"
    "  PAST — user wants events that have ALREADY happened before NOW:\n"
    "    Signals: 'recent', 'latest result', 'last match', 'scored', 'gần đây', 'vừa',\n"
    "    'mới nhất', 'kết quả', 'ai thắng', 'trận vừa rồi'\n"
    "  GENERAL — career stats, biography, standings (no strict date filtering needed)\n\n"
    "## STEP 3 — Filter events, then synthesize\n"
    "For every match or event in the articles that has an explicit date:\n"
    "  date < NOW  → mark PAST (already occurred)\n"
    "  date > NOW  → mark FUTURE (not yet played)\n"
    "Apply the filter:\n"
    "  FUTURE query  → keep ONLY events marked FUTURE; discard all PAST events\n"
    "  PAST query    → keep ONLY events marked PAST; discard all FUTURE events\n"
    "  GENERAL       → keep all relevant facts\n"
    "If no events survive the filter, say so explicitly:\n"
    "  e.g. 'No upcoming matches found after [NOW] in the available sources.'\n\n"
    "## OUTPUT\n"
    "Write a concise factual summary (max 300 words), flowing English prose, no bullet points.\n"
    "Cite concrete facts: dates, scores, names.\n"
    "NEVER present a past event as upcoming. NEVER present a future event as a completed result."
)


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("tool-text-only")
    return _llm


def _serper_search(query: str) -> List[dict]:
    """Run a Serper Google search; return the organic results (title/link/snippet)."""
    resp = requests.post(
        _SEARCH_URL,
        headers={"X-API-KEY": settings.SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "gl": "us", "hl": "en"},
        proxies=_PROXIES,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("organic") or []


def _serper_scrape(url: str) -> str:
    """Fetch full page text for a single URL via Serper's scrape endpoint."""
    resp = requests.post(
        _SCRAPE_URL,
        headers={"X-API-KEY": settings.SERPER_API_KEY, "Content-Type": "application/json"},
        json={"url": url, "includeMarkdown": False},
        proxies=_PROXIES,
        timeout=3,
    )
    resp.raise_for_status()
    return resp.json().get("text") or ""


def _tavily_scrape(url: str) -> str:
    """Fetch full page text for a single URL via Tavily's extract endpoint."""
    api_key = settings.TAVILY_API_KEYS[0] if settings.TAVILY_API_KEYS else ""
    resp = requests.post(
        _TAVILY_EXTRACT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"urls": [url], "extract_depth": "advanced", "format": "text"},
        proxies=_PROXIES,
        timeout=10,
    )
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results") or []
    if results:
        return results[0].get("raw_content") or ""
    # Log failed_results so we know WHY the URL wasn't extracted (auth wall, paywall, etc.)
    for failed in body.get("failed_results") or []:
        logger.warning(f"tavily_scrape failed for {failed.get('url')}: {failed.get('error')}")
    return ""


class WebNewsSearchInput(BaseModel):
    query: str = Field(
        description=(
            "The web search query. Should be specific and include the entity name "
            "(player/team/league) plus the angle (transfer, injury, match result, fixture, etc.). "
            "Example: 'Erling Haaland injury update', 'Real Madrid transfer news', "
            "'Lionel Messi World Cup 2026 schedule'."
        )
    )
    max_results: int = Field(
        default=5,
        description=(
            "How many top search results to fetch and read in full. "
            "Use 5 (default) for a focused single-entity query; increase toward 7 when the "
            f"query spans multiple entities. Clamped to [{_MIN_RESULTS}, {_MAX_RESULTS}]."
        ),
    )
    execution_agent_state: Annotated[dict, InjectedState] = Field(
        description="Injected worker state — provides time_context for temporal reasoning."
    )


class WebNewsSearchTool(BaseTool):
    name: str = "web_news_search"
    description: str = """
    Search for recent soccer/football news articles using the web.
    Use this tool when the user asks about current events, recent transfers, injury updates,
    upcoming fixtures, latest match results, or anything that requires up-to-date information
    that may not be in the static knowledge base.
    Do NOT use for historical stats, player career bios, or team founding history — use
    entity_augment for those. Do NOT use for specific past match data — use game tools for those.
    """
    args_schema: Type[BaseModel] = WebNewsSearchInput  # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def __init__(self):
        super().__init__(description=tool_description("web_news_search"))

    async def warmup(self) -> None:
        from app.soccer_agent.factory.llm_provider import warm_llm
        await warm_llm(_get_llm(), SystemMessage(content=_SYNTHESIS_SYSTEM), "web_news_search")

    def _run(
        self,
        query: str,
        execution_agent_state: Annotated[dict, InjectedState],
        max_results: int = 5,
        _run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[dict]]:
        return asyncio.run(self._arun(query, execution_agent_state, max_results=max_results))

    async def _arun(
        self,
        query: str,
        execution_agent_state: Annotated[dict, InjectedState],
        max_results: int = 5,
        _run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[dict]]:
        time_context = (execution_agent_state or {}).get("time_context")
        top_k = min(max(max_results, _MIN_RESULTS), _MAX_RESULTS)

        # 1. Search — Serper returns the full top-10; keep the top_k organic hits.
        try:
            organic = await asyncio.to_thread(_serper_search, query)
        except Exception as e:
            logger.error(f"web_news_search search failed: {e}", exc_info=True)
            return f"An error occurred while searching the web: {str(e)}.", []

        organic = organic[:top_k]
        if not organic:
            return f"No web results found for: {query}", []

        # 2. Scrape the top_k URLs in parallel — one document per URL.
        async def _fetch(hit: dict) -> dict:
            link = hit.get("link") or ""
            try:
                text = await asyncio.to_thread(_tavily_scrape, link)
            except Exception as e:
                logger.warning(f"web_news_search scrape failed for {link}: {e}")
                text = ""
            # Fall back to the snippet when the page can't be scraped; cap length.
            content = (text or hit.get("snippet") or "")[:_MAX_DOC_CHARS]
            return {
                "title": hit.get("title") or "",
                "link": link,
                "content": content,
            }

        results = await asyncio.gather(*(_fetch(h) for h in organic))
        results = [r for r in results if r.get("content")]
        if not results:
            return f"No readable web content found for: {query}", []

        # 3. Combine documents and synthesize an answer.
        doc_lines = []
        for i, r in enumerate(results, 1):
            doc_lines.append(f"\n=== Document {i}: {r['title']} ===")
            doc_lines.append(r["content"])
        documents = "\n".join(doc_lines)

        time_line = f"Current date/time (ICT/UTC+7): {time_context}\n" if time_context else ""
        try:
            synthesis = await _get_llm().ainvoke([
                SystemMessage(content=_SYNTHESIS_SYSTEM),
                HumanMessage(content=f"{time_line}Query: {query}\n\nArticles:\n{documents}"),
            ])
            answer = synthesis.content if hasattr(synthesis, "content") else str(synthesis)
        except Exception as e:
            logger.warning(f"web_news_search LLM synthesis failed: {e}, returning raw documents")
            answer = documents

        return answer, results
