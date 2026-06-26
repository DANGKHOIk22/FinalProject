import asyncio
import logging
from typing import Any, List, Literal, Optional, Tuple, Type

from langchain.tools import BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from pydantic import BaseModel, Field

from app.soccer_agent.services.tavily_service import TavilyService
from app.soccer_agent.toolbox._config_loader import tool_description

logger = logging.getLogger(__name__)

_tavily_service: Optional[TavilyService] = None


def _get_tavily() -> TavilyService:
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = TavilyService()
    return _tavily_service


class WebNewsSearchInput(BaseModel):
    query: str = Field(
        description=(
            "The news search query. Should be specific and include the entity name "
            "(player/team/league) plus the news angle (transfer, injury, match result, etc.). "
            "Example: 'Erling Haaland injury update', 'Real Madrid transfer news', "
            "'Premier League results this week'."
        )
    )
    time_range: Literal["day", "week", "month", "year"] = Field(
        default="month",
        description=(
            "Recency filter based on the temporal scope of the user query. "
            "'day': hôm nay / hôm qua / today / yesterday / breaking news. "
            "'week': tuần này / tuần trước / this week / last week / recent days. "
            "'month': tháng này / tháng trước / gần đây / recently / this month / last month (default). "
            "'year': mùa giải / năm nay / năm ngoái / this season / last season / this year / last year."
        ),
    )
    exact_match: bool = Field(
        default=False,
        description=(
            "When True, wraps the query in quotes for exact phrase matching. "
            "Use when the user asks about a very specific team name, player name, "
            "or match title and broad results are likely to be noisy."
        ),
    )
    start_date: Optional[str] = Field(
        default=None,
        description=(
            "Start date filter in YYYY-MM-DD format (with leading zeros, e.g. '2026-05-03'). "
            "When provided, overrides time_range. "
            "Use when the user specifies a concrete date or date range start "
            "(e.g. 'ngày 3/5/2026' → '2026-05-03'). Leave None for relative ranges."
        ),
    )
    time_context: Optional[str] = Field(
        default=None,
        description="Current date and time for temporal reasoning."
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

    _llm: Any = None

    def __init__(self):
        super().__init__(description=tool_description("web_news_search"))

    def _run(
        self,
        query: str,
        time_range: Literal["day", "week", "month", "year"] = "week",
        exact_match: bool = False,
        start_date: Optional[str] = None,
        _run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[dict]]:
        return asyncio.run(self._arun(query, time_range, exact_match, start_date))

    async def _arun(
        self,
        query: str,
        time_range: Literal["day", "week", "month", "year"] = "week",
        exact_match: bool = False,
        start_date: Optional[str] = None,
        time_context: Optional[str] = None,
        _run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[dict]]:
        _RANGE_UP: dict[str, str] = {"day": "week", "week": "month", "month": "year", "year":None}
        general_time_range = _RANGE_UP[time_range]

        service = _get_tavily()
        try:
            (news_answer, news_results), (general_answer, general_results) = await asyncio.gather(
                service.search_news(
                    query=query, time_range=time_range, start_date=start_date,
                    exact_match=exact_match, max_results=5,
                ),
                service.search_general(query, max_results=5, time_range=general_time_range),
            )
        except Exception as e:
            logger.error(f"web_news_search failed: {e}", exc_info=True)
            return (
                f"An error occurred while searching for news: {str(e)}.",
                [],
            )

        seen: dict[str, dict] = {}
        for r in general_results:
            url = r.get("url") or ""
            if url:
                seen[url] = r
        for r in news_results:
            url = r.get("url") or ""
            if url:
                seen[url] = r
        results = sorted(seen.values(), key=lambda r: r.get("published_date") or "", reverse=True)

        parts = [a for a in (news_answer, general_answer) if a]
        tavily_answer = "\n\n".join(parts) if parts else None

        if not results and not tavily_answer:
            return f"No recent news found for: {query}", []

        lines = [f'[web_news_search results for: "{query}" | time_range={time_range}]']
        if tavily_answer:
            lines.append(f"\nSummary: {tavily_answer}")
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            published = r.get("published_date", "")
            content = r.get("content", "")
            date_part = f" ({published})" if published else ""
            lines.append(f"\n{i}. {title}{date_part} — {url}")
            if content:
                lines.append(f"   {content[:300]}")

        return "\n".join(lines), results
