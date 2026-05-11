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
    time_range: Literal["day", "week", "month"] = Field(
        default="week",
        description=(
            "Recency filter for news results. "
            "'day' = last 24 hours (breaking news), "
            "'week' = last 7 days (recent news, default), "
            "'month' = last 30 days (background context)."
        ),
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
        time_range: Literal["day", "week", "month"] = "week",
        _run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[dict]]:
        return asyncio.run(self._arun(query, time_range))

    async def _arun(
        self,
        query: str,
        time_range: Literal["day", "week", "month"] = "week",
        _run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[dict]]:
        service = _get_tavily()
        try:
            results = await service.search_news(query=query, time_range=time_range)
        except Exception as e:
            logger.error(f"web_news_search failed: {e}", exc_info=True)
            return (
                f"An error occurred while searching for news: {str(e)}.",
                [],
            )

        if not results:
            return f"No recent news found for: {query}", []

        lines = [f'[web_news_search results for: "{query}" | time_range={time_range}]']
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
