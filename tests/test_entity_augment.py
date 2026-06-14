"""Unit tests for entity_augment freshness gating + parallel stale path.

All LLM and Tavily I/O is mocked; only the freshness decision logic and the
_arun branch routing are exercised.
"""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schema.soccerwiki_entities import PlayerSchema
from app.schema.textual_entity_search import SearchingResult
from app.soccer_agent.toolbox.entity_augment import (
    _AugmentedAnswer,
    _is_fresh,
    _parse_dt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FMT = "%Y-%m-%d %H:%M:%S"


def _ts(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime(_FMT)


def _result_with(last_updated) -> SearchingResult:
    sr = SearchingResult()
    sr.found_entities.append(
        PlayerSchema(
            NAME="Lionel Messi",
            ENTITY_TYPE="player",
            LAST_UPDATED=last_updated,
            PLAYER_URL="http://wiki/messi",
        )
    )
    return sr


@pytest.fixture
def tool():
    with patch("app.soccer_agent.toolbox.entity_augment.get_llm", return_value=MagicMock()):
        from app.soccer_agent.toolbox.entity_augment import EntityAugmentTool

        return EntityAugmentTool()


def _wiki_payload(source="wiki_extract"):
    return {
        "web_context": "wiki context",
        "source": source,
        "name": "Lionel Messi",
        "entity_type": "player",
        "is_missing": False,
        "wiki_url": "http://wiki/messi",
        "cleaned": "cleaned wiki content",
        "images": ["img1"],
    }


# ---------------------------------------------------------------------------
# _parse_dt
# ---------------------------------------------------------------------------

def test_parse_dt_plain_and_utc_suffix():
    assert _parse_dt("2026-06-10 12:00:00") == datetime(2026, 6, 10, 12, 0, 0)
    assert _parse_dt("2026-06-14 12:00:00 UTC") == datetime(2026, 6, 14, 12, 0, 0)


def test_parse_dt_none_and_garbage():
    assert _parse_dt(None) is None
    assert _parse_dt("") is None
    assert _parse_dt("not-a-date") is None


# ---------------------------------------------------------------------------
# _is_fresh
# ---------------------------------------------------------------------------

def test_is_fresh_recent_within_threshold():
    sr = _result_with(_ts(2))
    assert _is_fresh(sr, _ts(0) + " UTC") is True


def test_is_fresh_stale_beyond_threshold():
    sr = _result_with(_ts(10))
    assert _is_fresh(sr, _ts(0)) is False


def test_is_fresh_missing_last_updated_is_stale():
    sr = _result_with(None)
    assert _is_fresh(sr, _ts(0)) is False


def test_is_fresh_no_found_entities_is_stale():
    assert _is_fresh(SearchingResult(), _ts(0)) is False


def test_is_fresh_any_stale_entity_makes_whole_result_stale():
    sr = SearchingResult()
    sr.found_entities.append(PlayerSchema(NAME="A", LAST_UPDATED=_ts(1)))
    sr.found_entities.append(PlayerSchema(NAME="B", LAST_UPDATED=None))
    assert _is_fresh(sr, _ts(0)) is False


# ---------------------------------------------------------------------------
# _arun branch routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fresh_trusts_db_skips_tavily(tool):
    """Fresh DB data → return DB answer even when has_sufficient_info=False; no fetch."""
    sr = _result_with(_ts(1))
    tool._query_database = MagicMock(return_value=sr)
    tool._generate_answer_from_db = AsyncMock(
        return_value=_AugmentedAnswer(answer="DB ANSWER", has_sufficient_info=False, unknown_entities=[])
    )
    tool._fetch_wiki_context = AsyncMock()

    answer, result = await tool._arun(["Lionel Messi"], "q", time_context=_ts(0) + " UTC")

    assert answer == "DB ANSWER"
    tool._fetch_wiki_context.assert_not_called()
    assert result._upsert_payload is None


@pytest.mark.asyncio
async def test_stale_db_sufficient_returns_db_and_refreshes_mongo(tool):
    """Stale + DB sufficient → return DB answer, still pack payload to refresh Mongo."""
    sr = _result_with(_ts(10))
    tool._query_database = MagicMock(return_value=sr)
    tool._generate_answer_from_db = AsyncMock(
        return_value=_AugmentedAnswer(answer="DB SUFFICIENT", has_sufficient_info=True, unknown_entities=[])
    )
    tool._fetch_wiki_context = AsyncMock(return_value=_wiki_payload())
    tool._generate_web_answer = AsyncMock(return_value="WEB ANSWER")

    answer, result = await tool._arun(["Lionel Messi"], "q", time_context=_ts(0))

    assert answer == "DB SUFFICIENT"
    tool._fetch_wiki_context.assert_called_once()
    tool._generate_web_answer.assert_not_called()
    assert result._upsert_payload is not None
    assert result._upsert_payload["source"] == "wiki_extract"
    assert result._upsert_payload["cleaned_content"] == "cleaned wiki content"


@pytest.mark.asyncio
async def test_stale_db_insufficient_uses_wiki_answer(tool):
    """Stale + DB insufficient → regenerate from fetched wiki + pack payload."""
    sr = _result_with(_ts(10))
    tool._query_database = MagicMock(return_value=sr)
    tool._generate_answer_from_db = AsyncMock(
        return_value=_AugmentedAnswer(answer="DB PARTIAL", has_sufficient_info=False, unknown_entities=[])
    )
    tool._fetch_wiki_context = AsyncMock(return_value=_wiki_payload())
    tool._generate_web_answer = AsyncMock(return_value="WEB ANSWER")

    answer, result = await tool._arun(["Lionel Messi"], "q", time_context=_ts(0))

    assert answer == "WEB ANSWER"
    tool._generate_web_answer.assert_called_once()
    assert result._upsert_payload["source"] == "wiki_extract"


@pytest.mark.asyncio
async def test_stale_wiki_fetch_fails_falls_back_to_db(tool):
    """Stale + DB insufficient + wiki fetch unavailable → return DB answer, no payload."""
    sr = _result_with(_ts(10))
    tool._query_database = MagicMock(return_value=sr)
    tool._generate_answer_from_db = AsyncMock(
        return_value=_AugmentedAnswer(answer="DB PARTIAL", has_sufficient_info=False, unknown_entities=[])
    )
    tool._fetch_wiki_context = AsyncMock(return_value=None)
    tool._generate_web_answer = AsyncMock(return_value="WEB ANSWER")

    answer, result = await tool._arun(["Lionel Messi"], "q", time_context=_ts(0))

    assert answer == "DB PARTIAL"
    tool._generate_web_answer.assert_not_called()
    assert result._upsert_payload is None


@pytest.mark.asyncio
async def test_stale_general_search_source_not_upserted_to_mongo(tool):
    """Stale + general_search fetch → use wiki answer but do NOT pack Mongo payload."""
    sr = _result_with(_ts(10))
    tool._query_database = MagicMock(return_value=sr)
    tool._generate_answer_from_db = AsyncMock(
        return_value=_AugmentedAnswer(answer="DB PARTIAL", has_sufficient_info=False, unknown_entities=[])
    )
    tool._fetch_wiki_context = AsyncMock(return_value=_wiki_payload(source="general_search"))
    tool._generate_web_answer = AsyncMock(return_value="WEB ANSWER")

    answer, result = await tool._arun(["Lionel Messi"], "q", time_context=_ts(0))

    assert answer == "WEB ANSWER"
    assert result._upsert_payload is None
