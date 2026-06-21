"""Unit tests for CaseBankRetriever — Qdrant and embedding calls are mocked."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.soccer_agent.case_bank.retriever import CaseBankRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(use_case: str, label: str = "positive", score: float = 0.9):
    hit = MagicMock()
    hit.payload = {
        "use_case": use_case,
        "label": label,
        "tool_chains": '["game_info_retrieval"]',
        "sub_queries": '["Who scored?"]',
        "reasoning": "because reasons",
    }
    hit.score = score
    return hit


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _make_retriever(self):
        with patch("app.soccer_agent.case_bank.retriever.GoogleGenerativeAIEmbeddings"):
            r = CaseBankRetriever()
        r._embeddings = MagicMock()
        r._embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)
        return r

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_string(self):
        r = self._make_retriever()
        with patch.object(r, "_search", new=AsyncMock(return_value=[])):
            result = await r.retrieve("any query", has_media=False)
        assert result == ""

    @pytest.mark.asyncio
    async def test_top_3_use_cases_selected(self):
        r = self._make_retriever()
        hits = (
            [_make_hit("uc_A"), _make_hit("uc_B"), _make_hit("uc_A")]  # positive
        )
        negative_hits = [
            _make_hit("uc_C", label="negative"),
            _make_hit("uc_A", label="negative"),
            _make_hit("uc_B", label="negative"),
        ]

        async def fake_search(vector, has_media, label, top_k):
            return hits if label == "positive" else negative_hits

        with patch.object(r, "_search", side_effect=fake_search):
            result = await r.retrieve("query", has_media=False)

        assert "uc_A" in result
        assert "uc_B" in result
        assert "uc_C" in result

    @pytest.mark.asyncio
    async def test_uses_top_3_most_common_use_cases(self):
        r = self._make_retriever()
        # uc_A: 3 hits, uc_B: 2 hits, uc_C: 1 hit, uc_D: 1 hit → top3 = A, B, then C or D
        positive = [_make_hit("uc_A"), _make_hit("uc_A"), _make_hit("uc_B")]
        negative = [
            _make_hit("uc_A", label="negative"),
            _make_hit("uc_B", label="negative"),
            _make_hit("uc_C", label="negative"),
            _make_hit("uc_D", label="negative"),
            _make_hit("uc_D", label="negative"),
            _make_hit("uc_D", label="negative"),
        ]

        async def fake_search(vector, has_media, label, top_k):
            return positive if label == "positive" else negative

        with patch.object(r, "_search", side_effect=fake_search):
            result = await r.retrieve("query", has_media=True)

        assert "uc_A" in result
        assert "uc_B" in result
        # uc_D has 3 hits total, should appear; uc_C has only 1
        assert "uc_D" in result

    @pytest.mark.asyncio
    async def test_search_error_isolated_returns_empty_list(self):
        r = self._make_retriever()
        with patch(
            "app.soccer_agent.case_bank.retriever.qdrant_service.asearch",
            new=AsyncMock(side_effect=Exception("Qdrant unreachable")),
        ):
            result = await r._search([0.1] * 768, has_media=False, label="positive", top_k=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_has_media_forwarded_to_both_search_calls(self):
        r = self._make_retriever()
        calls_received = []

        async def fake_search(vector, has_media, _label, _top_k):
            calls_received.append(has_media)
            return []

        with patch.object(r, "_search", side_effect=fake_search):
            await r.retrieve("query", has_media=True)

        assert calls_received == [True, True]

    @pytest.mark.asyncio
    async def test_format_includes_correct_and_wrong_tags(self):
        r = self._make_retriever()
        positive = [_make_hit("uc_A", label="positive")]
        negative = [_make_hit("uc_A", label="negative")]

        async def fake_search(vector, has_media, label, top_k):
            return positive if label == "positive" else negative

        with patch.object(r, "_search", side_effect=fake_search):
            result = await r.retrieve("query", has_media=False)

        assert "✅ Correct" in result
        assert "❌ Wrong" in result

    @pytest.mark.asyncio
    async def test_malformed_payload_json_falls_back_to_raw(self):
        r = self._make_retriever()
        hit = MagicMock()
        hit.payload = {
            "use_case": "uc_X",
            "label": "positive",
            "tool_chains": "not valid json {{",
            "sub_queries": "also bad",
            "reasoning": "reason",
        }

        async def fake_search(vector, has_media, label, top_k):
            return [hit] if label == "positive" else []

        with patch.object(r, "_search", side_effect=fake_search):
            result = await r.retrieve("query", has_media=False)

        assert "not valid json" in result


# ---------------------------------------------------------------------------
# _format_examples
# ---------------------------------------------------------------------------

class TestFormatExamples:
    def _make_retriever(self):
        with patch("app.soccer_agent.case_bank.retriever.GoogleGenerativeAIEmbeddings"):
            return CaseBankRetriever()

    def test_empty_grouped_returns_empty_string(self):
        r = self._make_retriever()
        assert r._format_examples({}) == ""

    def test_single_use_case_formatted(self):
        r = self._make_retriever()
        hit = _make_hit("uc_test")
        result = r._format_examples({"uc_test": [hit]})
        assert "uc_test" in result
        assert "✅ Correct" in result
        assert "because reasons" in result

    def test_use_case_with_empty_hits_produces_header_only(self):
        r = self._make_retriever()
        result = r._format_examples({"uc_empty": []})
        assert "uc_empty" in result
        assert "✅" not in result
        assert "❌" not in result
