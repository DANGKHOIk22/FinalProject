"""Unit tests for game_retrieval tools — all MongoDB calls are mocked."""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.soccer_agent.toolbox.game_retrieval import (
    _GameFinder,
    _GameFound,
    _GameNotFound,
    _GameSearchError,
)
from app.schema.match import MatchInfo, Annotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match_info(**kwargs) -> MatchInfo:
    defaults = dict(
        team1="unknown", team2="unknown",
        league="unknown", season="unknown",
        year="unknown", month="unknown",
    )
    defaults.update(kwargs)
    return MatchInfo(**defaults)


def _make_candidate(home="Chelsea", away="Burnley", date="2015-02-21",
                    game_id="england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
                    league="England Premier League", score="1 - 1"):
    return dict(home_team=home, away_team=away, date=date, game_id=game_id,
                league=league, score=score)


# ---------------------------------------------------------------------------
# _GameFinder._build_date_filter
# ---------------------------------------------------------------------------

class TestBuildDateFilter:
    def setup_method(self):
        llm = MagicMock()
        collection = MagicMock()
        self.finder = _GameFinder(llm, collection)

    def test_both_year_and_month(self):
        info = _make_match_info(year="2015", month="02")
        result = self.finder._build_date_filter(info)
        assert result == {"$regex": "^2015-02"}

    def test_year_only(self):
        info = _make_match_info(year="2015")
        result = self.finder._build_date_filter(info)
        assert result == {"$regex": "^2015"}

    def test_month_only(self):
        info = _make_match_info(month="3")
        result = self.finder._build_date_filter(info)
        assert result == {"$regex": r"^\d{4}-03"}

    def test_neither(self):
        info = _make_match_info()
        result = self.finder._build_date_filter(info)
        assert result is None

    def test_month_zero_padded(self):
        info = _make_match_info(year="2020", month="7")
        result = self.finder._build_date_filter(info)
        assert result == {"$regex": "^2020-07"}


# ---------------------------------------------------------------------------
# _GameFinder._candidates  — filter construction
# ---------------------------------------------------------------------------

class TestCandidates:
    def setup_method(self):
        self.collection = MagicMock()
        self.collection.find.return_value.limit.return_value = []
        self.finder = _GameFinder(MagicMock(), self.collection)

    def _call(self, **kwargs):
        info = _make_match_info(**kwargs)
        self.finder._candidates(info)
        return self.collection.find.call_args[0][0]  # first positional arg = filter dict

    def test_no_filters_when_all_unknown(self):
        f = self._call()
        assert f == {}

    def test_league_filter(self):
        f = self._call(league="England Premier League")
        assert f["league"] == "England Premier League"

    def test_two_teams_creates_or(self):
        f = self._call(team1="Chelsea", team2="Burnley")
        assert "$or" in f
        assert len(f["$or"]) == 2

    def test_one_team_creates_or(self):
        f = self._call(team1="Chelsea")
        assert "$or" in f
        assert len(f["$or"]) == 2  # home OR away

    def test_year_month_combined_no_overwrite(self):
        f = self._call(year="2015", month="02")
        assert f["date"] == {"$regex": "^2015-02"}

    def test_year_month_not_two_separate_keys(self):
        f = self._call(year="2015", month="02")
        # Should be a single dict value, not overwritten by second assignment
        assert "$regex" in f.get("date", {})
        assert f["date"]["$regex"].startswith("^2015-02")


# ---------------------------------------------------------------------------
# _GameFinder._select
# ---------------------------------------------------------------------------

class TestSelect:
    def setup_method(self):
        self.llm = MagicMock()
        self.finder = _GameFinder(self.llm, MagicMock())

    def test_no_candidates_returns_not_found(self):
        info = _make_match_info()
        result = self.finder._select([], info, "query")
        assert isinstance(result, _GameNotFound)

    def test_single_candidate_returns_found(self):
        info = _make_match_info()
        candidates = [_make_candidate()]
        result = self.finder._select(candidates, info, "query")
        assert isinstance(result, _GameFound)
        assert "england_epl" in result.game_id

    def test_multiple_candidates_llm_confident(self):
        from app.soccer_agent.toolbox.game_retrieval import _FinalResult
        mock_response = _FinalResult(
            game_id="england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
            response_llm="Found the match.",
        )
        self.llm.with_structured_output.return_value.__or__ = MagicMock()
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response
        self.llm.with_structured_output.return_value.__ror__ = MagicMock(return_value=chain_mock)

        # Patch the prompt template
        with patch(
            "app.soccer_agent.toolbox.game_retrieval.get_match_selection_prompt_template"
        ) as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.return_value.__or__ = MagicMock(return_value=mock_chain)

            candidates = [_make_candidate(), _make_candidate(home="Arsenal", away="Chelsea", game_id="x/y")]
            info = _make_match_info(team1="Chelsea", team2="Burnley")
            result = self.finder._select(candidates, info, "query")

        assert isinstance(result, _GameFound)
        assert result.game_id == "england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

    def test_multiple_candidates_llm_uncertain(self):
        from app.soccer_agent.toolbox.game_retrieval import _FinalResult
        mock_response = _FinalResult(game_id=None, response_llm="Not sure which match.")

        with patch(
            "app.soccer_agent.toolbox.game_retrieval.get_match_selection_prompt_template"
        ) as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_prompt.return_value.__or__ = MagicMock(return_value=mock_chain)

            candidates = [_make_candidate(), _make_candidate(home="Arsenal", away="Chelsea", game_id="x/y")]
            info = _make_match_info()
            result = self.finder._select(candidates, info, "query")

        assert isinstance(result, _GameNotFound)

    def test_find_wraps_exception_in_search_error(self):
        with patch.object(self.finder, "_extract", side_effect=RuntimeError("LLM down")):
            result = self.finder.find("some query")
        assert isinstance(result, _GameSearchError)
        assert "LLM down" in result.detail


# ---------------------------------------------------------------------------
# GameHistoryRetrievalTool._resolve_history_context
# ---------------------------------------------------------------------------

class TestResolveHistoryContext:
    def setup_method(self):
        from app.soccer_agent.toolbox.game_retrieval import GameHistoryRetrievalTool
        with patch("app.soccer_agent.toolbox.game_retrieval._get_collection", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_llm", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval._GameFinder"):
            self.tool = GameHistoryRetrievalTool()
        # Override private attrs directly after construction
        object.__setattr__(self.tool, "_llm", MagicMock())
        object.__setattr__(self.tool, "_collection", MagicMock())
        object.__setattr__(self.tool, "_finder", MagicMock())

    def test_list_annotation_artifact_used_directly(self):
        annotations = [Annotation(description="Goal", label="goal", gameTime="1 - 45")]
        with patch.object(self.tool, "_history_from_game_id") as mock_db:
            context, game_id = self.tool._resolve_history_context("q", annotations)
        mock_db.assert_not_called()
        assert game_id is None
        parsed = json.loads(context)
        assert parsed[0]["label"] == "goal"

    def test_valid_game_id_string_queries_db(self):
        game_id = "england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
        with patch.object(self.tool, "_history_from_game_id", return_value='["event"]') as mock_db:
            context, returned_id = self.tool._resolve_history_context("q", game_id)
        mock_db.assert_called_once_with(game_id)
        assert returned_id == game_id

    def test_string_without_slash_triggers_search(self):
        self.tool._finder.find.return_value = _GameFound(
            message="found",
            game_id="england_epl_2014-2015/match",
        )
        with patch.object(self.tool, "_history_from_game_id", return_value='[]') as mock_db:
            context, game_id = self.tool._resolve_history_context("q", "notavalidid")
        self.tool._finder.find.assert_called_once()
        assert game_id == "england_epl_2014-2015/match"

    def test_none_artifact_triggers_search(self):
        self.tool._finder.find.return_value = _GameFound(
            message="found",
            game_id="england_epl_2014-2015/match",
        )
        with patch.object(self.tool, "_history_from_game_id", return_value='[]'):
            _, game_id = self.tool._resolve_history_context("q", None)
        assert game_id == "england_epl_2014-2015/match"

    def test_search_error_raises(self):
        self.tool._finder.find.return_value = _GameSearchError(detail="timeout")
        with pytest.raises(RuntimeError, match="timeout"):
            self.tool._resolve_history_context("q", None)

    def test_not_found_raises_value_error(self):
        self.tool._finder.find.return_value = _GameNotFound(reason="no match")
        with pytest.raises(ValueError, match="no match"):
            self.tool._resolve_history_context("q", None)


# ---------------------------------------------------------------------------
# _GameFinder._candidates — remaining branches
# ---------------------------------------------------------------------------

class TestCandidatesExtraBranches:
    def setup_method(self):
        self.collection = MagicMock()
        self.collection.find.return_value.limit.return_value = []
        self.finder = _GameFinder(MagicMock(), self.collection)

    def _call(self, **kwargs):
        info = _make_match_info(**kwargs)
        self.finder._candidates(info)
        return self.collection.find.call_args[0][0]

    def test_season_filter(self):
        f = self._call(season="2015-2016")
        assert f["season"] == "2015-2016"

    def test_only_team2_creates_or(self):
        f = self._call(team2="Burnley")
        assert "$or" in f
        assert len(f["$or"]) == 2  # home OR away


# ---------------------------------------------------------------------------
# GameHistoryRetrievalTool._history_from_game_id
# ---------------------------------------------------------------------------

class TestHistoryFromGameId:
    def setup_method(self):
        from app.soccer_agent.toolbox.game_retrieval import GameHistoryRetrievalTool
        with patch("app.soccer_agent.toolbox.game_retrieval._get_collection", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_llm", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval._GameFinder"):
            self.tool = GameHistoryRetrievalTool()
        self.collection = MagicMock()
        object.__setattr__(self.tool, "_collection", self.collection)

    def test_doc_not_found_raises_runtime_error(self):
        self.collection.find_one.return_value = None
        with pytest.raises(RuntimeError, match="Game not found"):
            self.tool._history_from_game_id("some/game_id")

    def test_annotations_branch_parsed(self):
        self.collection.find_one.return_value = {"raw": {
            "annotations": [
                {"description": "Goal by Hazard", "label": "goal",
                 "contrastive_aligned_gameTime": "1 - 45", "gameTime": "1 - 44"},
            ]
        }}
        result = json.loads(self.tool._history_from_game_id("x/y"))
        assert result[0]["label"] == "goal"
        assert result[0]["description"] == "Goal by Hazard"
        assert result[0]["gameTime"] == "1 - 45"  # contrastive_aligned preferred

    def test_annotations_fallback_to_gameTime(self):
        self.collection.find_one.return_value = {"raw": {
            "annotations": [
                {"description": "Corner", "label": "corner",
                 "contrastive_aligned_gameTime": "", "gameTime": "1 - 30"},
            ]
        }}
        result = json.loads(self.tool._history_from_game_id("x/y"))
        assert result[0]["gameTime"] == "1 - 30"

    def test_comments_branch_parsed(self):
        self.collection.find_one.return_value = {"raw": {
            "comments": [
                {"comments_text": "Penalty awarded", "comments_type": "penalty",
                 "half": "1", "time_stamp": "38"},
            ]
        }}
        result = json.loads(self.tool._history_from_game_id("x/y"))
        assert result[0]["label"] == "penalty"
        assert result[0]["description"] == "Penalty awarded"
        assert "1" in result[0]["gameTime"]

    def test_neither_key_raises_value_error(self):
        self.collection.find_one.return_value = {"raw": {"something_else": []}}
        with pytest.raises(ValueError, match="annotations.*comments"):
            self.tool._history_from_game_id("x/y")

    def test_empty_annotations_raises_value_error(self):
        self.collection.find_one.return_value = {"raw": {"annotations": []}}
        with pytest.raises(ValueError, match="no events"):
            self.tool._history_from_game_id("x/y")


# ---------------------------------------------------------------------------
# GameInfoRetrievalTool._fetch_metadata
# ---------------------------------------------------------------------------

class TestFetchMetadata:
    def setup_method(self):
        from app.soccer_agent.toolbox.game_retrieval import GameInfoRetrievalTool
        with patch("app.soccer_agent.toolbox.game_retrieval._get_collection", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_llm", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval._GameFinder"):
            self.tool = GameInfoRetrievalTool()
        self.collection = MagicMock()
        object.__setattr__(self.tool, "_collection", self.collection)

    def test_doc_not_found_raises_runtime_error(self):
        self.collection.find_one.return_value = None
        with pytest.raises(RuntimeError, match="Game not found"):
            self.tool._fetch_metadata("some/game_id")

    def test_annotations_and_comments_stripped(self):
        self.collection.find_one.return_value = {"raw": {
            "home_team": "Chelsea", "away_team": "Burnley",
            "annotations": [{"label": "goal"}],
            "comments": [{"text": "kick off"}],
        }}
        result = json.loads(self.tool._fetch_metadata("x/y"))
        assert result["home_team"] == "Chelsea"
        assert "annotations" not in result
        assert "comments" not in result


# ---------------------------------------------------------------------------
# GameInfoRetrievalTool._run — dispatch branches
# ---------------------------------------------------------------------------

class TestGameInfoRetrievalRun:
    def setup_method(self):
        from app.soccer_agent.toolbox.game_retrieval import GameInfoRetrievalTool
        with patch("app.soccer_agent.toolbox.game_retrieval._get_collection", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_llm", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval._GameFinder"):
            self.tool = GameInfoRetrievalTool()
        object.__setattr__(self.tool, "_llm", MagicMock())
        object.__setattr__(self.tool, "_collection", MagicMock())
        object.__setattr__(self.tool, "_finder", MagicMock())

    def test_search_error_returns_error_string(self):
        self.tool._finder.find.return_value = _GameSearchError(detail="DB timeout")
        content, artifact = self.tool._run("query", execution_agent_state={})
        assert "DB timeout" in content
        assert artifact is None

    def test_not_found_returns_reason(self):
        self.tool._finder.find.return_value = _GameNotFound(reason="No match found, please provide more info.")
        content, artifact = self.tool._run("query", execution_agent_state={})
        assert "No match found" in content
        assert artifact is None

    def test_found_returns_answer_and_game_id(self):
        from app.soccer_agent.toolbox.game_retrieval import ToolOutput
        self.tool._finder.find.return_value = _GameFound(message="found", game_id="x/y")
        with patch.object(self.tool, "_fetch_metadata", return_value='{"home_team":"Chelsea"}'), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_game_info_retrieval_prompt_template") as mock_tmpl:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = ToolOutput(answer="Chelsea won 2-0.")
            mock_tmpl.return_value.__or__ = MagicMock(return_value=mock_chain)
            self.tool._llm.with_structured_output.return_value = MagicMock()
            content, artifact = self.tool._run("query", execution_agent_state={})
        assert content == "Chelsea won 2-0."
        assert artifact == "x/y"

    def test_exception_returns_error_string(self):
        self.tool._finder.find.side_effect = Exception("unexpected crash")
        content, artifact = self.tool._run("query", execution_agent_state={})
        assert "unexpected crash" in content
        assert artifact is None


# ---------------------------------------------------------------------------
# GameHistoryRetrievalTool._run — dispatch branches
# ---------------------------------------------------------------------------

class TestGameHistoryRetrievalRun:
    def setup_method(self):
        from app.soccer_agent.toolbox.game_retrieval import GameHistoryRetrievalTool
        with patch("app.soccer_agent.toolbox.game_retrieval._get_collection", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_llm", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval._GameFinder"):
            self.tool = GameHistoryRetrievalTool()
        object.__setattr__(self.tool, "_llm", MagicMock())
        object.__setattr__(self.tool, "_collection", MagicMock())
        object.__setattr__(self.tool, "_finder", MagicMock())

    def _run(self, last_artifact=None):
        return self.tool._run(
            query="query",
            execution_agent_state={"last_tool_artifact": last_artifact},
        )

    def test_normal_path_returns_answer_and_game_id(self):
        from app.soccer_agent.toolbox.game_retrieval import ToolOutput
        with patch.object(self.tool, "_resolve_history_context", return_value=('[]', "x/y")), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_game_history_retrieval_prompt_template") as mock_tmpl:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = ToolOutput(answer="Hazard scored in 45'.")
            mock_tmpl.return_value.__or__ = MagicMock(return_value=mock_chain)
            self.tool._llm.with_structured_output.return_value = MagicMock()
            content, artifact = self._run(last_artifact="x/y")
        assert content == "Hazard scored in 45'."
        assert artifact == "x/y"

    def test_exception_returns_error_string(self):
        with patch.object(self.tool, "_resolve_history_context", side_effect=ValueError("no events")):
            content, artifact = self._run()
        assert "no events" in content
        assert artifact is None


# ---------------------------------------------------------------------------
# about_current_game fast-path gating (renamed from about_current_match)
# ---------------------------------------------------------------------------

class TestAboutCurrentGameFastPath:
    def setup_method(self):
        from app.soccer_agent.toolbox.game_retrieval import GameInfoRetrievalTool, GameHistoryRetrievalTool
        with patch("app.soccer_agent.toolbox.game_retrieval._get_collection", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval.get_llm", return_value=MagicMock()), \
             patch("app.soccer_agent.toolbox.game_retrieval._GameFinder"):
            self.info_tool = GameInfoRetrievalTool()
            self.history_tool = GameHistoryRetrievalTool()
        # Only _finder is exercised by these tests; _llm/_collection are bypassed via patching.
        for tool in (self.info_tool, self.history_tool):
            object.__setattr__(tool, "_finder", MagicMock())

    def test_info_fast_path_skips_finder_when_flag_and_game_id(self):
        # Covers AE3 — flag true + active game_id resolves to the watched match without search.
        active_id = "europe_uefa-champions-league/2023-2024/2023-11-29/real-madrid-vs-napoli"
        with patch.object(self.info_tool, "_fetch_metadata", return_value='{"home_team":"Real Madrid"}'), \
             patch.object(self.info_tool, "_answer_from_context", return_value="Real Madrid 1 - 0 Napoli.") as mock_answer:
            content, artifact = self.info_tool._run(
                "what is the score",
                execution_agent_state={"additional_material": {"game_id": active_id}},
                about_current_game=True,
            )
        self.info_tool._finder.find.assert_not_called()
        assert content == "Real Madrid 1 - 0 Napoli."
        assert artifact == active_id
        mock_answer.assert_called_once()

    def test_history_fast_path_resolves_from_active_id_without_search(self):
        active_id = "england_epl/2014-2015/2015-02-21/chelsea-vs-burnley"
        with patch.object(self.history_tool, "_history_from_game_id", return_value='["event"]') as mock_db:
            context, game_id = self.history_tool._resolve_history_context(
                query="what just happened",
                last_artifact=None,
                active_game_id=active_id,
                about_current_game=True,
            )
        self.history_tool._finder.find.assert_not_called()
        mock_db.assert_called_once_with(active_id, active_vct=None)
        assert game_id == active_id

    def test_info_flag_true_but_no_game_id_falls_through_to_search(self):
        # No active video → fast path cannot fire even with the flag set; tool searches.
        self.info_tool._finder.find.return_value = _GameNotFound(reason="No match found.")
        content, artifact = self.info_tool._run(
            "what is the score",
            execution_agent_state={},
            about_current_game=True,
        )
        self.info_tool._finder.find.assert_called_once()
        assert "No match found" in content
        assert artifact is None

    def test_info_flag_false_with_active_game_id_uses_search_not_fast_path(self):
        # Flag false → fast path skipped; a different fixture from search is returned, not the active game.
        from app.soccer_agent.toolbox.game_retrieval import ToolOutput
        self.info_tool._finder.find.return_value = _GameFound(message="found", game_id="spain_laliga/other/match")
        with patch.object(self.info_tool, "_fetch_metadata", return_value='{"home_team":"Other"}'), \
             patch.object(self.info_tool, "_answer_from_context", return_value="Other match result."):
            content, artifact = self.info_tool._run(
                "Barcelona vs Sevilla 2019 result",
                execution_agent_state={"game_id": "europe_uefa-champions-league/2023-2024/2023-11-29/real-madrid-vs-napoli"},
                about_current_game=False,
            )
        self.info_tool._finder.find.assert_called_once()
        assert artifact == "spain_laliga/other/match"


class TestFlagRename:
    def test_input_schemas_expose_about_current_game_not_old_name(self):
        from app.soccer_agent.toolbox.game_retrieval import GameQueryInput, GameHistoryInput
        for model in (GameQueryInput, GameHistoryInput):
            assert "about_current_game" in model.model_fields
            assert "about_current_match" not in model.model_fields
