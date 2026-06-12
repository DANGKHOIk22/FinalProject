import json
import pytest
from unittest.mock import MagicMock

from app.cache.semantic_cache import SemanticCache, _image_tag, _NO_VIDEO
from app.cache.standard_cache import StandardCache


# ---------------------------------------------------------------------------
# _image_tag — stable tag for the image_id filter
# ---------------------------------------------------------------------------

def test_image_tag_none_and_empty():
    assert _image_tag(None) == "None"
    assert _image_tag([]) == "None"

def test_image_tag_order_insensitive():
    assert _image_tag(["a", "b"]) == _image_tag(["b", "a"])

def test_image_tag_safe_with_commas():
    """Tag values must never contain commas (RediSearch splits tags on them)."""
    tag = _image_tag(["uuid-1", "uuid-2"])
    assert "," not in tag

def test_image_tag_differs_for_different_images():
    assert _image_tag(["a"]) != _image_tag(["b"])


# ---------------------------------------------------------------------------
# SemanticCache — video vs non-video paths share one consistent tag scheme
# ---------------------------------------------------------------------------

@pytest.fixture
def semantic_cache_instance():
    cache = SemanticCache.__new__(SemanticCache)
    cache.threshold = 0.15
    cache.is_active = True
    cache.vector_store = MagicMock()
    return cache

def test_set_non_video_uses_hashed_image_tag(semantic_cache_instance):
    material = {"image_id": ["b", "a"], "game_id": None}
    semantic_cache_instance.set("query", "answer", material)

    metadata = semantic_cache_instance.vector_store.add_texts.call_args.kwargs["metadatas"][0]
    assert metadata["image_id"] == _image_tag(["a", "b"])
    assert metadata["game_id"] == _NO_VIDEO
    assert metadata["response"] == "answer"

def test_set_video_path_stores_game_id_and_timestamp(semantic_cache_instance):
    material = {"game_id": "g1", "image_id": []}
    semantic_cache_instance.set("query", "answer", material, current_time=42.0)

    metadata = semantic_cache_instance.vector_store.add_texts.call_args.kwargs["metadatas"][0]
    assert metadata["game_id"] == "g1"
    assert metadata["timestamp"] == 42.0

def test_check_hit_returns_response(semantic_cache_instance):
    doc = MagicMock()
    doc.metadata = {"response": "cached answer"}
    semantic_cache_instance.vector_store.similarity_search_with_score.return_value = [(doc, 0.05)]

    assert semantic_cache_instance.check("query", {"image_id": ["a"]}) == "cached answer"

def test_check_miss_returns_none(semantic_cache_instance):
    semantic_cache_instance.vector_store.similarity_search_with_score.return_value = []
    assert semantic_cache_instance.check("query") is None

def test_inactive_cache_is_noop(semantic_cache_instance):
    semantic_cache_instance.is_active = False
    assert semantic_cache_instance.check("query") is None
    semantic_cache_instance.set("query", "answer")
    semantic_cache_instance.vector_store.add_texts.assert_not_called()

def test_set_skips_empty_response(semantic_cache_instance):
    semantic_cache_instance.set("query", "")
    semantic_cache_instance.vector_store.add_texts.assert_not_called()


# ---------------------------------------------------------------------------
# StandardCache — atomic TTL + decorator behavior (sync and async)
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_cache_instance():
    cache = StandardCache.__new__(StandardCache)
    cache.client = MagicMock()
    return cache

def test_set_key_atomic_ttl(standard_cache_instance):
    """set_key must apply TTL in a single atomic SET command."""
    standard_cache_instance.set_key("k", "v", ttl=120)

    standard_cache_instance.client.set.assert_called_once_with("k", "v", ex=120)
    standard_cache_instance.client.expire.assert_not_called()

@pytest.mark.asyncio
async def test_async_decorator_miss_executes_and_stores(standard_cache_instance):
    standard_cache_instance.client.get.return_value = None
    calls = []

    @standard_cache_instance.cache(ttl=60)
    async def my_func(x: int):
        calls.append(x)
        return {"v": x}

    result = await my_func(5)

    assert result == {"v": 5}
    assert calls == [5]
    standard_cache_instance.client.set.assert_called_once()
    assert standard_cache_instance.client.set.call_args.kwargs.get("ex") == 60

@pytest.mark.asyncio
async def test_async_decorator_hit_skips_execution(standard_cache_instance):
    standard_cache_instance.client.get.return_value = json.dumps({"v": 5}).encode()
    calls = []

    @standard_cache_instance.cache(ttl=60)
    async def my_func(x: int):
        calls.append(x)
        return {"v": x}

    result = await my_func(5)

    assert result == {"v": 5}
    assert calls == []  # function body never ran
    standard_cache_instance.client.set.assert_not_called()

def test_sync_decorator_miss_stores_result(standard_cache_instance):
    """Regression: the sync wrapper must write results back to the cache."""
    standard_cache_instance.client.get.return_value = None

    @standard_cache_instance.cache(ttl=60)
    def my_func(x: int):
        return {"v": x}

    assert my_func(5) == {"v": 5}
    standard_cache_instance.client.set.assert_called_once()

@pytest.mark.asyncio
async def test_async_decorator_redis_down_degrades_gracefully(standard_cache_instance):
    standard_cache_instance.client.get.side_effect = ConnectionError("redis down")

    @standard_cache_instance.cache(ttl=60)
    async def my_func(x: int):
        return {"v": x}

    assert await my_func(7) == {"v": 7}
