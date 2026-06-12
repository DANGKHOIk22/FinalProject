import json
import pytest
from unittest.mock import MagicMock

from app.cache.semantic_cache import SubQuerySemanticCache, _material_tag
from app.cache.standard_cache import StandardCache


# ---------------------------------------------------------------------------
# _material_tag — stable tag for the material filter
# ---------------------------------------------------------------------------

def test_material_tag_none_and_empty():
    assert _material_tag(None) == "None"
    assert _material_tag([]) == "None"

def test_material_tag_order_insensitive():
    assert _material_tag(["a.png", "b.png"]) == _material_tag(["b.png", "a.png"])

def test_material_tag_safe_with_commas():
    """Media URLs with commas must not produce comma-containing tags
    (RediSearch splits tag values on commas)."""
    tag = _material_tag(["https://blob.host/img.png?sv=2024,sig=a,b"])
    assert "," not in tag

def test_material_tag_differs_for_different_materials():
    assert _material_tag(["a.png"]) != _material_tag(["b.png"])


# ---------------------------------------------------------------------------
# SubQuerySemanticCache — check/set use the same stable tag
# ---------------------------------------------------------------------------

@pytest.fixture
def semantic_cache_instance():
    cache = SubQuerySemanticCache.__new__(SubQuerySemanticCache)
    cache.threshold = 0.15
    cache.is_active = True
    cache.vector_store = MagicMock()
    return cache

def test_semantic_cache_set_uses_hashed_material(semantic_cache_instance):
    material = ["b.png", "a.png"]
    semantic_cache_instance.set("query", "answer", material)

    kwargs = semantic_cache_instance.vector_store.add_texts.call_args.kwargs
    assert kwargs["metadatas"][0]["material"] == _material_tag(material)
    assert kwargs["metadatas"][0]["response"] == "answer"

def test_semantic_cache_check_hit(semantic_cache_instance):
    doc = MagicMock()
    doc.metadata = {"response": "cached answer"}
    semantic_cache_instance.vector_store.similarity_search_with_score.return_value = [(doc, 0.05)]

    assert semantic_cache_instance.check("query", ["a.png"]) == "cached answer"

def test_semantic_cache_check_miss(semantic_cache_instance):
    semantic_cache_instance.vector_store.similarity_search_with_score.return_value = []
    assert semantic_cache_instance.check("query") is None

def test_semantic_cache_inactive_returns_none(semantic_cache_instance):
    semantic_cache_instance.is_active = False
    assert semantic_cache_instance.check("query") is None
    semantic_cache_instance.set("query", "answer")
    semantic_cache_instance.vector_store.add_texts.assert_not_called()

def test_semantic_cache_set_skips_empty_response(semantic_cache_instance):
    semantic_cache_instance.set("query", "")
    semantic_cache_instance.vector_store.add_texts.assert_not_called()


# ---------------------------------------------------------------------------
# StandardCache — atomic TTL + async decorator off the event loop
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
    set_args = standard_cache_instance.client.set.call_args
    assert set_args.kwargs.get("ex") == 60

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

@pytest.mark.asyncio
async def test_async_decorator_redis_down_degrades_gracefully(standard_cache_instance):
    standard_cache_instance.client.get.side_effect = ConnectionError("redis down")

    @standard_cache_instance.cache(ttl=60)
    async def my_func(x: int):
        return {"v": x}

    assert await my_func(7) == {"v": 7}
