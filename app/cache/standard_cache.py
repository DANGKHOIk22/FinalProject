import json
import logging
import asyncio
from functools import wraps
from typing import Any
from uuid import UUID

import redis

from app.config.settings import settings

logger = logging.getLogger(__name__)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return super().default(obj)


class StandardCache:
    def __init__(self):
        # We can use redis.from_url to safely connect
        redis_url = settings.REDIS_URL or "redis://localhost:6379"
        self.client = redis.Redis.from_url(redis_url)

    def _cache_logic(self, func, args, kwargs, ttl, validatedModel, is_async=False):
        """Shared cache logic for both sync and async functions"""
        module_name = func.__module__
        func_name = func.__qualname__

        args_to_serialize = args
        # Attempt to ignore 'self' or 'cls' for method calls
        if args and "." in func.__qualname__:
            class_name = func.__qualname__.split(".")[0]
            if getattr(args[0], '__class__', None) and args[0].__class__.__name__ == class_name:
                args_to_serialize = args[1:]
            elif getattr(args[0], '__name__', None) == class_name:
                args_to_serialize = args[1:]

        try:
            dumped_args = self.serialize(args_to_serialize)
            dumped_kwargs = self.serialize(kwargs)
        except Exception as e:
            logger.warning(f"Could not serialize args/kwargs for key generation: {e}")
            return None, None
        
        prefix = "soccer_agent:cache"
        key = f"{prefix}:{module_name}:{func_name}:{dumped_args}:{dumped_kwargs}"
        
        # Truncate long key for display
        display_key = (key[:100] + "...") if len(key) > 100 else key
        logger.debug(f"Checking Standard Cache for [{module_name}.{func_name}] with key: {display_key}")

        try:
            cached_result = self.client.get(key)
        except Exception as e:
            logger.warning(f"Redis not available for key: {key}, error: {e}")
            return None, None

        if cached_result:
            logger.info(f"🎯 Standard Cache HIT for [{module_name}.{func_name}] -> Skipping function execution.")
            data = self.deserialize(cached_result)
            if validatedModel:
                try:
                    data = validatedModel(**data)
                except Exception as e:
                    logger.warning(f"Failed to validate cached data: {e}. Considering it a miss.")
                    return None, None
            return "hit", data

        logger.debug(f"⏳ Standard Cache MISS for [{module_name}.{func_name}] -> Executing function.")
        return "miss", key

    def cache(self, *, ttl: int = 60 * 60, validatedModel: Any = None):
        """
        Decorator supports both sync and async functions
        - Automatically detects function type (sync/async)
        - Caches result in Redis with TTL
        """
        def inner(func):
            is_async = asyncio.iscoroutinefunction(func)

            if is_async:
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    cache_result, data = self._cache_logic(
                        func, args, kwargs, ttl, validatedModel, True
                    )

                    if cache_result is None:
                        return await func(*args, **kwargs)
                    elif cache_result == "hit":
                        return data
                    else:
                        result = await func(*args, **kwargs)
                        self._store_result(data, result, ttl, validatedModel)
                        return result

                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    cache_result, data = self._cache_logic(
                        func, args, kwargs, ttl, validatedModel, False
                    )

                    if cache_result is None:
                        return func(*args, **kwargs)
                    elif cache_result == "hit":
                        return data
                    else:
                        result = func(*args, **kwargs)
                        # self._store_result(data, result, ttl, validatedModel)
                        return result

                return sync_wrapper

        return inner

    def _store_result(self, key, result, ttl, validatedModel=None):
        """Store the result into the cache."""
        data_to_serialize = None
        # Check if result is a Pydantic model
        if hasattr(result, "model_dump"):
            data_to_serialize = result.model_dump()
        # Check if it's a list of Pydantic models
        elif isinstance(result, list) and result and hasattr(result[0], "model_dump"):
            data_to_serialize = [r.model_dump() for r in result]
        else:
            data_to_serialize = result

        try:
            serialized_result = self.serialize(data_to_serialize)
        except TypeError as e:
            logger.warning(f"Could not serialize result for key: {key}, error: {e}")
            return

        self.set_key(key, serialized_result, ttl)
        # Truncate long key for display
        display_key = (key[:100] + "...") if len(key) > 100 else key
        logger.debug(f"💾 Standard Cache STORED for key: {display_key}")

    def set_key(self, key: str, value: Any, ttl: int = 60 * 60):
        self.client.set(key, value)
        self.client.expire(key, ttl)

    def remove_key(self, key: str):
        self.client.delete(key)

    def serialize(self, value: Any) -> str:
        return json.dumps(value, cls=CustomEncoder, sort_keys=True)

    def deserialize(self, value: str) -> dict:
        return json.loads(value)

    def list_keys(self, pattern: str = "soccer_agent:cache:*") -> Any:
        return self.client.keys(pattern)


standard_cache = StandardCache()