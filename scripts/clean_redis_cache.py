"""
Clean all Redis caches (Semantic Cache + Standard Cache).
Usage:
    uv run scripts/clean_redis_cache.py              # clean all
    uv run scripts/clean_redis_cache.py --semantic    # semantic cache only
    uv run scripts/clean_redis_cache.py --standard    # standard cache only
"""
import os
import argparse
import logging
import redis
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CleanRedisCache")


def clean_standard_cache(client: redis.Redis):
    keys = client.keys("soccer_agent:cache:*")
    count = len(keys)
    if keys:
        client.delete(*keys)
    logger.info(f"Standard Cache: deleted {count} keys")


def clean_semantic_cache(client: redis.Redis):
    # 1. Delete RediSearch index if exists
    try:
        from redisvl.index import SearchIndex
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        index = SearchIndex.from_existing("semantic_cache", redis_url=redis_url)
        index.delete(drop=True)
        logger.info("Semantic Cache: RediSearch index dropped")
    except Exception as e:
        logger.info(f"Semantic Cache: no RediSearch index to drop ({e})")

    # 2. Delete all semantic cache keys (including legacy versioned prefixes)
    total = 0
    for pattern in ["semantic_cache:*", "soccer_agent_semantic_cache*"]:
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            total += len(keys)
    logger.info(f"Semantic Cache: deleted {total} keys")


def main():
    parser = argparse.ArgumentParser(description="Clean Redis caches")
    parser.add_argument("--semantic", action="store_true", help="Clean semantic cache only")
    parser.add_argument("--standard", action="store_true", help="Clean standard cache only")
    args = parser.parse_args()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    client = redis.Redis.from_url(redis_url)

    # Show current state
    all_keys = client.keys("*")
    logger.info(f"Total Redis keys: {len(all_keys)}")

    clean_all = not args.semantic and not args.standard

    if clean_all or args.standard:
        clean_standard_cache(client)

    if clean_all or args.semantic:
        clean_semantic_cache(client)

    remaining = client.keys("*")
    logger.info(f"✅ Done. Remaining Redis keys: {len(remaining)}")


if __name__ == "__main__":
    main()
