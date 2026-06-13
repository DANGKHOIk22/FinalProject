"""
Migrate existing MongoDB game documents from old game_id format to new structured format.

Old: "england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
New: "england_epl/2014-2015/2015-02-21/chelsea-vs-burnley"
"""
import os
import re
from dotenv import load_dotenv
from pymongo import MongoClient
from dns import resolver
from tqdm import tqdm

load_dotenv()

MONGO_SRV = os.getenv("MONGO_SRV")
SOCCER_DB_NAME = os.getenv("SOCCER_DB_NAME", "SoccerWikiDemo")
COLLECTION_NAME = os.getenv("GAME_COLLECTION_NAME", "games")

assert MONGO_SRV, "MONGO_SRV not set — check .env"

resolver.default_resolver = resolver.Resolver(configure=False)
resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]

client = MongoClient(MONGO_SRV)
collection = client[SOCCER_DB_NAME][COLLECTION_NAME]


def slugify_team(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-")


def build_game_id(doc: dict) -> str:
    league = doc.get("league", "unknown")
    season = doc.get("season", "unknown")
    date = doc.get("date", "unknown")
    home = slugify_team(doc.get("home_team", "home"))
    away = slugify_team(doc.get("away_team", "away"))
    return f"{league}/{season}/{date}/{home}-vs-{away}"


def is_old_format(game_id: str) -> bool:
    """Old format contains spaces or has league+season merged with underscore."""
    return " " in game_id


def migrate():
    total = collection.count_documents({})
    print(f"Total documents: {total}")

    docs = list(collection.find({}, {"_id": 1, "game_id": 1, "league": 1, "season": 1, "date": 1, "home_team": 1, "away_team": 1}))
    to_migrate = [d for d in docs if is_old_format(d.get("game_id", ""))]
    print(f"Documents needing migration: {len(to_migrate)}")

    if not to_migrate:
        print("Nothing to migrate.")
        return

    confirm = input(f"Proceed with migrating {len(to_migrate)} documents? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    updated = 0
    skipped = 0
    errors = 0

    for doc in tqdm(to_migrate, desc="Migrating"):
        new_id = build_game_id(doc)
        try:
            # Check for collision
            existing = collection.find_one({"game_id": new_id}, {"_id": 1})
            if existing and existing["_id"] != doc["_id"]:
                skipped += 1
                continue
            collection.update_one({"_id": doc["_id"]}, {"$set": {"game_id": new_id}})
            updated += 1
        except Exception as e:
            errors += 1
            print(f"Error on {doc['_id']}: {e}")

    print(f"\n✅ Updated: {updated} | Skipped (collision): {skipped} | Errors: {errors}")
    print(f"Sample: {build_game_id(to_migrate[0])}")


if __name__ == "__main__":
    migrate()
