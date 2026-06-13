import argparse
import json
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def _ensure_repo_on_syspath() -> None:
    # Find repo root by locating pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:  # Stop at filesystem root
        if (current / "pyproject.toml").exists() or (current / "app").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return
        current = current.parent
    
    # Fallback: use parents[2] (two levels up from test file)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _pydantic_dump(obj):
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _print_missing_env(missing: list[str]) -> None:
    print("Missing required environment variables:")
    for k in missing:
        print(f"- {k}")
    print("\nTip: set them in your .env (repo root) or in the shell environment.")


def main() -> int:
    _ensure_repo_on_syspath()

    parser = argparse.ArgumentParser(description="Quick test runner for EntityRecognitionTool")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a local image file (jpg/png) to run entity recognition on",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON result",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2

    # Import after sys.path fix
    from app.config.settings import settings
    from app.soccer_agent.toolbox.entity_recognition import EntityRecognitionTool

    required_env = [
        "MONGO_SRV",
        "SOCCER_DB_NAME",
        "SOCCER_COLLECTION_NAME",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION_NAME",
        "INSIGHTFACE_ENDPOINT_URI",
        "INSIGHTFACE_ENDPOINT_KEY",
    ]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        _print_missing_env(missing)
        return 2

    print("Running EntityRecognitionTool on:", str(image_path))

    try:
        tool = EntityRecognitionTool()
        message, result = tool._run(material=[str(image_path)])
    except Exception as e:
        print("Tool failed:")
        print(str(e))
        return 1

    print("\n=== Message ===")
    print(message)

    # print("\n=== Result (SearchingResult) ===")
    # payload = _pydantic_dump(result)
    # if args.pretty:
    #     print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    # else:
    #     print(json.dumps(payload, ensure_ascii=False, default=str))

    # Convenience summary
    found = getattr(result, "found_entities", []) or []
    missing_entities = getattr(result, "missing_entities", []) or []

    print("\n=== Summary ===")
    print("Found:", len(found))
    for ent in found:
        name = getattr(ent, "NAME", None) or getattr(ent, "name", None) or str(ent)
        etype = getattr(ent, "ENTITY_TYPE", None) or getattr(ent, "entity_type", None) or "unknown"
        print(f"- {etype}: {name}")

    print("Missing:", len(missing_entities))
    for name in missing_entities:
        print(f"- {name}")

    # Optional: validate Settings (note: validate() also requires GroundingDino vars)
    # settings.Settings.validate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
