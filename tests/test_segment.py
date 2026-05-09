import argparse
import json
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def _ensure_repo_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _print_missing_env(missing: list[str]) -> None:
    print("Missing required environment variables:")
    for k in missing:
        print(f"- {k}")
    print("\nTip: set them in your .env (repo root) or in the shell environment.")


def main() -> int:
    _ensure_repo_on_syspath()

    parser = argparse.ArgumentParser(description="Quick test runner for SegmentTool")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a local image file (jpg/png) to run segmentation on",
    )
    parser.add_argument(
        "--query",
        required=True,
        nargs="+",
        help="Entity descriptions to segment (e.g., 'the player in red jersey' 'the referee in black')",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2

    # Import after sys.path fix
    from app.config import settings
    from app.toolbox.segment import SegmentTool

    required_env = [
        "GROUNDINGDINO_ENDPOINT_URI",
        "GROUNDINGDINO_ENDPOINT_KEY",
    ]
    missing = [k for k in required_env if not os.getenv(k)]
    if missing:
        _print_missing_env(missing)
        return 2

    print("Running SegmentTool on:", str(image_path))
    print("Query entities:", args.query)
    print()

    try:
        tool = SegmentTool()
        message, segmented_paths = tool._run(
            query_entity_recognition_task=args.query,
            material=[str(image_path)]
        )
    except Exception as e:
        print("Tool failed:")
        print(str(e))
        import traceback
        traceback.print_exc()
        return 1

    print("\n=== Message ===")
    print(message)

    print("\n=== Segmented Image Paths ===")
    if segmented_paths:
        for idx, path in enumerate(segmented_paths, 1):
            print(f"{idx}. {path}")
            if os.path.exists(path):
                print(f"   ✅ File exists")
            else:
                print(f"   ❌ File not found")
    else:
        print("No segmented images returned")

    # Summary
    print("\n=== Summary ===")
    print(f"Total segmented objects: {len(segmented_paths) if segmented_paths else 0}")
    print(f"Query entities: {len(args.query)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
