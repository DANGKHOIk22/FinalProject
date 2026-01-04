import argparse
import logging
import os
import sys
from pathlib import Path
from app.config import settings

logging.basicConfig(level=logging.INFO)


def _ensure_repo_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _print_missing_env(missing: list[str]) -> None:
    print("Missing required environment variables:")
    for key in missing:
        print(f"- {key}")
    print("\nTip: set them in your .env (repo root) or in the shell environment.")


def main() -> int:
    _ensure_repo_on_syspath()

    parser = argparse.ArgumentParser(description="Quick test runner for FrameSelectionTool")
    parser.add_argument("--video", required=True, help="Path to a local video file to run frame selection on")
    parser.add_argument("--query", default="a key soccer moment", help="Text query describing the desired frame")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 2

    required_env = [
        "CLIP_ENDPOINT_URI",
        "CLIP_ENDPOINT_KEY",
    ]
    missing = [key for key in required_env if not os.getenv(key)]
    if missing:
        _print_missing_env(missing)
        return 2

    # Import after sys.path fix
    from app.toolbox.frame_selection import FrameSelectionTool

    print("Running FrameSelectionTool on:", str(video_path))
    print("Query:", args.query)

    try:
        tool = FrameSelectionTool()
        message, result_path = tool._run(query=args.query, material=[str(video_path)])
    except Exception as exc:
        print("Tool failed:")
        print(str(exc))
        return 1

    print("\n=== Message ===")
    print(message)

    print("\n=== Selected Frame ===")
    if result_path:
        print(result_path)
    else:
        print("No frame path returned (see logs for details)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
