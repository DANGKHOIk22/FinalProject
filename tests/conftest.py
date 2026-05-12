import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set dummy API keys for CI environments where they are not configured.
# These are only used during test collection (module-level init of embeddings).
# Actual API calls are mocked in individual tests.
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-key")
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
