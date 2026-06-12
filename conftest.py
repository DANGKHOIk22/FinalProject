# Ensures the project root is on sys.path so tests can `import app...`.
# pytest inserts the directory containing the rootdir conftest into sys.path.
import os

# Dummy values for CI environments where real secrets are not configured.
# Only used during test collection (module-level init); real calls are mocked.
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-key")
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "4a2c9f8b1d7e6c3a5b0f8e9d2c1b3a4f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b",
)
