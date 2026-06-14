import os
from pathlib import Path

# Project root — anchored to this file's location, not os.getcwd()
PROJECT_PATH = str(Path(__file__).parent.parent.parent)

# Temporary / artifact folders
TEMPORARY_DIR = os.path.join(PROJECT_PATH, "temporary")

# Model Configuration
GEMINI_GEMMA_4_31B = "gemma-4-31b-it"
GEMINI_3_1_FLASH = "gemini-3.1-flash-preview"
GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite-preview"

# Backwards-compatible default model
DEFAULT_MODEL = GEMINI_3_1_FLASH_LITE
MODEL_TEMPERATURE = 1.0
MODEL_TOP_P = 0.95
MAX_COMPLETION_TOKENS = 8000

# Qdrant Configuration
QDRANT_SEARCH_SCORE_THRESHOLD = 0.5

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = 'INFO'

# Session Memory
SESSION_MEMORY_TOKEN_THRESHOLD = 12600  # len(text) // 4 ≈ số token
SESSION_MEMORY_RECENT_KEEP = 5         # tin nhắn giữ verbatim sau khi vượt ngưỡng
SESSION_MEMORY_REDIS_TTL = 86400       # TTL Redis key (giây) = 24 giờ

# Planning — per-chain confidence gate
PLANNING_CONFIDENCE_THRESHOLD: float = 0.5

# Game tools — Tavily fallback result count when DB has no match
GAME_FALLBACK_TOP_K: int = 10

# entity_augment — entity freshness gate
# If a found entity's LAST_UPDATED is within this many days of the query time,
# trust the DB answer and skip the Tavily fallback entirely.
ENTITY_FRESHNESS_DAYS: int = 5

# Soccer-topic guardrail
GUARDRAIL_RECENT_TURNS: int = 4          # trailing messages the classifier sees for context
GUARDRAIL_TIMEOUT_SECONDS: float = 10.0  # fail-open ceiling; primary (gemini flash-lite) is ~1s, headroom covers failover to the gpt-4o-mini backup
GUARDRAIL_REFUSAL_MESSAGE: str = (
    "Xin lỗi, tôi là trợ lý chuyên về bóng đá nên chỉ có thể trả lời các câu hỏi "
    "liên quan đến bóng đá (cầu thủ, đội bóng, huấn luyện viên, trận đấu, giải đấu...). "
    "Bạn vui lòng đặt câu hỏi về bóng đá nhé!"
)




