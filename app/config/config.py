import os

# Project root
PROJECT_PATH = os.getcwd()  # Đường dẫn tuyệt đối đến thư mục dự án

# Temporary / artifact folders
TEMPORARY_DIR = os.path.join(PROJECT_PATH, "temporary")

# Model Configuration
GEMINI_GEMMA_4_31B = "gemma-4-31b-it"
GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_3_1_FLASH = "gemini-3.1-flash-preview"
GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite-preview"

# Backwards-compatible default model
DEFAULT_MODEL = GEMINI_3_1_FLASH_LITE
MODEL_TEMPERATURE = 1.0
MODEL_TOP_P = 0.95
MAX_COMPLETION_TOKENS = 8000

# Grounding DINO model for image segmentation
MODEL_SEGMENT = "IDEA-Research/grounding-dino-base"
SEGMENT_IMAGE_FOLDER = os.path.join(TEMPORARY_DIR, "segmented_images")

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




