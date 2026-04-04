import os

# Project root
PROJECT_PATH = os.getcwd()  # Đường dẫn tuyệt đối đến thư mục dự án

# Temporary / artifact folders
TEMPORARY_DIR = os.path.join(PROJECT_PATH, "temporary")

# Model Configuration
# Canonical model name constants
GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_2_0_FLASH_LITE = GEMINI_2_5_FLASH_LITE
# Backwards-compatible default model
DEFAULT_MODEL = GEMINI_2_5_FLASH
MODEL_TEMPERATURE = 0.2
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




