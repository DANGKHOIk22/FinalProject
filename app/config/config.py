import os

# Project root
PROJECT_PATH = os.getcwd()  # Đường dẫn tuyệt đối đến thư mục dự án

# Temporary / artifact folders
TEMPORARY_DIR = os.path.join(PROJECT_PATH, "temporary")

# Model Configuration
# Canonical model name constants
GEMINI_2_5_FLASH = "gemini-2.5-flash"

# Backwards-compatible default model
DEFAULT_MODEL = GEMINI_2_5_FLASH
MODEL_TEMPERATURE = 0.6
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




