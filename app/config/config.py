import os

# Model Configuration
DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_TEMPERATURE = 0.6
MODEL_TOP_P = 0.95
MAX_COMPLETION_TOKENS = 8000
MODEL_SEGMENT = "IDEA-Research/grounding-dino-base"

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = 'INFO'
PROJECT_PATH = os.getcwd()  # Đường dẫn tuyệt đối đến thư mục dự án




