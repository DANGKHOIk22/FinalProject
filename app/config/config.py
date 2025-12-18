import os
import torch

# Model Configuration
# Canonical model name constants
GEMINI_2_5_FLASH = "gemini-2.5-flash"

# Backwards-compatible default model
DEFAULT_MODEL = GEMINI_2_5_FLASH
MODEL_TEMPERATURE = 0.6
MODEL_TOP_P = 0.95
MAX_COMPLETION_TOKENS = 8000
MODEL_SEGMENT = "IDEA-Research/grounding-dino-base"
SEGMENT_IMAGE_FOLDER = "segmented_images"
# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = 'INFO'
PROJECT_PATH = os.getcwd()  # Đường dẫn tuyệt đối đến thư mục dự án
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




