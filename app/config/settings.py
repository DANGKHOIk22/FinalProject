"""
Configuration settings loader for the soccer agent application.
Loads settings from environment variables with default fallbacks.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from app.config.config import PROJECT_PATH
load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""
    
    # MongoDB Configuration
    MONGO_SRV: Optional[str] = os.getenv('MONGO_SRV')
    SOCCER_DB_NAME: str = os.getenv('SOCCER_DB_NAME', 'SoccerWikiDemo')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    SOCCER_COLLECTION_NAME: str = os.getenv('SOCCER_COLLECTION_NAME', 'EntityInformation')
    QDRANT_URL: Optional[str] = os.getenv('QDRANT_URL')
    QDRANT_API_KEY: Optional[str] = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME: str = os.getenv('QDRANT_COLLECTION_NAME')
    DEEPFACE_HOME: Optional[str] = os.path.join(PROJECT_PATH, os.getenv('DEEPFACE_HOME', './temporary/cache'))

    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        if not cls.MONGO_SRV:
            raise ValueError("MONGO_SRV is required but not set")
        if not cls.SOCCER_DB_NAME:
            raise ValueError("SOCCER_DB_NAME is required but not set")
        if not cls.SOCCER_COLLECTION_NAME:
            raise ValueError("SOCCER_COLLECTION_NAME is required but not set")
        if not cls.QDRANT_URL:
            raise ValueError("QDRANT_URL is required but not set")
        if not cls.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY is required but not set")
        if not cls.QDRANT_COLLECTION_NAME:
            raise ValueError("QDRANT_COLLECTION_NAME is required but not set")
        return True


# Create a singleton instance
settings = Settings()


if settings.DEEPFACE_HOME:
    try:
        os.makedirs(os.path.join(settings.DEEPFACE_HOME, ".deepface", "weights"), exist_ok=True)
    except Exception:
        pass

