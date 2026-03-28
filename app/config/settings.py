"""
Configuration settings loader for the soccer agent application.
Loads settings from environment variables with default fallbacks.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from app.config.config import PROJECT_PATH
load_dotenv(override=True)

class Settings:
    """Application settings loaded from environment variables."""
    
    # MongoDB Configuration
    MONGO_SRV: Optional[str] = os.getenv('MONGO_SRV')
    SOCCER_DB_NAME: str = os.getenv('SOCCER_DB_NAME', 'SoccerWikiDemo')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    DASHSCOPE_API_KEY: Optional[str] = os.getenv('DASHSCOPE_API_KEY')
    SOCCER_COLLECTION_NAME: str = os.getenv('SOCCER_COLLECTION_NAME', 'EntityInformation')
    
    # Qdrant Configuration
    QDRANT_URL: Optional[str] = os.getenv('QDRANT_URL')
    QDRANT_API_KEY: Optional[str] = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME: Optional[str] = os.getenv('QDRANT_COLLECTION_NAME')

    # Postgres Configuration
    POSTGRES_DATABASE_URL: Optional[str] = os.getenv('POSTGRES_DATABASE_URL')
    DEEPFACE_HOME: Optional[str] = os.path.join(PROJECT_PATH, os.getenv('DEEPFACE_HOME', './temporary/cache'))
    
    # Endpoint Configuration
    DEEPFACE_ENDPOINT_URI: Optional[str] = os.getenv('DEEPFACE_ENDPOINT_URI') 
    DEEPFACE_ENDPOINT_KEY: Optional[str] = os.getenv('DEEPFACE_ENDPOINT_KEY')
    GROUNDINGDINO_ENDPOINT_URI: Optional[str] = os.getenv('GROUNDINGDINO_ENDPOINT_URI')
    GROUNDINGDINO_ENDPOINT_KEY: Optional[str] = os.getenv('GROUNDINGDINO_ENDPOINT_KEY')
    CLIP_ENDPOINT_URI: Optional[str] = os.getenv('CLIP_ENDPOINT_URI')
    CLIP_ENDPOINT_KEY: Optional[str] = os.getenv('CLIP_ENDPOINT_KEY')
    CLIP_GROUNDINGDINO_ENDPOINT_URI: Optional[str] = os.getenv('CLIP_GROUNDINGDINO_ENDPOINT_URI')
    CLIP_GROUNDINGDINO_ENDPOINT_KEY: Optional[str] = os.getenv('CLIP_GROUNDINGDINO_ENDPOINT_KEY')
    

    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        if not cls.MONGO_SRV:
            raise ValueError("MONGO_SRV is required but not set")
        if not cls.SOCCER_DB_NAME:
            raise ValueError("SOCCER_DB_NAME is required but not set")
        if not cls.SOCCER_COLLECTION_NAME:
            raise ValueError("SOCCER_COLLECTION_NAME is required but not set")
        
        # Validate Qdrant settings
        if not cls.QDRANT_URL:
            raise ValueError("QDRANT_URL is required but not set")
        if not cls.QDRANT_API_KEY:
            raise ValueError("QDRANT_API_KEY is required but not set")
        if not cls.QDRANT_COLLECTION_NAME:
            raise ValueError("QDRANT_COLLECTION_NAME is required but not set")
        
        # Validate endpoint settings
        if not cls.DEEPFACE_ENDPOINT_URI:
            raise ValueError("DEEPFACE_ENDPOINT_URI is required but not set")
        if not cls.DEEPFACE_ENDPOINT_KEY:
            raise ValueError("DEEPFACE_ENDPOINT_KEY is required but not set")
        if not cls.GROUNDINGDINO_ENDPOINT_URI:
            raise ValueError("GROUNDINGDINO_ENDPOINT_URI is required but not set")
        if not cls.GROUNDINGDINO_ENDPOINT_KEY:
            raise ValueError("GROUNDINGDINO_ENDPOINT_KEY is required but not set")
        if not cls.CLIP_ENDPOINT_URI:
            raise ValueError("CLIP_ENDPOINT_URI is required but not set")
        if not cls.CLIP_ENDPOINT_KEY:
            raise ValueError("CLIP_ENDPOINT_KEY is required but not set")
        return True


# Create a singleton instance
settings = Settings()


