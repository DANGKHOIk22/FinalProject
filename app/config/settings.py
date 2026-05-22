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
    GAME_COLLECTION_NAME: str = os.getenv('GAME_COLLECTION_NAME', 'games')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    DASHSCOPE_API_KEY: Optional[str] = os.getenv('DASHSCOPE_API_KEY')
    SOCCER_COLLECTION_NAME: str = os.getenv('SOCCER_COLLECTION_NAME', 'EntityInformation')
    
    # Qdrant Configuration
    QDRANT_URL: Optional[str] = os.getenv('QDRANT_URL')
    QDRANT_API_KEY: Optional[str] = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME: Optional[str] = os.getenv('QDRANT_COLLECTION_NAME')
    QDRANT_CASE_BANK_COLLECTION_NAME: str = os.getenv('QDRANT_CASE_BANK_COLLECTION_NAME', 'planning_case_bank')

    # Postgres Configuration
    POSTGRES_DATABASE_URL: Optional[str] = os.getenv('POSTGRES_DATABASE_URL')
    DEEPFACE_HOME: Optional[str] = os.path.join(PROJECT_PATH, os.getenv('DEEPFACE_HOME', './temporary/cache'))
    
    # Endpoint Configuration
    INSIGHTFACE_ENDPOINT_URI: Optional[str] = os.getenv('INSIGHTFACE_ENDPOINT_URI')
    INSIGHTFACE_ENDPOINT_KEY: Optional[str] = os.getenv('INSIGHTFACE_ENDPOINT_KEY')

    # Redis Configuration
    REDIS_URL: Optional[str] = os.getenv('REDIS_URL')

    # JWT Configuration
    JWT_SECRET_KEY: Optional[str] = os.getenv('JWT_SECRET_KEY')
    JWT_ALGORITHM: str = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', 10080)) # 60 * 24 * 7 =7 days


    # Tavily Configuration — comma-separated keys for round-robin + failover.
    # Single TAVILY_API_KEY is also accepted for backwards compatibility.
    TAVILY_API_KEYS: list[str] = [
        k.strip()
        for k in (os.getenv('TAVILY_API_KEYS') or os.getenv('TAVILY_API_KEY') or '').split(',')
        if k.strip()
    ]


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
        if not cls.INSIGHTFACE_ENDPOINT_URI:
            raise ValueError("INSIGHTFACE_ENDPOINT_URI is required but not set")
        if not cls.INSIGHTFACE_ENDPOINT_KEY:
            raise ValueError("INSIGHTFACE_ENDPOINT_KEY is required but not set")
        

        #Validate Redis settings
        if not cls.REDIS_URL:
            raise ValueError("REDIS_URL is required but not set")
            
        # Validate JWT settings
        if not cls.JWT_SECRET_KEY:
            raise ValueError("The JWT_SECRET_KEY is not added to the .env file, you should add it with a 64-character hex string to .env. For example: 4a2c9f8b1d7e6c3a5b0f8e9d2c1b3a4f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b")
            
        return True
    
    


# Create a singleton instance
settings = Settings()
settings.validate()


