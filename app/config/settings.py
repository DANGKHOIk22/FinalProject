"""
Configuration settings loader for the soccer agent application.
Loads settings from environment variables with default fallbacks.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""
    
    # MongoDB Configuration
    MONGO_SRV: Optional[str] = os.getenv('MONGO_SRV')
    SOCCER_DB_NAME: str = os.getenv('SOCCER_DB_NAME', 'SoccerWikiDemo')
    SOCCER_COLLECTION_NAME: str = os.getenv('SOCCER_COLLECTION_NAME', 'EntityInformation')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        if not cls.MONGO_SRV:
            raise ValueError("MONGO_SRV is required but not set")
        if not cls.SOCCER_DB_NAME:
            raise ValueError("SOCCER_DB_NAME is required but not set")
        if not cls.SOCCER_COLLECTION_NAME:
            raise ValueError("SOCCER_COLLECTION_NAME is required but not set")
        return True


# Create a singleton instance
settings = Settings()
