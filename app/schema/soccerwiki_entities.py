from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union


class PlayerSchema(BaseModel):
    """Schema for Player entity based on SoccerWiki data structure"""
    NAME: str
    ENTITY_TYPE: str = "player"
    PLAYER_URL: Optional[str] = None
    PLAYER_IMAGE_URL: Optional[str] = None
    INFOBOX: Optional[Any] = None
    CONTENT: Optional[Union[str, Dict]] = None
    IMAGES: Optional[List[str]] = Field(default_factory=list)
    SUMMARY: Optional[str] = None
    LAST_UPDATED: Optional[str] = None

class RefereeSchema(BaseModel):
    """Schema for Referee entity based on SoccerWiki data structure"""
    NAME: str
    ENTITY_TYPE: str = "referee"
    REFEREE_URL: Optional[str] = None
    REFEREE_IMAGE_URL: Optional[str] = None
    INFOBOX: Optional[Any] = None
    CONTENT: Optional[Union[str, Dict]] = None
    IMAGES: Optional[List[str]] = Field(default_factory=list)
    SUMMARY: Optional[str] = None
    LAST_UPDATED: Optional[str] = None

class VenueSchema(BaseModel):
    """Schema for Venue entity based on SoccerWiki data structure"""
    NAME: str
    ENTITY_TYPE: str = "venue"
    CITY: Optional[str] = None
    TEAM: Optional[str] = None
    CAPACITY: Optional[int] = None
    VENUE_URL: Optional[str] = None
    VENUE_IMAGE_URL: Optional[str] = None
    INFOBOX: Optional[Any] = None
    CONTENT: Optional[Union[str, Dict]] = None
    IMAGES: Optional[List[str]] = Field(default_factory=list)
    SUMMARY: Optional[str] = None
    LAST_UPDATED: Optional[str] = None

class TeamSchema(BaseModel):
    """Schema for Team entity based on SoccerWiki data structure"""
    NAME: str
    ENTITY_TYPE: str = "team"
    TEAM_URL: Optional[str] = None
    TEAM_IMAGE_URL: Optional[str] = None
    INFOBOX: Optional[Any] = None
    CONTENT: Optional[Union[str, Dict]] = None
    IMAGES: Optional[List[str]] = Field(default_factory=list)
    SUMMARY: Optional[str] = None
    LAST_UPDATED: Optional[str] = None

