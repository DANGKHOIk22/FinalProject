from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from app.schema.soccerwiki_entities import PlayerSchema, RefereeSchema, VenueSchema, TeamSchema

class SoccerEntities(BaseModel):
    """
    A schema for extracting soccer-related entities from a text query.
    """
    unknown: Optional[List[str]] = Field(default=None, description="List of entities that not sure about their type")
    player: Optional[List[str]] = Field(default=None, description="List of player names mentioned in the query")
    team: Optional[List[str]] = Field(default=None, description="List of team names mentioned in the query")
    venue: Optional[List[str]] = Field(default=None, description="List of venue names mentioned in the query")
    referee: Optional[List[str]] = Field(default=None, description="List of referee names mentioned in the query")

class SearchingResult(BaseModel):
    """
    Schema for searching entities information from database.
    _upsert_payload is populated by entity_augment when wiki content needs to be saved.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    found_entities: List[PlayerSchema | RefereeSchema | VenueSchema | TeamSchema] = Field(default_factory=list, description="Entities found in database\\other sources")
    missing_entities: List[str] = Field(default_factory=list, description="Entities not found in database\\other sources")
    _upsert_payload: Optional[Dict[str, Any]] = None