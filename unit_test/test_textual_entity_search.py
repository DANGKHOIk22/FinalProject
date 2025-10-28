import unittest
import sys
import os
from unittest.mock import patch
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from toolbox.textual_entity_search import (
    textual_entity_search,
    parse_entity_result,
    SoccerEntities,
    SearchingResult,
)

from models.soccerwiki_entities import (
    PlayerSchema,
    RefereeSchema,
    VenueSchema,
    TeamSchema
)


class TestTextualEntitySearch(unittest.TestCase):
    """Unit tests for textual_entity_search function and related components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        load_dotenv()
    
    def test_textual_entity_search_player_query(self):
        """Test searching for a player entity"""
        query = "What is the height of the goalkeeper Neuer?"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
        self.assertIsInstance(result.found_entities, list)
        self.assertIsInstance(result.missing_entities, list)
    
    def test_textual_entity_search_team_query(self):
        """Test searching for a team entity"""
        query = "Who is the current coach of Liverpool and when was he hired?"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
        self.assertIsInstance(result.found_entities, list)
        self.assertIsInstance(result.missing_entities, list)
    
    def test_textual_entity_search_venue_query(self):
        """Test searching for a venue entity"""
        query = "Who won the last match held at Old Trafford?"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
        self.assertIsInstance(result.found_entities, list)
        self.assertIsInstance(result.missing_entities, list)
    
    def test_textual_entity_search_multiple_entities(self):
        """Test searching with multiple entities"""
        query = "Did Madrid win the Champions League final?"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
        # Should have found or missing entities
        total_entities = len(result.found_entities) + len(result.missing_entities)
        self.assertGreaterEqual(total_entities, 0)
    
    def test_textual_entity_search_ambiguous_query(self):
        """Test searching with ambiguous entity"""
        query = "Tell me about Alexander?"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
        self.assertIsInstance(result.found_entities, list)
        self.assertIsInstance(result.missing_entities, list)
    
    def test_textual_entity_search_referee_query(self):
        """Test searching for referee entity"""
        query = "What is the full name of Pierluigi Collina?"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
        self.assertIsInstance(result.found_entities, list)
        self.assertIsInstance(result.missing_entities, list)
    
    def test_textual_entity_search_empty_query(self):
        """Test with empty query"""
        query = ""
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
    
    def test_textual_entity_search_no_entities(self):
        """Test with query containing no soccer entities"""
        query = "What is the weather today?"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
    
    def test_parse_entity_result_player(self):
        """Test parsing player entity data"""
        entity_data = {
            'NAME': 'Test Player',
            'ENTITY_TYPE': 'player',
            'PLAYER_URL': 'http://example.com/player',
            'SUMMARY': 'A test player'
        }
        
        result = parse_entity_result(entity_data)
        self.assertIsInstance(result, PlayerSchema)
        self.assertEqual(result.NAME, 'Test Player')
        self.assertEqual(result.ENTITY_TYPE, 'player')
    
    def test_parse_entity_result_team(self):
        """Test parsing team entity data"""
        entity_data = {
            'NAME': 'Test Team',
            'ENTITY_TYPE': 'team',
            'TEAM_URL': 'http://example.com/team',
        }
        
        result = parse_entity_result(entity_data)
        self.assertIsInstance(result, TeamSchema)
        self.assertEqual(result.NAME, 'Test Team')
        self.assertEqual(result.ENTITY_TYPE, 'team')
    
    def test_parse_entity_result_venue(self):
        """Test parsing venue entity data"""
        entity_data = {
            'NAME': 'Test Stadium',
            'ENTITY_TYPE': 'venue',
            'CITY': 'Test City',
        }
        
        result = parse_entity_result(entity_data)
        self.assertIsInstance(result, VenueSchema)
        self.assertEqual(result.NAME, 'Test Stadium')
        self.assertEqual(result.ENTITY_TYPE, 'venue')
    
    def test_parse_entity_result_referee(self):
        """Test parsing referee entity data"""
        entity_data = {
            'NAME': 'Test Referee',
            'ENTITY_TYPE': 'referee',
        }
        
        result = parse_entity_result(entity_data)
        self.assertIsInstance(result, RefereeSchema)
        self.assertEqual(result.NAME, 'Test Referee')
        self.assertEqual(result.ENTITY_TYPE, 'referee')
    
    def test_parse_entity_result_invalid_type(self):
        """Test parsing entity with invalid type"""
        entity_data = {
            'NAME': 'Test Entity',
            'ENTITY_TYPE': 'invalid_type',
        }
        
        result = parse_entity_result(entity_data)
        self.assertIsNone(result)
    
    def test_parse_entity_result_no_type(self):
        """Test parsing entity without ENTITY_TYPE"""
        entity_data = {
            'NAME': 'Test Entity',
        }
        
        result = parse_entity_result(entity_data)
        self.assertIsNone(result)
    
    @patch('toolbox.textual_entity_search.extract_entity')
    def test_textual_entity_search_with_mock_extract(self, mock_extract):
        """Test textual_entity_search with mocked entity extraction"""
        # Mock the extract_entity to return specific entities
        mock_entities = SoccerEntities(
            player=["Messi"],
            team=["Barcelona"],
            unknown=None,
            venue=None,
            referee=None
        )
        mock_extract.return_value = mock_entities
        
        query = "Test query"
        result = textual_entity_search(query)
        
        mock_extract.assert_called_once_with(query)
        self.assertIsInstance(result, SearchingResult)
    
    @patch('toolbox.textual_entity_search.extract_entity')
    def test_textual_entity_search_extract_returns_none(self, mock_extract):
        """Test when extract_entity returns None"""
        mock_extract.return_value = None
        
        query = "Test query"
        result = textual_entity_search(query)
        
        self.assertIsInstance(result, SearchingResult)
        self.assertEqual(len(result.found_entities), 0)
        self.assertEqual(len(result.missing_entities), 0)


class TestSoccerEntitiesSchema(unittest.TestCase):
    """Test SoccerEntities Pydantic model"""
    
    def test_soccer_entities_creation(self):
        """Test creating SoccerEntities instance"""
        entities = SoccerEntities(
            player=["Player1", "Player2"],
            team=["Team1"],
            venue=None,
            referee=None,
            unknown=None
        )
        
        self.assertEqual(entities.player, ["Player1", "Player2"])
        self.assertEqual(entities.team, ["Team1"])
        self.assertIsNone(entities.venue)
        self.assertIsNone(entities.referee)
        self.assertIsNone(entities.unknown)
    
    def test_soccer_entities_all_none(self):
        """Test creating SoccerEntities with all None values"""
        entities = SoccerEntities()
        
        self.assertIsNone(entities.player)
        self.assertIsNone(entities.team)
        self.assertIsNone(entities.venue)
        self.assertIsNone(entities.referee)
        self.assertIsNone(entities.unknown)


class TestSearchingResultSchema(unittest.TestCase):
    """Test SearchingResult Pydantic model"""
    
    def test_searching_result_empty(self):
        """Test creating empty SearchingResult"""
        result = SearchingResult()
        
        self.assertEqual(result.found_entities, [])
        self.assertEqual(result.missing_entities, [])
    
    def test_searching_result_with_data(self):
        """Test creating SearchingResult with data"""
        player = PlayerSchema(NAME="Test Player")
        team = TeamSchema(NAME="Test Team")
        
        result = SearchingResult(
            found_entities=[player, team],
            missing_entities=["Unknown Entity"]
        )
        
        self.assertEqual(len(result.found_entities), 2)
        self.assertEqual(len(result.missing_entities), 1)
        self.assertIsInstance(result.found_entities[0], PlayerSchema)
        self.assertIsInstance(result.found_entities[1], TeamSchema)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
