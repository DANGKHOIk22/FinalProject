# CLAUDE.md - Soccer Agent Project Guide

## Project Overview

This is a **Soccer Agent** application - an intelligent AI-powered chatbot that answers soccer-related questions using a multi-agent architecture built with LangGraph. The system can handle complex queries about soccer matches, players, teams, and analyze video/image content.

### Core Technology Stack
- **Framework**: FastAPI (Python 3.11)
- **AI/LLM**: Google Gemini 2.5 Flash, LangChain, LangGraph
- **Databases**: 
  - PostgreSQL (conversation history)
  - MongoDB (entity information)
  - Qdrant (vector embeddings)
- **Video/Image Processing**: OpenCV, FFmpeg
- **Computer Vision Models**: DeepFace, CLIP, GroundingDINO (deployed to Azure endpoints)

## Project Structure

```
FinalProject/
├── app/
│   ├── api/                    # FastAPI routes
│   │   ├── chat.py            # Main chat endpoint
│   │   └── user.py            # User management
│   ├── config/                # Configuration management
│   │   ├── config.py          # App-wide constants
│   │   └── settings.py        # Environment variables loader
│   ├── database/              # Database models and connections
│   │   ├── db.py              # PostgreSQL connection
│   │   ├── models.py          # SQLAlchemy models
│   │   └── Game_dataset/      # Soccer match data (10 years, multiple leagues)
│   ├── schema/                # Pydantic schemas
│   │   ├── chat.py            # Chat request/response models
│   │   ├── match.py           # Match data models
│   │   └── soccerwiki_entities.py  # Entity schemas
│   └── soccer_agent/          # Core agent implementation
│       ├── agent.py           # Main SoccerAgent class (LangGraph)
│       ├── factory/           # Agent factory patterns
│       ├── memory/            # Conversation & system memory
│       ├── prompts/           # LLM prompts
│       └── toolbox/           # Agent tools
├── scripts/
│   ├── init_db.py            # Database initialization
│   └── host_model/           # Model deployment scripts (Azure ML)
├── unit_test/                # Test suites
└── main.py                   # FastAPI application entry point
```

## Key Components

### 1. Agent Architecture (`app/soccer_agent/agent.py`)

The **SoccerAgent** implements a parallel multi-worker architecture using LangGraph:

#### State Management
- **AgentState**: Parent state containing user query, `claried_query` (pronoun-resolved query), tool chains, parallel results
- **WorkerState**: Individual worker state for parallel tool execution

#### Agent Flow
1. **Planning Node** (`_tool_chain_planning`): Analyzes user query, resolves pronouns → `claried_query`, decomposes into sub-queries, plans tool chains
2. **Worker Dispatch** (`_trigger_workers`): Routes sub-queries to parallel workers; uses `claried_query` as fallback when no sub_queries
3. **Execution Workers** (`_execution_node`): Each executes a tool chain independently
4. **Aggregator Node** (`_aggregator_node`): Combines parallel results into final response; uses `claried_query` as effective user query

#### Key Methods
- `run(request: ChatRequest)` - Main entry point for processing user queries
- `_tool_chain_planning()` - Query decomposition, pronoun resolution, and tool chain planning
- `_worker_node()` / `_execution_node()` - Individual tool chain execution
- `_aggregator_node()` - Result aggregation
- `_trigger_workers()` - Dispatches to workers or direct response
- `_build_tool_summary_for_memory()` - Formats tool call details (name, args, response, artifact) for saving into conversation memory

### 2. Toolbox (`app/soccer_agent/toolbox/`)

Available tools for the agent:

#### Text/Knowledge Tools
- **textual_entity_search**: Search for soccer entities (players, teams, coaches)
- **textual_retrieval_augment**: RAG-based retrieval from knowledge base
- **game_history_retrieval**: Get historical match data
- **game_info_retrieval**: Get specific match information
- **game_search**: Search matches by criteria

#### Visual Tools
- **segment**: Segment images to detect objects (uses GroundingDINO)
- **commentary_generation**: Generate commentary from visual analysis
- ~~**entity_recognition**~~: *Currently disabled* (uses DeepFace) — commented out in `tool_registry`
- ~~**frame_selection**~~: *Currently disabled* (uses CLIP) — commented out in `tool_registry`

#### Utility Tools
- **choice_selection**: Select best option from choices

### 3. Memory System (`app/soccer_agent/memory/`)

- **PostgreSQL Memory**: Stores conversation history with `langchain-postgres`
- **System Prompt Memory** (`CustomSystemPromptMemory`): Manages conversation context and clarifications; trims to last `max_history=15` messages
- **Tool-enriched history**: Each saved turn includes tool call details (tool name, input args, response content, artifact if any) formatted via `_build_tool_summary_for_memory()`, followed by the final response. This lets future turns see what tools were used and what they returned.

### 4. API Endpoints (`app/api/`)

#### POST `/chat`
```python
{
    "user_id": "string",
    "session_id": "string (optional)",
    "user_query": "string",
    "additional_material": ["image_url or path (optional)"]
}
```

Response:
```python
{
    "status": "success",
    "user_query": "original query",
    "agent_response": "AI response"
}
```

## Environment Configuration

Required environment variables in `.env`:

```env
# LLM API Keys
GOOGLE_API_KEY=your_gemini_api_key
DASHSCOPE_API_KEY=your_dashscope_key (optional)

# MongoDB (Entity Storage)
MONGO_SRV=mongodb+srv://...
SOCCER_DB_NAME=SoccerWikiDemo
SOCCER_COLLECTION_NAME=EntityInformation

# Qdrant (Vector DB)
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION_NAME=your_collection

# PostgreSQL (Conversation History)
POSTGRES_DATABASE_URL=postgresql://user:pass@host:port/db

# Azure ML Endpoints (Computer Vision)
DEEPFACE_ENDPOINT_URI=https://...
DEEPFACE_ENDPOINT_KEY=...
GROUNDINGDINO_ENDPOINT_URI=https://...
GROUNDINGDINO_ENDPOINT_KEY=...
CLIP_ENDPOINT_URI=https://...
CLIP_ENDPOINT_KEY=...
CLIP_GROUNDINGDINO_ENDPOINT_URI=https://...
CLIP_GROUNDINGDINO_ENDPOINT_KEY=...

# Optional
DEEPFACE_HOME=./temporary/cache
```

## Database Schema

### PostgreSQL (SQLAlchemy Models in `app/database/models.py`)
- **Users**: User authentication and profiles
- **Conversations**: Chat session metadata
- **Messages**: Individual chat messages (managed by langchain-postgres)

### MongoDB Collections
- **EntityInformation**: Soccer entities (players, teams, coaches, competitions)

### Qdrant Collections
- Vector embeddings for semantic search

## Running the Application

### Development Setup
```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e .

# Initialize database
python scripts/init_db.py

# Run the application
python main.py
# or
uvicorn main:app --reload --port 8000
```

### Application Lifecycle
- **Startup**: Initializes database connections (MongoDB, Qdrant, PostgreSQL), validates configurations
- **Runtime**: Agent service processes chat requests asynchronously
- **Shutdown**: Gracefully closes database connections

## Development Guidelines

### Adding New Tools
1. Create tool file in `app/soccer_agent/toolbox/`
2. Implement as `BaseTool` from LangChain
3. Register in `app/soccer_agent/toolbox/__init__.py`
4. Update agent initialization in `agent.py`

### Modifying Agent Behavior
- **Planning prompts**: `app/soccer_agent/prompts/agent.py` - `get_planning_prompt_template()`
- **Execution prompts**: `app/soccer_agent/prompts/agent.py` - `get_execution_prompt_template()`
- **Aggregation prompts**: `app/soccer_agent/prompts/agent.py` - `get_aggregator_prompt_template()`

### Testing
```bash
# Run unit tests
pytest unit_test/

# Test specific components
pytest unit_test/agent/test_parallel_architecture.py
pytest unit_test/tools/test_entity_recognition.py
```

## AI Model Configuration

Models are configured in `app/config/config.py`:

```python
DEFAULT_MODEL = "gemini-2.0-flash-exp"
GEMINI_2_5_FLASH = "gemini-2.0-flash-exp"
GEMINI_2_5_FLASH_LITE = "gemini-2.0-flash-lite"
MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.95
MAX_COMPLETION_TOKENS = 8192
```

## Logging & Monitoring

- **Logging**: Python's standard logging + structlog
- **Tracing**: Langfuse integration for LLM call tracking
- **Observability**: `@observe` decorator on key agent methods

## Data Sources

### Game Dataset (`app/database/Game_dataset/`)
Contains 10 years of match data from:
- England EPL (2014-2024)
- Europe Champions League (2014-2024)
- France Ligue 1 (2014-2024)
- Germany Bundesliga (2014-2024)
- And more leagues...

Each match includes: teams, scores, lineups, statistics, events, etc.

## Common Issues & Solutions

### Database Connection Errors
- Verify `.env` file has correct credentials
- Check MongoDB/Qdrant/PostgreSQL services are running
- Test connections before agent initialization

### Tool Execution Failures
- Check Azure ML endpoints are deployed and accessible
- Verify API keys for external services
- Review tool input/output schemas

### Memory Issues with Video Processing
- Videos are processed frame-by-frame
- Frames stored temporarily in `temporary/frames/`
- Cleanup happens after processing

## API Integration Example

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "user_id": "user123",
        "user_query": "Who scored in the last Manchester United vs Liverpool match?",
        "session_id": "session456"
    }
)

result = response.json()
print(result["agent_response"])
```

## Future Enhancements

- [ ] Streaming responses for long-running queries
- [ ] Multi-language support
- [ ] Real-time match data integration
- [ ] Enhanced video analysis capabilities
- [ ] Performance optimization for parallel workers

## Important Notes for Claude

1. **LangGraph Architecture**: State-based routing with conditional edges. Flow: planning → dispatch → workers → aggregation.

2. **Async/Await**: Most operations are async. The agent uses `asyncio` for parallel worker execution.

3. **Tool Calling**: Tools are registered with the LLM via `bind_tools`. The execution LLM decides which tool to call at each step; only one tool is called per generation cycle.

4. **Memory Management**: Conversation history saved to PostgreSQL via `CustomSystemPromptMemory`. Each turn stores: tool usage summary (name + args + response + artifact per step) + final response as a single assistant message.

5. **Pronoun Resolution (`claried_query`)**: The planning node resolves pronouns in the user query by checking conversation history. The resolved query is stored in `AgentState.claried_query` and used by workers (as sub_query fallback) and the aggregator (as the effective user query). The original `user_query` is preserved for traceability.

6. **Error Handling**: Most errors are caught and logged. Check `logger.error()` calls for debugging. Worker timeouts are set to 120 seconds.

7. **Configuration Priority**: Environment variables > `settings.py` > `config.py` defaults

## Contact & Support

For questions about this codebase, review:
- Agent implementation: [app/soccer_agent/agent.py](app/soccer_agent/agent.py)
- API endpoints: [app/api/chat.py](app/api/chat.py)
- Configuration: [app/config/settings.py](app/config/settings.py)

---

**Last Updated**: 2026-03-28
**Python Version**: 3.11
**Framework Version**: FastAPI 0.118.2, LangGraph (latest)
