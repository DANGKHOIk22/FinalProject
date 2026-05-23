# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Soccer Agent** — an AI-powered chatbot that answers soccer-related questions using a parallel multi-agent architecture built with LangGraph. Handles complex queries about matches, players, teams, and can analyze video/image content.

### Core Technology Stack
- **Framework**: FastAPI (Python 3.11)
- **AI/LLM**: Google Gemini 2.0/2.5 Flash & Flash Lite, LangChain, LangGraph
- **Databases**: 
  - PostgreSQL (raw conversation history)
  - MongoDB (soccer entity information)
  - Qdrant (vector embeddings for RAG)
  - Redis (session memory caching and semantic cache)
- **Video/Image Processing**: OpenCV, FFmpeg
- **Computer Vision Models**: DeepFace, CLIP, GroundingDINO (deployed to Azure endpoints)
- **Observability**: Langfuse (tracing, monitoring, evaluation)

## Project Structure

```
FinalProject/
├── app/
│   ├── api/                    # FastAPI routes
│   │   ├── chat.py            # Main chat endpoint
│   │   └── user.py            # User management
│   ├── cache/                 # Caching logic
│   │   ├── semantic_cache.py  # Semantic similarity cache (Redis + Embeddings)
│   │   └── standard_cache.py  # Basic Redis wrapper
│   ├── config/                # Configuration management
│   │   ├── config.py          # App-wide constants (Models, Thresholds)
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
│       │   ├── chat_history.py # PostgreSQL integration
│       │   ├── session_memory.py # History summarization & Redis cache
│       │   └── query_understanding.py # Pronoun resolution pipeline
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
1. **Query Understanding** (`QueryUnderstandingPipeline`): Resolves pronouns/abbreviations and handles domain jargon. Detects ambiguity and can short-circuit to ask clarifying questions.
2. **Planning Node** (`_tool_chain_planning`): Analyzes `claried_query`, checks history, plans parallel tool chains (`PlanningOutput`).
3. **Cache Node** (`_check_cache_node`): Checks Semantic Cache for similar previous queries to bypass execution.
4. **Worker Dispatch** (`_trigger_workers`): Dispatches sub-queries to parallel workers via `Send` commands; uses `claried_query` as fallback.
5. **Execution Workers** (`_execution_node`): Each executes a tool chain independently; supports Thinking models with a thinking budget.
6. **Aggregator Node** (`_aggregator_node`): Synthesizes parallel results into a final definitive response; uses `claried_query` as effective user query.

#### Key Methods
- `run(request: ChatRequest)` - Main entry point. Handles memory loading, query clarification, graph execution, and background memory saving.
- `_tool_chain_planning()` - Query decomposition, pronoun resolution, and tool chain planning.
- `_worker_node()` / `_execution_node()` - Individual tool chain execution.
- `_aggregator_node()` - Result aggregation.
- `_trigger_workers()` - Dispatches to workers or direct response.
- `_build_tool_summary_for_memory()` - Formats tool details for saving into conversation memory.
- `_background_save_memory()` - Asynchronously saves results to PostgreSQL and updates Session Memory.

### Toolbox (`app/soccer_agent/toolbox/`)

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

- **PostgreSQL Context**: Stores raw message history via `langchain-postgres`.
- **Query Understanding** (`query_understanding.py`): A single-pass Flash Lite call that resolves "it", "they", "that team" to specific entities.
- **Session Memory** (`session_memory.py`):
    - **Path A**: If history < threshold, uses raw text context.
    - **Path B**: If history > threshold, loads a structured summary from Redis.
    - **Background Tasks**: Triggers summarization of old messages and incremental updates of "tool findings" to keep history compact.
- **System Prompt Memory** (`CustomSystemPromptMemory`): Manages conversation context and clarifications; trims to last `max_history=15` messages.
- **Tool-enriched history**: Each turn stores: tool usage summary (step details) + final response. This lets future turns see what tools were used.

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
# LLM
GOOGLE_API_KEY=
DASHSCOPE_API_KEY=          # Required — SoccerAgent is None without it; /chat returns 503

# Databases
MONGO_SRV=
SOCCER_DB_NAME=SoccerWikiDemo
SOCCER_COLLECTION_NAME=EntityInformation
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=
QDRANT_CASE_BANK_COLLECTION_NAME=planning_case_bank
QDRANT_HLS_COLLECTION_NAME=hls_frame_index
POSTGRES_DATABASE_URL=
REDIS_URL=redis://localhost:6379/0

# Tavily (web search + wiki extract fallback)
# Comma-separated list of keys for round-robin + 429 failover.
# Single key also accepted: TAVILY_API_KEY=...
TAVILY_API_KEYS=key1,key2,key3

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
GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
DEFAULT_MODEL = GEMINI_2_5_FLASH
MODEL_TEMPERATURE = 0.2
MODEL_TOP_P = 0.95
MAX_COMPLETION_TOKENS = 8000
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

5. **Pronoun Resolution (`claried_query`)**: Managed by `QueryUnderstandingPipeline` before graph entry. It resolves pronouns/abbreviations using recent context and Session Memory. The resolved query is used by workers and the aggregator.

6. **Thinking Models**: The agent utilizes Gemini 2.0/2.5 Thinking models with specific `thinking_budget` (3000-4000) for complex reasoning during planning and execution.

7. **Semantic Cache**: Queries with >0.9 similarity bypass workers via Redis-backed semantic cache.

8. **Incremental Memory**: After turns with tools, key facts are distilled into `SessionMemory.tool_findings` in the background.

9. **Error Handling**: Most errors are caught and logged. Check `logger.error()` calls for debugging. Worker timeouts are set to 120 seconds.

7. **Configuration Priority**: Environment variables > `settings.py` > `config.py` defaults

## Contact & Support

For questions about this codebase, review:
- Agent implementation: [app/soccer_agent/agent.py](app/soccer_agent/agent.py)
- API endpoints: [app/api/chat.py](app/api/chat.py)
- Configuration: [app/config/settings.py](app/config/settings.py)

---

**Last Updated**: 2026-04-05
**Python Version**: 3.11+
**Framework Version**: FastAPI 0.118.2, LangGraph (latest)

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
