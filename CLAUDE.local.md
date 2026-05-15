# CLAUDE.md - Soccer Agent Project Guide

## Project Overview

This is a **Soccer Agent** application - an intelligent AI-powered chatbot that answers soccer-related questions using a multi-agent architecture built with LangGraph. The system can handle complex queries about soccer matches, players, teams, and analyze video/image content.

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
│       ├── agent.py           # Main SoccerAgent orchestrator (LangGraph Builder)
│       ├── factory/           # Agent factory patterns
│       ├── memory/            # Conversation & system memory
│       │   ├── chat_history.py # PostgreSQL integration
│       │   ├── session_memory.py # History summarization & Redis cache
│       │   └── query_understanding.py # Pronoun resolution pipeline
│       ├── nodes/             # LangGraph Nodes
│       │   ├── conversation_history.py
│       │   ├── user_message_understanding.py
│       │   ├── case_retrieval.py
│       │   ├── tool_planning.py
│       │   ├── worker.py
│       │   ├── aggregator.py
│       │   └── memory_saving.py
│       ├── prompts/           # LLM prompts
│       └── toolbox/           # Agent tools
├── scripts/
│   ├── init_db.py            # Database initialization
│   └── host_model/           # Model deployment scripts (Azure ML)
├── unit_test/                # Test suites
└── main.py                   # FastAPI application entry point
```

## Key Components

### 1. Agent Architecture (`app/soccer_agent/`)

The **SoccerAgent** implements a parallel multi-worker architecture using LangGraph. The main `agent.py` acts as an orchestrator, while the actual node logic is split into `app/soccer_agent/nodes/` for better maintainability.

#### State Management
- **AgentState**: Parent state containing user query, `claried_query` (pronoun-resolved query), tool chains, parallel results, `recent_msgs_for_qu`, and `effective_memory`.
- **WorkerState**: Individual worker state for parallel tool execution.

#### Agent Flow & Node Files
1. **Conversation History** (`nodes/conversation_history.py`): Retrieves chat history from Postgres/Redis. Determines if memory needs summarization.
2. **Query Understanding** (`nodes/user_message_understanding.py`): Resolves pronouns/abbreviations and handles domain jargon. Detects ambiguity and can short-circuit to ask clarifying questions.
3. **Case Retrieval** (`nodes/case_retrieval.py`): Pulls few-shot planning examples from CaseBank/Redis.
4. **Planning Node** (`nodes/tool_planning.py`): Analyzes `claried_query`, checks history, plans parallel tool chains (`PlanningOutput`).
5. **Worker Graph** (`nodes/worker.py`): Contains the execution subgraph:
   - **Cache Node** (`_check_cache_node`): Checks Semantic Cache for similar previous queries to bypass execution.
   - **Worker Dispatch** (`trigger_workers`): Dispatches sub-queries to parallel workers via `Send` commands; uses `claried_query` as fallback.
   - **Execution Workers** (`_execution_node`): Each executes a tool chain independently; supports Thinking models.
6. **Aggregator Node** (`nodes/aggregator.py`): Synthesizes parallel results into a final definitive response.
7. **Memory Saving** (`nodes/memory_saving.py`): Asynchronously saves structured tool summaries and final response to PostgreSQL and Session Memory.

#### Key Orchestrator Methods (`agent.py`)
- `run(request: ChatRequest)` - Main entry point. Initializes state, updates tracing, and invokes the graph.
- `_build_graph()` - Compiles the parent state graph with edges between the injected Node classes.


### 2. Toolbox (`app/soccer_agent/toolbox/`)

Available tools for the agent:

#### Text/Knowledge Tools
- **entity_augment** (`entity_augment.py`): Merged replacement for the old `textual_entity_search` + `textual_retrieval_augment` two-step chain. Accepts `entity_names: List[str]` + `query: str`. Internally:
  1. Detects entity type from Vietnamese/English type labels in `query` (e.g. "câu lạc bộ", "cầu thủ", "coach", "stadium") to build a typed `SoccerEntities` for a precise MongoDB query.
  2. Falls back to `unknown` (unfiltered name-only search) when no label matches.
  3. Name matching uses `_name_regex_variants()` which strips FC/AFC/CF/SC suffixes so "Manchester City FC" matches "Manchester City" in DB.
  4. If DB answer is insufficient (`has_sufficient_info=False`), runs Tavily fallback: wiki extract → clean markdown → LLM answer. Background-upserts result to MongoDB via `asyncio.create_task`.
  5. If no wiki URL found → `search_general`, no DB save.
  - **Input rule**: entity_names must be core names only — no FC/AFC/CF/SC suffixes, no type labels.
- **web_news_search** (`web_search.py`): Search recent soccer news via Tavily. Use for transfers, injuries, live results, upcoming fixtures. Inputs: `query`, `time_range` (day/week/month).
- **game_history_retrieval**: Get historical match event log (goals, cards, substitutions).
- **game_info_retrieval**: Get match metadata (score, lineup, venue, attendance).

#### Visual Tools
- **segment**: Segment images to detect objects (uses GroundingDINO)
- **commentary_generation**: Generate commentary from visual analysis
- ~~**entity_recognition**~~: *Currently disabled* (uses DeepFace) — commented out in `tool_registry`
- ~~**frame_selection**~~: *Currently disabled* (uses CLIP) — commented out in `tool_registry`

#### Utility Tools
- **choice_selection**: Select best option from choices

#### New Services (`app/soccer_agent/services/`)
- **`tavily_service.py`** (`TavilyService`): Wraps all Tavily calls. Key methods: `find_wiki_url(name)`, `extract_wiki(url)` (full page, no query), `search_news(query, time_range)`, `search_general(query)`. Supports multi-key pool via `TAVILY_API_KEYS` env var (comma-separated) with round-robin + 429 failover.
- **`content_cleaner.py`**: `clean_wiki_markdown(raw)` strips nav/citations/refs; `extract_summary(md)` returns intro paragraph after `# Title` heading.

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

# Redis Configuration
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
- **EntityInformation**: Soccer entities (players, teams, coaches, venues, referees)
  - New schema (post-migration): `NAME`, `ENTITY_TYPE`, `<TYPE>_URL`, `SUMMARY` (intro paragraph), `CONTENT` (full cleaned markdown string), `IMAGES` (list of URLs). `INFOBOX` field removed.
  - `CONTENT` field is `Union[str, Dict]` in Pydantic schemas for backward compatibility with old documents that still store a dict.

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

5. **Pronoun Resolution (`claried_query`)**: Managed by `QueryUnderstandingPipeline` before graph entry. It resolves pronouns/abbreviations using recent context and Session Memory. The resolved query is used by workers and the aggregator. The QU prompt also rewrites entity mentions with type labels (e.g. "câu lạc bộ Manchester City", "cầu thủ Ronaldo") so downstream tools know the entity type.

6. **Thinking Models**: The agent utilizes Gemini 2.0/2.5 Thinking models with specific `thinking_budget` (3000-4000) for complex reasoning during planning and execution.

7. **Semantic Cache**: Queries with >0.9 similarity bypass workers via Redis-backed semantic cache.

8. **Incremental Memory**: After turns with tools, key facts are distilled into `SessionMemory.tool_findings` in the background.

9. **Error Handling**: Most errors are caught and logged. Check `logger.error()` calls for debugging. Worker timeouts are set to 120 seconds.

10. **Configuration Priority**: Environment variables > `settings.py` > `config.py` defaults

11. **entity_augment tool** (replaces old `textual_entity_search` + `textual_retrieval_augment` chain):
    - Planning LLM must pass **one entity per chain** — use parallel chains for multiple entities.
    - `entity_names` must be **core names only** (no FC/AFC/CF/SC/SSC suffixes, no type labels).
    - Entity type is conveyed via the `query` / sub-query text using Vietnamese or English type labels ("câu lạc bộ X", "cầu thủ X", "coach X", "stadium X"). The tool parses this to issue a typed MongoDB query.
    - Tavily fallback is **internal** — the planning LLM does not need to know about it.
    - Background `asyncio.create_task` upserts wiki content to MongoDB after a Tavily fetch; does not block the response.

12. **Planning prompt entity conventions** (`app/soccer_agent/prompts/agent.py`):
    - Sub-queries for `entity_augment` must include type label: `"cầu thủ Ronaldo"`, `"câu lạc bộ Manchester City"`, `"sân Old Trafford"`.
    - No FC/AFC/CF/SC suffixes in sub-queries.
    - `current_date` is injected so the LLM can reason about recency.

13. **MongoDB fuzzy name matching**: `_name_regex_variants()` in `entity_augment.py` builds `$or` queries with suffix-stripped variants so "Manchester City FC" finds "Manchester City" in the DB.

## Contact & Support

For questions about this codebase, review:
- Agent implementation: [app/soccer_agent/agent.py](app/soccer_agent/agent.py)
- API endpoints: [app/api/chat.py](app/api/chat.py)
- Configuration: [app/config/settings.py](app/config/settings.py)

---

**Last Updated**: 2026-05-11
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
