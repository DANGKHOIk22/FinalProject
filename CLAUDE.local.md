# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Soccer Agent** — an AI-powered chatbot that answers soccer-related questions using a parallel multi-agent architecture built with LangGraph. Handles complex queries about matches, players, teams, and can analyze video/image content.

### Core Technology Stack
- **Framework**: FastAPI (Python 3.11)
- **AI/LLM**: LiteLLM router — `openai/gpt-5.4-nano` primary with `gemini-3.1-flash-lite` fallback (see `app/soccer_agent/factory/llm_config.yaml`); LangChain, LangGraph
- **Databases**: 
  - PostgreSQL (raw conversation history + LangGraph checkpointer)
  - MongoDB (soccer entity information)
  - Qdrant (vector embeddings for RAG + case bank)
  - Redis (semantic cache + standard cache)
- **Video/Image Processing**: OpenCV, FFmpeg
- **Computer Vision Models**: Qwen-VL (DashScope) for entity localization, InsightFace (Azure ML endpoint) for face embeddings
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
│       ├── factory/           # LLM router config (llm_config.yaml)
│       ├── nodes/             # LangGraph nodes (guardrail, planning, worker, aggregator, ...)
│       ├── case_bank/         # Few-shot planning example retriever + cache
│       ├── memory/            # Conversation & long-term memory
│       │   ├── chat_history.py        # PostgreSQL integration
│       │   ├── checkpointer.py        # LangGraph Postgres checkpointer
│       │   ├── conversation_memory.py # Conversation context management
│       │   └── long_term_memory.py    # pgvector user-scoped knowledge
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
- **AgentState**: Parent state containing user query, `clarified_query` (pronoun-resolved query), tool chains, parallel results, guardrail verdict (`is_off_topic`), `video_current_time`
- **WorkerState**: Individual worker state for parallel tool execution

#### Agent Flow (see `SoccerAgent._build_graph`)
1. **Parallel fan-out from START**: `get_history` (ConversationHistoryNode), `context_retrieval` (case bank + long-term memory), `guardrail_classify` (soccer-topic classifier, fail-open with timeout).
2. **Guardrail Gate** (`guardrail_gate`): joins the three branches; off-topic queries route to `guardrail_refusal` → `save_memory`. On-topic queries proceed.
3. **Unified Planning** (`UnifiedPlanningNode.unified_planning_node`): combined query understanding + tool chain planning with per-chain confidence; low-confidence chains become clarifying questions (unless an active `game_id` anchors them); resolves `game_id`/`video_current_time` from CopilotKit context.
4. **Worker Dispatch** (`WorkerNodes.trigger_workers`): dispatches sub-queries to parallel workers via `Send`; short-circuits to aggregator when no tools needed.
5. **Worker Graph** (`WorkerNodes`): per-worker semantic cache check (`_check_cache_node`) → iterative tool execution (`_execution_node`); successful results cached to `sub_query_cache`, errors never cached.
6. **Aggregator** (`AggregatorNode.aggregator_node`): synthesizes parallel results; surfaces system errors explicitly (planning_error, worker error sentinels) instead of disguising them.
7. **Memory Saving** (`SaveToMemoryNode.save_to_memory_node`): persists the turn to PostgreSQL in the background.

### Toolbox (`app/soccer_agent/toolbox/`)

Active tools (see `tool_registry` in `app/soccer_agent/agent.py`):

#### Text/Knowledge Tools
- **entity_augment**: Entity lookup (MongoDB) with internal Tavily wiki fallback
- **game_history_retrieval**: Historical match event log (goals, cards, subs); Tavily match-report fallback; video-position grounding for the active stream
- **game_info_retrieval**: Match metadata (score, lineup, venue); fast path for the active video; Tavily fallback
- **web_news_search**: Recent soccer news via Tavily

#### Visual Tools
- **entity_recognition**: Identify people in images — Qwen-VL localization → OpenCV face refine → InsightFace embeddings → Qdrant voting search. Disabled at startup (tool skipped, app still boots) if the InsightFace endpoint is unreachable.
- **commentary_generation**: Generate commentary from a video clip (vision LLM via `retrieval-augment` route)

> `segment`, `frame_selection`, `choice_selection`, `textual_entity_search`, `textual_retrieval_augment`, `game_search` have been **removed** from the codebase.

### 3. Memory System (`app/soccer_agent/memory/`)

- **Chat History** (`chat_history.py`): Stores raw message history in PostgreSQL via `langchain-postgres`, scoped per session.
- **Checkpointer** (`checkpointer.py`): LangGraph `AsyncPostgresSaver` — agent state persisted per `thread_id`.
- **Conversation Memory** (`conversation_memory.py`): Manages the conversation context injected into prompts.
- **Long-term Memory** (`long_term_memory.py`): pgvector store of user-scoped knowledge (entities, news findings); retrieved by `context_retrieval` each turn, never cached across users.
- **Query understanding** now happens inside `unified_planning_node` (no separate pipeline/file).
- **Tool-enriched history**: Each turn stores: tool usage summary (step details) + final response. This lets future turns see what tools were used.

### 4. API Endpoints (`app/api/`)

#### POST `/chat`
```python
{
    "user_id": "string",
    "session_id": "string (optional)",
    "user_query": "string",
    "additional_material": {            # optional
        "game_id": "string or null",    # from POST /hls/sessions when watching a stream
        "image_id": ["media path/uuid"]
    }
}
```
Legacy clients sending `additional_material` as a bare list of paths are auto-converted to `{"game_id": null, "image_id": [...]}` by a `field_validator` on `ChatRequest`.

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
OPENAI_API_KEY=             # Primary LLM route (gpt-5.4-nano via LiteLLM)
GOOGLE_API_KEY=             # Gemini fallback route + embeddings
DASHSCOPE_API_KEY=          # Required — Qwen-VL/ASR; SoccerAgent is None without it; /chat returns 503

# Computer vision
INSIGHTFACE_ENDPOINT_URI=   # Azure ML endpoint for face embeddings
INSIGHTFACE_ENDPOINT_KEY=

# Databases
MONGO_SRV=
SOCCER_DB_NAME=SoccerWikiDemo
SOCCER_COLLECTION_NAME=EntityInformation
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=
QDRANT_CASE_BANK_COLLECTION_NAME=Soccer_Case_Bank
QDRANT_HLS_COLLECTION_NAME=hls_frame_index
POSTGRES_DATABASE_URL=
REDIS_URL=redis://localhost:6379/0

# Auth
JWT_SECRET_KEY=             # 64-char hex string, required at startup

# Tavily (web search + wiki extract fallback)
# Comma-separated list of keys for round-robin + 429 failover.
# Single key also accepted: TAVILY_API_KEY=...
TAVILY_API_KEYS=key1,key2,key3
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
- **Planning prompts**: `app/soccer_agent/prompts/agent.py` - `get_unified_planning_prompt_template()`
- **Execution prompts**: `app/soccer_agent/prompts/agent.py` - `get_execution_system_prompt()` / `get_execution_human_prompt()`
- **Aggregation prompts**: `app/soccer_agent/prompts/agent.py` - `get_aggregator_prompt_template()`
- **Guardrail prompt**: `app/soccer_agent/prompts/agent.py` - `get_guardrail_prompt_template()`

### Testing
```bash
# Run the full suite — pyproject testpaths covers both tests/ and unit_test/
pytest

# Test specific components
pytest tests/test_worker.py
pytest unit_test/nodes/test_guardrail.py
```

## AI Model Configuration

LLM routing lives in `app/soccer_agent/factory/llm_config.yaml` (LiteLLM router). Roles: `planning`, `execution`, `retrieval-augment`, `aggregator`, `tool`, `guardrail`. Every role uses `openai/gpt-5.4-nano` as primary with `gemini/gemini-3.1-flash-lite` fallback — except `guardrail`, which uses Gemini flash-lite as primary (≈1s latency) with the OpenAI model as backup. Gemini fallbacks use `thinking_level` (not `thinking_budget`).

Constants in `app/config/config.py`:

```python
GEMINI_3_1_FLASH = "gemini-3.1-flash-preview"
GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite-preview"
DEFAULT_MODEL = GEMINI_3_1_FLASH_LITE
MODEL_TEMPERATURE = 1.0
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

1. **LangGraph Architecture**: State-based routing with conditional edges. Flow: `[get_history ∥ context_retrieval ∥ guardrail_classify] → guardrail_gate → (blocked → guardrail_refusal → save_memory) | (proceed → unified_planning → trigger_workers → workers → aggregator → save_memory)`.

2. **Async/Await**: Most operations are async. The agent uses `asyncio` for parallel worker execution; sync cache/embedding I/O runs via `asyncio.to_thread`.

3. **Tool Calling**: Tools are registered with the LLM via `bind_tools`. The execution LLM decides which tool to call at each step; only one tool is called per generation cycle.

4. **Memory Management**: Conversation history saved to PostgreSQL. Each turn stores: tool usage summary (name + args + response + artifact per step) + final response as a single assistant message.

5. **Query Understanding (`clarified_query`)**: Handled inside `unified_planning_node` — pronouns/abbreviations resolved in the same LLM call that plans tool chains. The resolved query is used by workers and the aggregator.

6. **LLM Routing**: All roles route through LiteLLM (`llm_config.yaml`); primary `gpt-5.4-nano`, Gemini 3.1 flash-lite fallback (with `thinking_level`). Guardrail is the exception (Gemini primary for latency).

7. **Semantic Cache** (`sub_query_cache`, Redis vector index): cosine distance threshold 0.15 (≈0.85 similarity) bypasses workers. Video queries filter by `game_id` + a 5s timestamp window; image queries filter by an md5 hash of the sorted `image_id` list. Error results are never cached.

8. **Error Handling**: Most errors are caught and logged. Worker timeouts are 120 seconds; worker/planning failures surface as explicit system-error messages from the aggregator, never disguised as user ambiguity.

9. **Configuration Priority**: Environment variables > `settings.py` > `config.py` defaults

## Contact & Support

For questions about this codebase, review:
- Agent implementation: [app/soccer_agent/agent.py](app/soccer_agent/agent.py)
- API endpoints: [app/api/chat.py](app/api/chat.py)
- Configuration: [app/config/settings.py](app/config/settings.py)

---

**Last Updated**: 2026-06-12
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
