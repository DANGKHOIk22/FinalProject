# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (use uv, not pip)
uv sync

# Run dev server
uv run uvicorn main:app --reload --port 8000

# Run Celery worker (required for HLS background tasks)
# -P solo is required on Windows — the default prefork pool is POSIX-only
python -m celery -A app.celery_app worker --loglevel=info -P solo

# Purge all pending Celery tasks
python -m celery -A app.celery_app purge -f

# Initialize DB tables and Qdrant collections
uv run python scripts/init_db.py

# Run all tests  ← pyproject.toml has testpaths=["tests"] which is wrong; always pass path explicitly
uv run pytest unit_test/

# Run a single test file
uv run pytest unit_test/tools/test_video_streaming.py

# Async tests require @pytest.mark.asyncio (asyncio_mode = "strict" in pyproject.toml)
```

## Agent Execution Flow

```
User Query
  → get_history          (PostgreSQL conversation history)    ─┐
  → context_retrieval    (Qdrant case bank, top-3 positive)   ─┤ parallel
  → guardrail_classify   (soccer-topic classifier, fail-open) ─┘
  → guardrail_gate       (fan-in join)
      ├─ is_off_topic=True  → guardrail_refusal → save_memory → END
      └─ is_off_topic=False → unified_planning
  → unified_planning     (pronoun resolution + chain decomposition → UnifiedPlanningOutput)
      ├─ need_call_tools=False → aggregator (direct response from history)
      └─ need_call_tools=True  → worker_graph (LangGraph Send → N parallel worker subgraphs)
            worker subgraph:
              → check_cache_node  (Redis sub_query_cache, cosine threshold=0.15, ttl=3600)
                  ├─ cache hit → END (skip execution)
                  └─ miss    → execution_node (LLM + tool calls, loops until done)
                                   └─ tool_node (ToolNode executes one tool per cycle)
  → aggregator           (merges parallel_results using clarified_query)
  → save_memory          (PostgreSQL + Redis, background-async)
```

**Critical**: Workers and aggregator always use `clarified_query`, never `user_query`. Set by `unified_planning` after pronoun resolution.

**Guardrail**: Runs in parallel with `get_history` / `context_retrieval`. Classifies the query as soccer-related using a fast LLM (`guardrail` role in `llm_config.yaml`). Fails open on any error or timeout (`GUARDRAIL_TIMEOUT_SECONDS = 10.0`). Off-topic queries get a canned refusal (`GUARDRAIL_REFUSAL_MESSAGE`) and skip planning/workers/aggregator entirely.

**`unified_planning` reads `game_id`** from `additional_material` dict in state, or from the CopilotKit context blob (JSON under `state.copilotkit.context`) if not set on state directly. `video_current_time` is a top-level `AgentState` field (float, seconds). All planned chains are dispatched regardless of confidence — the `PLANNING_CONFIDENCE_THRESHOLD` ambiguity check is currently disabled (commented out in `unified_planning.py`).

**`trigger_workers` resolves `game_id`** from `additional_material` dict (not a top-level state field), then merges it into each worker's `additional_material` dict before dispatch.

**Streaming UI events**: `unified_planning` emits a `manually_emit_tool_call` custom LangGraph event (via `adispatch_custom_event`) so the frontend CopilotKit adapter can render planning progress in real time.

## LLM Model Routing

Actual models are defined in `app/soccer_agent/factory/llm_config.yaml` via LiteLLM Router — this file is authoritative.

| Role | Primary | Fallback |
|------|---------|----------|
| planning | `openai/gpt-5.4-nano` (reasoning_effort: high) | `gemini/gemini-3.1-flash-lite` |
| execution | `openai/gpt-5.4-nano` | `gemini/gemini-3.1-flash-lite` |
| retrieval-augment | `openai/gpt-5.4-nano` (streaming) | `gemini/gemini-3.1-flash-lite` |
| aggregator | `openai/gpt-5.4-nano` (reasoning_effort: medium, streaming) | `gemini/gemini-3.1-flash-lite` |
| tool | `openai/gpt-5.4-nano` | `gemini/gemini-3.1-flash-lite` |

`OPENAI_API_KEY` is required for the primary path. `GOOGLE_API_KEY` enables the fallback.

## Current Tool Registry

All tools registered in `SoccerAgent.tool_registry` (`app/soccer_agent/agent.py`):

| Tool | File | Purpose |
|------|------|---------|
| `entity_augment` | `entity_augment.py` | Search + RAG for soccer entities (players, teams, coaches) |
| `game_history_retrieval` | `game_retrieval.py` | Historical match data lookup |
| `game_info_retrieval` | `game_retrieval.py` | Specific match info lookup |
| `entity_recognition` | `entity_recognition.py` | Player recognition via face recognition (InsightFace) + Qdrant |
| `commentary_generation` | `commentary_generation.py` | Visual commentary from frame analysis |
| `web_news_search` | `web_search.py` | Tavily web search for post-2024 or news queries |

## Storage Layer

| Store | Purpose |
|-------|---------|
| MongoDB | Soccer entities (players, teams, coaches) |
| Qdrant | RAG knowledge base, HLS frame embeddings, case bank |
| PostgreSQL | Chat history, LangGraph checkpoints |
| Redis | Semantic cache (`sub_query_cache`, `context_cache`), session memory summaries, Celery broker |

## Critical Gotchas

**`testpaths = ["tests"]` in pyproject.toml is misconfigured.** Real tests are in `unit_test/`. Always pass path explicitly.

**`SoccerAgent` is `None` at startup if `DASHSCOPE_API_KEY` is missing.** `/chat` returns 503 — not a bug.

**Semantic cache is split into two isolated Redis indexes** (`sub_query_cache` and `context_cache`), both `SemanticCache` instances in `app/cache/semantic_cache.py`. Reads and writes are fully active (`threshold=0.15`, `ttl=3600`). The cache filters by `game_id` + `timestamp` window for video queries and by `image_id` + `game_id == "__none__"` for non-video queries. **If you have a stale `semantic_cache` index from before the split, flush it**: `redis-cli DEL semantic_cache`.

**`additional_material` is now a dict**, not a list. Structure: `{"game_id": str | None, "image_id": List[str]}`. The `game_id` key in this dict is the canonical place `trigger_workers` and the cache use to route video-aware execution. Do not pass `additional_material` as `List[str]`.

**Celery on Windows requires `-P solo`.** The default prefork pool is POSIX-only.

**ffmpeg and ffprobe must be in `$PATH`.** Hard-required for segment generation; ffprobe falls back to OpenCV if missing.

**HLS sessions are in-memory only.** `_videos` dict in `app/api/hls_stream.py` is lost on restart. Task status is preserved in Redis (`hls:task:<task_id>` key, 7-day TTL).

**`STT_BACKEND=stub` by default.** Set to `gemini` / `whisper_api` / `whisper_local` plus the appropriate API key for real transcription.

**`VIDEO_SEGMENT_DURATION` defaults to 5 seconds** (not 30 — the `.env` default and config value differ from older docs).

**`IncrementalSegmentIndexer` uses VLM captions + `text-embedding-v4`.** Each frame is captioned by `qwen3-vl-flash-2026-01-22` then embedded with `dashscope.TextEmbedding` (`text-embedding-v4`). The `hls_frame_index` Qdrant collection uses `dense_caption` (not `dense_image`). **Breaking**: if you have an existing `hls_frame_index` collection from the old vision-embedding schema, drop and recreate it.

**`processor.py` is HLS-only.** It no longer downloads or splits full videos — it assumes HLS segments already exist on disk and only does per-segment frame/audio extraction.

**Do not remove `@observe` decorators** (Langfuse) — they are the primary production debugging tool.

## Key Files

| File | Purpose |
|------|---------|
| `app/soccer_agent/agent.py` | `SoccerAgent` — builds graph, registers tools, entry point `run()` |
| `app/soccer_agent/nodes/unified_planning.py` | Pronoun resolution + tool chain decomposition in one LLM call |
| `app/soccer_agent/nodes/worker.py` | `WorkerNodes` — cache check, execution loop, `trigger_workers` dispatch |
| `app/soccer_agent/nodes/aggregator.py` | Merges parallel worker results into final response |
| `app/soccer_agent/nodes/conversation_history.py` | Loads PostgreSQL chat history into state |
| `app/soccer_agent/nodes/context_retrieval.py` | Qdrant case bank few-shot retrieval |
| `app/soccer_agent/nodes/memory_saving.py` | Async background save to PostgreSQL + Redis |
| `app/soccer_agent/nodes/guardrail.py` | `GuardrailNode` — soccer-topic classifier; `classify_node` (parallel), `gate_node` (fan-in), `gate_router`, `refusal_node` |
| `app/schema/soccer_agent/state.py` | `AgentState`, `WorkerState`, `UnifiedPlanningOutput`, `PlannedChain` |
| `app/soccer_agent/factory/llm_config.yaml` | **Authoritative** LiteLLM router config — actual model names |
| `app/soccer_agent/prompts/agent.py` | Planning, execution, and aggregator prompt templates |
| `app/config/config.py` | Numeric thresholds, model alias constants, `TEMPORARY_DIR` |
| `app/config/settings.py` | Env-var loader — `.env` anchored to `FinalProject/` to survive Celery CWD changes |
| `app/soccer_agent/toolbox/__init__.py` | Tool imports — add new tools here |
| `app/cache/semantic_cache.py` | `SemanticCache`, `sub_query_cache`, `context_cache` instances |
| `app/celery_app.py` | Celery app instance; Redis broker+backend; includes `hls_tasks` |
| `app/api/hls_stream.py` | HLS session registration, playlist proxy, segment proxy, status polling |
| `app/video_processing/processor.py` | `VideoStreamingProcessor`, `HLSSegmentWatcher`, `VideoSegment` |
| `app/video_processing/segment_index.py` | `IncrementalSegmentIndexer` — DashScope vision embeddings → Qdrant |
| `app/video_processing/speech_to_text.py` | `SpeechToTextService` — DashScope Qwen3-ASR |
| `app/video_processing/services/match_indexing_service.py` | Orchestrates watcher + indexer per HLS session |
| `app/video_processing/tasks/hls_tasks.py` | Celery task `hls.process_video_session` |
| `app/services/media_registry.py` | `MediaRegistryService` — Azure Blob Storage + Redis media registry |
| `scripts/case_bank_manager.ipynb` | Manage planning few-shot examples in `planning_case_bank` Qdrant collection |

## Environment Variables

In `FinalProject/.env`. `Settings.validate()` exists but is **not** called at startup — missing vars cause runtime errors, not startup failures.

```
# LLM
OPENAI_API_KEY=             # Required — primary LLM provider (gpt-5.4-nano)
GOOGLE_API_KEY=             # Required for Gemini fallback + semantic cache embeddings
DASHSCOPE_API_KEY=          # Required — SoccerAgent is None without it; /chat returns 503

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

# Tavily (web search)
TAVILY_API_KEYS=key1,key2   # Comma-separated for round-robin + 429 failover

# Azure ML endpoints (vision tools)
DEEPFACE_ENDPOINT_URI=
DEEPFACE_ENDPOINT_KEY=
INSIGHTFACE_ENDPOINT_URI=
INSIGHTFACE_ENDPOINT_KEY=
GROUNDINGDINO_ENDPOINT_URI=
GROUNDINGDINO_ENDPOINT_KEY=
CLIP_ENDPOINT_URI=
CLIP_GROUNDINGDINO_ENDPOINT_URI=
CLIP_GROUNDINGDINO_ENDPOINT_KEY=

# Azure Storage (MediaRegistryService — upload endpoint)
AZURE_STORAGE_ACCOUNT_URL=
AZURE_CONTAINER_NAME=
UMRS_REDIS_URL=             # Redis URL for media registry (can be same as REDIS_URL)

# Video
STT_BACKEND=stub            # gemini | whisper_api | whisper_local | stub
VIDEO_SEGMENT_DURATION=5    # Seconds per HLS segment
VIDEO_MAX_DOWNLOAD_DURATION=600
VIDEO_DIR=../video/

# JWT (required — startup raises ValueError if missing unless CI=true or TESTING=true)
JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Observability (optional)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGSMITH_API_KEY=
```

## API Endpoints

- `POST /chat` — Main agent endpoint (`user_id`, `user_query`, optional `additional_material: dict`, `session_id`)
- AG-UI streaming endpoint — mounted at `/soccer_agent/copilotkit`; used by the frontend CopilotKit adapter
- `GET /` — Health check (shows DB connection status)
- `POST /hls/sessions` — Register HLS directory for streaming
- `GET /hls/{id}/playlist.m3u8` — Rewritten HLS playlist
- `GET /hls/{id}/{file}.ts` — Proxy TS segment bytes
- `POST /hls/{id}/analyze` — Enqueue Celery background analysis task
- `GET /hls/{id}/analyze/status` — Poll Celery task status
- `DELETE /hls/sessions/{id}` — Revoke Celery task (SIGTERM)
- `/upload` — Media upload router (Azure Blob via `MediaRegistryService`)
- `/user` — User CRUD

## HLS Video Processing Pipeline

```
POST /hls/sessions
  → Celery task: hls.process_video_session
      ├─ ffmpeg: video → HLS segments (seg000.ts, …) in temporary/hls_sessions/{game_id}/
      │   stderr drained in background thread to prevent pipe-buffer deadlock on Windows
      └─ MatchIndexingService.process_hls_session (concurrent with ffmpeg)
            → HLSSegmentWatcher polls playlist, yields segments as ffmpeg writes them
            → VideoStreamingProcessor.extract_frames_from_segment
                  → 3 frames/segment (start/mid/end for seg 0; mid/end for others)
            → VideoStreamingProcessor.extract_audio_from_segment (ffmpeg -vn → 16kHz WAV)
            → SpeechToTextService.transcribe (DashScope qwen3-asr-flash)
            → IncrementalSegmentIndexer.index_segment_frames  ─┐ → Qdrant hls_frame_index
            → IncrementalSegmentIndexer.index_transcript_chunk ─┘
```

**Qdrant `hls_frame_index` schema**: named vectors `dense_caption` (`text-embedding-v4` of VLM caption) + `dense_text` (`text-embedding-v4` of transcript) + `sparse` (BM25 of caption + transcript). All points carry `game_id` for filtering.

## Adding a Tool

1. Create `app/soccer_agent/toolbox/my_tool.py` as a `BaseTool` subclass.
2. Import and export it in `app/soccer_agent/toolbox/__init__.py`.
3. Add it to `self.tool_registry` in `SoccerAgent.__init__()` (`app/soccer_agent/agent.py`).

## Key Config Values (`app/config/config.py`)

- `SESSION_MEMORY_TOKEN_THRESHOLD = 12_600` — triggers Redis compression
- `SESSION_MEMORY_RECENT_KEEP = 5` — messages kept verbatim post-compression
- `PLANNING_CONFIDENCE_THRESHOLD = 0.5` — disabled; all chains dispatch regardless of confidence
- `QDRANT_SEARCH_SCORE_THRESHOLD = 0.5`
- `GUARDRAIL_RECENT_TURNS = 4` — trailing messages the guardrail classifier sees for context
- `GUARDRAIL_TIMEOUT_SECONDS = 10.0` — fail-open ceiling for the classifier
- `GUARDRAIL_REFUSAL_MESSAGE` — canned off-topic refusal string

## GitNexus — Code Intelligence

This project is indexed by GitNexus as **FinalProject**. Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

**Always do before editing:**
- Run `gitnexus_impact({target: "symbolName", direction: "upstream"})` — report blast radius, warn on HIGH/CRITICAL
- Run `gitnexus_detect_changes()` before committing — verify only expected symbols changed

**Never do:**
- Edit a function/class/method without first running `gitnexus_impact`
- Rename symbols with find-and-replace — use `gitnexus_rename` (call-graph aware)
- Commit without running `gitnexus_detect_changes()`

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/FinalProject/context` | Codebase overview, check index freshness |
| `gitnexus://repo/FinalProject/processes` | All execution flows |
| `gitnexus://repo/FinalProject/process/{name}` | Step-by-step execution trace |
