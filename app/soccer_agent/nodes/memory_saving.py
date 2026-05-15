import logging
import re
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import pymongo
from dns import resolver
from pymongo.server_api import ServerApi
from langchain_core.messages import ToolMessage, ToolCall, HumanMessage, AIMessage
from langgraph.graph.state import RunnableConfig

from app.config import settings
from app.schema.soccer_agent.state import AgentState
from app.soccer_agent.memory.chat_history import get_postgres_memory
from app.soccer_agent.memory.conversation_memory import CustomSystemPromptMemory
from app.soccer_agent.memory.long_term_memory import long_term_memory_manager
from app.soccer_agent.services.content_cleaner import extract_summary, strip_summary

logger = logging.getLogger(__name__)

_ENTITY_URL_FIELD = {
    "player": "PLAYER_URL",
    "team": "TEAM_URL",
    "venue": "VENUE_URL",
    "referee": "REFEREE_URL",
}


class SaveToMemoryNode:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 1. Chat History (Short-term)
    # ------------------------------------------------------------------

    async def _background_save_memory(
        self, session_id: str, user_query: str, final_response: str
    ) -> None:
        """Save clean conversation context in background."""
        try:
            mem, conn, pool = get_postgres_memory(session_id)
            mem_obj = CustomSystemPromptMemory(
                memory_key="history",
                chat_memory=mem.chat_memory,
                return_messages=True,
                max_history=15,
            )
            await asyncio.to_thread(
                mem_obj.save_context,
                {"input": user_query},
                {"output": final_response},
            )
            if conn and pool:
                conn.commit()
                pool.putconn(conn)
            logger.debug(f"[BackgroundSave] Successfully saved chat turn for session {session_id}")
        except Exception as e:
            logger.warning(f"[BackgroundSave] Failed for session {session_id}: {e}")

    # ------------------------------------------------------------------
    # 2. MongoDB Upsert (Entity Knowledge Base)
    # ------------------------------------------------------------------

    async def _background_upsert_mongo(self, payload: Dict[str, Any]) -> None:
        """
        Upsert entity into MongoDB.
        - If entity exists in DB (is_missing=False): Force-replace content fields.
        - If entity is new (is_missing=True): Insert new document.
        - entity_type rules:
            - If DB already has entity_type → keep it (no override).
            - If DB has "unknown" → allow LLM to override.
            - If new entity → use LLM classification or "unknown".
        """
        name = payload.get("name", "")
        entity_type = payload.get("entity_type", "unknown")
        is_missing = payload.get("is_missing", True)
        wiki_url = payload.get("wiki_url", "")
        cleaned = payload.get("cleaned_content", "")
        images = payload.get("images", [])
        db_entity_data = payload.get("db_entity_data")  # dict from model_dump()

        if not name or not cleaned:
            return

        try:
            try:
                import main
                client = main.mongo_client if hasattr(main, "mongo_client") and main.mongo_client else None
            except (ImportError, AttributeError):
                client = None

            if client is None:
                resolver.default_resolver = resolver.Resolver(configure=False)
                resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
                client = pymongo.MongoClient(settings.MONGO_SRV, server_api=ServerApi("1"))

            collection = client[settings.SOCCER_DB_NAME][settings.SOCCER_COLLECTION_NAME]

            if not is_missing and db_entity_data is not None:
                # Force-replace content fields, keep entity_type from DB
                doc = {k: v for k, v in db_entity_data.items() if k != "_id" and v is not None}
                doc.pop("INFOBOX", None)
                doc["SUMMARY"] = extract_summary(cleaned)
                doc["CONTENT"] = strip_summary(cleaned)
                doc["IMAGES"] = images
                doc["LAST_UPDATED"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # entity_type override logic:
                # Only allow LLM to override if current DB value is "unknown"
                existing_type = doc.get("ENTITY_TYPE", "").lower()
                if existing_type == "unknown" and entity_type != "unknown":
                    doc["ENTITY_TYPE"] = entity_type
                # Otherwise, keep existing entity_type from DB

                collection.replace_one({"NAME": {"$regex": f"^{re.escape(name)}$", "$options": "i"}}, doc)
                logger.info(f"[SaveMemory] Force-replaced Mongo entity: {name}")
            else:
                # Insert new entity
                if entity_type not in _ENTITY_URL_FIELD:
                    logger.warning(f"[SaveMemory] entity_type '{entity_type}' for '{name}' not in URL map, storing as-is")
                
                url_field = _ENTITY_URL_FIELD.get(entity_type, "")
                doc = {
                    "NAME": name,
                    "ENTITY_TYPE": entity_type,
                    "SUMMARY": extract_summary(cleaned),
                    "CONTENT": strip_summary(cleaned),
                    "IMAGES": images,
                    "LAST_UPDATED": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                if url_field and wiki_url:
                    doc[url_field] = wiki_url
                
                collection.insert_one(doc)
                logger.info(f"[SaveMemory] Inserted new Mongo entity: {name} ({entity_type})")

        except Exception as e:
            logger.error(f"[SaveMemory] Mongo upsert failed for '{name}': {e}")

    # ------------------------------------------------------------------
    # 3. Long-term Memory (pgvector deduplication)
    # ------------------------------------------------------------------

    async def _background_upsert_longterm(self, thread_id: str, payload: Dict[str, Any]) -> None:
        """Sync entity wiki content to Long-term Memory with smart deduplication."""
        name = payload.get("name", "")
        cleaned = payload.get("cleaned_content", "")
        wiki_url = payload.get("wiki_url", "")
        entity_type = payload.get("entity_type", "unknown")

        if not cleaned or len(cleaned.strip()) < 50:
            return

        try:
            await long_term_memory_manager.upsert_memory_smart(
                user_id=thread_id,
                entity_name=name,
                content=cleaned,
                metadata={
                    "source": "wiki_extract",
                    "wiki_url": wiki_url,
                    "entity_type": entity_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"[SaveMemory] pgvector upsert failed for '{name}': {e}")

    # ------------------------------------------------------------------
    # Main Node Entry Point
    # ------------------------------------------------------------------

    async def save_to_memory_node(self, state: AgentState, config: RunnableConfig) -> dict:
        """
        - Saves clean chat history (Human vs AI) to Postgres conversation memory.
        - Processes entity_augment artifacts for Mongo + pgvector upsert.
        - Syncs news results to Long-term Memory.
        All operations run in background to avoid blocking.
        """
        messages = state.get("messages", [])
        tool_calls_history = state.get("tool_calls_history", [])
        tool_results_history = state.get("tool_results_history", [])

        metadata = config.get("metadata", {})
        thread_id = metadata.get("thread_id", str(uuid.uuid4()))

        # --- 1. Save Clean Conversation History ---
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break

        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                last_ai_message = msg
                break

        if last_user_message and last_ai_message:
            asyncio.create_task(
                self._background_save_memory(
                    session_id=thread_id,
                    user_query=last_user_message.content,
                    final_response=last_ai_message.content
                )
            )

        # --- 2. Process Tool Results ---
        if tool_calls_history:
            for i, tool_call in enumerate(tool_calls_history):
                tool_name = tool_call.get("name")
                if i >= len(tool_results_history):
                    continue

                tool_msg = tool_results_history[i]

                # Case A: entity_augment — check for _upsert_payload in artifact
                if tool_name == "entity_augment":
                    artifact = getattr(tool_msg, "artifact", None)
                    if artifact and hasattr(artifact, "_upsert_payload") and artifact._upsert_payload:
                        payload = artifact._upsert_payload
                        
                        # Only upsert Mongo when we have fresh wiki data
                        if payload.get("source") == "wiki_extract":
                            if artifact.found_entities:
                                payload["db_entity_data"] = artifact.found_entities[0].model_dump()
                            asyncio.create_task(self._background_upsert_mongo(payload))
                        
                        # Always upsert pgvector (with deduplication)
                        asyncio.create_task(self._background_upsert_longterm(thread_id, payload))

                # Case B: web_news_search
                elif tool_name == "web_news_search":
                    news_results = getattr(tool_msg, "artifact", [])
                    if isinstance(news_results, list):
                        for res in news_results:
                            if isinstance(res, dict) and res.get("content"):
                                asyncio.create_task(
                                    long_term_memory_manager.upsert_memory_smart(
                                        user_id=thread_id,
                                        entity_name="news",
                                        content=f"Title: {res.get('title', 'N/A')}\nContent: {res.get('content')}",
                                        metadata={
                                            "source": "web_news_search",
                                            "url": res.get("url"),
                                            "timestamp": datetime.now().isoformat()
                                        }
                                    )
                                )

        return {}
