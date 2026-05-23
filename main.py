from contextlib import asynccontextmanager
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from copilotkit import LangGraphAGUIAgent
from langfuse import get_client
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

import pymongo
from pymongo.server_api import ServerApi
from dns import resolver
from qdrant_client import QdrantClient

# Import cấu hình và database
from app.config.settings import settings
from app.config.config import LOG_FORMAT, LOG_LEVEL
from sqlalchemy import text
from app.database.db import engine

# Import SoccerAgent (but don't initialize yet)
from app.soccer_agent.agent import SoccerAgent
from app.soccer_agent.memory.checkpointer import init_checkpointer, close_checkpointer
from app.services.media_registry import MediaRegistryService

# Chỉ import router chat và user
from app.api.chat import router as chat_router
from app.api.chat import get_copilotkit_router
from app.api.user import router as user_router
from app.api.upload import router as upload_router

# Cấu hình logging đơn giản thay vì structlog
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global variables for database connections
mongo_client = None
qdrant_client = None
postgres_connected = False

# Global variable for agent service
agent_service = None

# Langfuse callback handler for tracing
class InlineCallbackHandler(CallbackHandler):
    run_inline = True

# --- 1. Lifespan: Quản lý khởi động/tắt app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load heavy models on startup and cleanup on shutdown.
    This ensures models are loaded once when the app starts, not on each request.
    """
    global mongo_client, qdrant_client
    global agent_service
    
    logger.info("🚀 Starting FastAPI application...")
    logger.info("📦 Initializing database connections...")
    
    global postgres_connected
    try:
        # Initialize database connections
        logger.info("Initializing database connections...")
        
        # MongoDB connection
        mongo_srv = settings.MONGO_SRV
        if mongo_srv:
            resolver.default_resolver = resolver.Resolver(configure=False)
            resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']
            mongo_client = pymongo.MongoClient(mongo_srv, server_api=ServerApi('1'))
            # Test connection
            mongo_client[settings.SOCCER_DB_NAME].command('ping')
            logger.info("✅ MongoDB client initialized and connected successfully")
        else:
            logger.warning("⚠️ MongoDB SRV not configured, skipping MongoDB initialization")
        
        # Qdrant connection
        qdrant_url = settings.QDRANT_URL
        qdrant_api_key = settings.QDRANT_API_KEY
        if qdrant_url and qdrant_api_key:
            qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            logger.info("✅ Qdrant client initialized successfully")
        else:
            logger.warning("⚠️ Qdrant credentials not configured, skipping Qdrant initialization")

        # Postgres connection check
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            postgres_connected = True
            logger.info("✅ Postgres connected successfully")
        except Exception as e:
            postgres_connected = False
            logger.error(f"❌ Postgres connection failed: {e}")

        # Initialize LangGraph checkpointer (async Postgres pool)
        checkpointer = None
        if postgres_connected:
            try:
                checkpointer = await init_checkpointer()
                logger.info("✅ LangGraph checkpointer initialized")
            except Exception as e:
                logger.error(f"❌ Checkpointer init failed: {e}")

        # Initialize SoccerAgent AFTER all models and databases are loaded
        if settings.DASHSCOPE_API_KEY:
            logger.info("Initializing SoccerAgent...")
            agent_service = SoccerAgent(checkpointer=checkpointer)
            logger.info("✅ SoccerAgent initialized successfully")

            # Initialize MediaRegistryService
            media_service = MediaRegistryService(
                redis_url=settings.UMRS_REDIS_URL,
                azure_storage_url=settings.AZURE_STORAGE_ACCOUNT_URL,
                azure_container_name=settings.AZURE_CONTAINER_NAME
            )
            app.state.media_registry = media_service
            logger.info("✅ MediaRegistryService initialized successfully")

            # Create a RunnableConfig containing the Langfuse callback handler for tracing
            langfuse_handler = InlineCallbackHandler()
            config = RunnableConfig(
                callbacks=[langfuse_handler],
                configurable={"media_registry": media_service}
            )

            # Create LangGraph Endpoint for Copilotkit Integration
            app.include_router(
                get_copilotkit_router(
                    agent=LangGraphAGUIAgent(
                        name="SoccerAgent",
                        graph=agent_service.graph, 
                        config=config
                    )
                ),
                prefix="/soccer_agent/copilotkit",
                tags=["CopilotKit"]
            )
        else:
            logger.warning("⚠️ DASHSCOPE_API_KEY not set, skipping SoccerAgent initialization")
            agent_service = None

        logger.info("✅ All models, databases, and services initialized - Application ready!")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}", exc_info=True)
        raise
    
    # Yield control to the application
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down application...")
    logger.info("🧹 Cleaning up resources...")
    
    # Close database connections
    if mongo_client is not None:
        mongo_client.close()
        logger.info("MongoDB connection closed")
    
    if qdrant_client is not None:
        qdrant_client.close()
        logger.info("Qdrant connection closed")
    
    # Close checkpointer pool
    await close_checkpointer()

    # Clear agent service
    agent_service = None
    
    # Clear model instances to free memory
    mongo_client = None
    qdrant_client = None
    
    langfuse_client = get_client()
    langfuse_client.flush()
    logger.info("✅ Langfuse traces flushed successfully")

    logger.info("✅ Cleanup completed")

# --- 2. Khởi tạo FastAPI App ---
app = FastAPI(
    title="Soccer Knowledge Agent API",
    description="API for the LangGraph-based Soccer Agent for tool chain execution.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- 3. Cấu hình CORS ---
# Cho phép Frontend gọi vào API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --- 4. Đăng ký Router ---
app.include_router(chat_router, tags=["Soccer Chat Agent"])
app.include_router(user_router, prefix="/user", tags=["User"])
app.include_router(upload_router, tags=["Upload"])

# --- 5. Helper Functions để truy cập preloaded models ---
def get_mongo_client():
    """Get the preloaded MongoDB client."""
    return mongo_client

def get_qdrant_client():
    """Get the preloaded Qdrant client."""
    return qdrant_client

# --- 6. Root Endpoint (Health check) ---
@app.get("/")
def root():
    return {
        "message": "Soccer Agent API is running!", 
        "docs": "/docs",
        "databases_connected": {
            "mongodb": mongo_client is not None,
            "qdrant": qdrant_client is not None,
            "postgres": postgres_connected,
        }
    }

# --- 7. Chạy server ---
if __name__ == "__main__":

    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )