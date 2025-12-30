from contextlib import asynccontextmanager
import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, infer_device
from transformers import CLIPProcessor, CLIPModel
from deepface.modules import modeling

import pymongo
from pymongo.server_api import ServerApi
from dns import resolver
from qdrant_client import QdrantClient

# Import cấu hình và database
from app.config.settings import settings
from app.config.config import MODEL_SEGMENT, TEMPORARY_DIR

# Import SoccerAgent (but don't initialize yet)
from app.soccer_agent.agent import SoccerAgent

# Chỉ import router chat
from app.api.chat import router as chat_router

# Cấu hình logging đơn giản thay vì structlog
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables to store preloaded models
segment_processor = None
segment_model = None
segment_device = None
deepface_recognition_model = None
deepface_detector_model = None
clip_model = None
clip_processor = None

# Global variables for database connections
mongo_client = None
qdrant_client = None

# Global variable for agent service
agent_service = None

# --- 1. Lifespan: Quản lý khởi động/tắt app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load heavy models on startup and cleanup on shutdown.
    This ensures models are loaded once when the app starts, not on each request.
    """
    global segment_processor, segment_model, segment_device
    global deepface_recognition_model, deepface_detector_model
    global clip_model, clip_processor
    global mongo_client, qdrant_client
    global agent_service
    
    logger.info("🚀 Starting FastAPI application...")
    logger.info("📦 Loading heavy models and initializing database connections...")
    
    try:
        # Preload Segment models (GroundingDINO)
        logger.info("Loading Segment models (GroundingDINO)...")
        segment_device = infer_device()
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        token_kwargs = {"token": hf_token} if hf_token else {}
        
        segment_processor = AutoProcessor.from_pretrained(
            MODEL_SEGMENT, 
            cache_dir=TEMPORARY_DIR, 
            **token_kwargs
        )
        segment_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_SEGMENT, 
            cache_dir=TEMPORARY_DIR, 
            **token_kwargs
        ).to(segment_device)
        logger.info("✅ Segment models (GroundingDINO) loaded successfully")
        
        # Preload Entity Recognition models (DeepFace)
        logger.info("Loading Entity Recognition models (DeepFace)...")
        deepface_recognition_model = modeling.build_model(
            task="facial_recognition", 
            model_name="Facenet512"
        )
        deepface_detector_model = modeling.build_model(
            task="face_detector", 
            model_name="retinaface"
        )
        logger.info("✅ Entity Recognition models (DeepFace) loaded successfully")
        
        # Preload CLIP models for frame selection
        logger.info("Loading CLIP models...")
        clip_model_name = "openai/clip-vit-large-patch14"
        clip_processor = CLIPProcessor.from_pretrained(
            clip_model_name, 
            cache_dir=TEMPORARY_DIR, 
            **token_kwargs
        )
        clip_model = CLIPModel.from_pretrained(
            clip_model_name, 
            cache_dir=TEMPORARY_DIR, 
            **token_kwargs
        ).to(segment_device)
        clip_model.eval()
        logger.info("✅ CLIP models loaded successfully")
        
        # Initialize database connections
        logger.info("Initializing database connections...")
        
        # MongoDB connection
        mongo_srv = settings.MONGO_SRV
        if mongo_srv:
            resolver.default_resolver = resolver.Resolver(configure=False)
            resolver.default_resolver.nameservers = ['8.8.8.8', '1.1.1.1']
            mongo_client = pymongo.MongoClient(mongo_srv, server_api=ServerApi('1'))
            # Test connection
            mongo_client.admin.command('ping')
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
        
        # Initialize SoccerAgent AFTER all models and databases are loaded
        logger.info("Initializing SoccerAgent...")
        agent_service = SoccerAgent()
        logger.info("✅ SoccerAgent initialized successfully")
        
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
    
    # Clear agent service
    agent_service = None
    
    # Clear model instances to free memory
    segment_processor = None
    segment_model = None
    segment_device = None
    deepface_recognition_model = None
    deepface_detector_model = None
    clip_model = None
    clip_processor = None
    mongo_client = None
    qdrant_client = None
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
# Chỉ đăng ký duy nhất chat router
app.include_router(chat_router, tags=["Soccer Chat Agent"])

# --- 5. Helper Functions để truy cập preloaded models ---
def get_segment_models():
    """Get the preloaded segment models (processor, model, device)."""
    return {
        "processor": segment_processor,
        "model": segment_model,
        "device": segment_device
    }

def get_deepface_models():
    """Get the preloaded DeepFace models (recognition and detector)."""
    return {
        "recognition": deepface_recognition_model,
        "detector": deepface_detector_model
    }

def get_clip_models():
    """Get the preloaded CLIP models (processor and model)."""
    return {
        "processor": clip_processor,
        "model": clip_model
    }

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
        "models_loaded": {
            "segment_processor": segment_processor is not None,
            "segment_model": segment_model is not None,
            "deepface_recognition": deepface_recognition_model is not None,
            "deepface_detector": deepface_detector_model is not None,
            "clip_processor": clip_processor is not None,
            "clip_model": clip_model is not None,
        },
        "databases_connected": {
            "mongodb": mongo_client is not None,
            "qdrant": qdrant_client is not None,
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