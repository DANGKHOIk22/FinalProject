from transformers import CLIPProcessor, CLIPModel
import torch
import os
from app.config.config import DEVICE,TEMPORARY_DIR
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

class CLIPHelper:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        # Try to use preloaded models from main.py
        try:
            import main
            if main.clip_model is not None and main.clip_processor is not None:
                self.model = main.clip_model
                self.processor = main.clip_processor
                print("✅ Using preloaded CLIP models from lifespan")
                return
        except (ImportError, AttributeError):
            pass
        
        # Fallback: Load models if not preloaded
        print("Loading CLIP models on demand...")
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        token_kwargs = {"token": hf_token} if hf_token else {}
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=TEMPORARY_DIR, **token_kwargs).to(DEVICE)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=TEMPORARY_DIR, **token_kwargs)
        print("✅ CLIP models loaded successfully")

    def get_unit_per_image(self, images: Image.Image , texts: list[str]):
        """
        Tính toán vector embedding cho hình ảnh và văn bản.
        Trả về unit vectors để dễ dàng tính cosine similarity.
        """
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            similarity_outputs = self.model(**inputs).logits_per_image.item()

        return similarity_outputs