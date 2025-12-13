from transformers import CLIPProcessor, CLIPModel
import torch
from app.config.config import DEVICE
from PIL import Image

class CLIPHelper:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch32"):
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def get_unit_per_image(self, images: Image.Image , texts: list[str]):
        """
        Tính toán vector embedding cho hình ảnh và văn bản.
        Trả về unit vectors để dễ dàng tính cosine similarity.
        """
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            similarity_outputs = self.model(**inputs).logits_per_image.item()

        return similarity_outputs