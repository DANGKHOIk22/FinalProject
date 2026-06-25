import base64
import json
import logging
import re as _re
from collections import defaultdict
from io import BytesIO
from typing import Annotated, Any, Dict, List, Literal, Optional, Type

import cv2
import numpy as np
import requests
from langsmith import get_current_run_tree
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import ToolRuntime
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from langfuse import get_client

from app.config.config import QDRANT_SEARCH_SCORE_THRESHOLD as THRESHOLD
from app.config.settings import settings
from app.services.qdrant_service import qdrant_service
from app.soccer_agent.factory.llm_provider import get_llm, warm_llm
from app.soccer_agent.prompts.toolbox.entity_recognition import get_entity_disambiguation_prompt_template
from app.soccer_agent.toolbox._config_loader import tool_description

logger = logging.getLogger(__name__)

_QWEN_VL_MODEL = "qwen3-vl-flash-2025-10-15"
_LOCALIZE_SYSTEM_PROMPT = (
    "You are a helpful assistant to detect objects in images. "
    "When asked to detect elements based on a description, "
    'return valid JSON: [{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "placeholder"}]. '
    "Return ONLY ONE bounding box for the single most prominent person matching the description."
)


class EntityRecognitionInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_entity_recognition_task: str = Field(
        ...,
        description=(
            "Visual description of the single entity to locate and identify. Rules:\n"
            "1. MANDATORY OBJECT CLASS: Include the noun (e.g. 'person', 'man', 'woman'). Never output an adjective alone.\n"
            "2. VISUAL ATTRIBUTES ONLY: Include color, clothing, position (e.g. 'wearing a white shirt on the left').\n"
            "3. REMOVE NAMED ENTITIES: Remove proper names — the localization model does not recognize names.\n"
            "4. SINGLE OBJECT: Describe exactly one entity per tool call. For multiple, use parallel chains.\n"
            "5. ENGLISH ONLY."
        ),
        examples=[
            "the person wearing a white shirt on the left",
            "the person wearing number 10",
            "the person in black uniform",
        ],
    )
    image_id: str = Field(..., description="UUID of the image to search in")
    runtime: Annotated[Optional[ToolRuntime], InjectedToolArg] = Field(default=None)


class EntityRecognitionTool(BaseTool):
    name: str = "entity_recognition"
    description: str = ""
    response_format: Literal["content"] = "content"
    args_schema: Type[BaseModel] = EntityRecognitionInput

    _vl_client: Any = PrivateAttr(default=None)
    _disambiguation_llm: Any = PrivateAttr(default=None)
    _insight_endpoint_uri: str = PrivateAttr(default="")
    _insight_payload_header: Dict[str, str] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(description=tool_description("entity_recognition"), **data)
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        # Qwen-VL via DashScope (used for entity localization)
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY must be configured")
        self._vl_client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        # Qdrant access goes through the shared qdrant_service singleton.

        # InsightFace Azure endpoint
        uri = settings.INSIGHTFACE_ENDPOINT_URI or ""
        key = settings.INSIGHTFACE_ENDPOINT_KEY or ""
        if not uri:
            raise ValueError("INSIGHTFACE_ENDPOINT_URI is not configured")
        if not key:
            raise ValueError("INSIGHTFACE_ENDPOINT_KEY is not configured")
        self._insight_endpoint_uri = uri
        self._insight_payload_header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        probe = requests.post(url=uri, headers=self._insight_payload_header, timeout=60)
        if probe.status_code != 200:
            raise ConnectionError(
                f"InsightFace endpoint unreachable: {probe.status_code} — {probe.text}"
            )

        self._disambiguation_llm = get_llm("tool")
        logger.info("✅ EntityRecognitionTool clients initialized")

    async def warmup(self) -> None:
        import asyncio
        try:
            await asyncio.to_thread(
                self._vl_client.chat.completions.create,
                model=_QWEN_VL_MODEL,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": _LOCALIZE_SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": "warmup"}]},
                ],
                max_tokens=1,
                extra_headers={"X-DashScope-WorkSpace": ""},
            )
            logger.info("✅ entity_recognition Qwen-VL warmed up")
        except Exception as e:
            logger.warning(f"⚠️ entity_recognition Qwen-VL warmup failed (non-fatal): {e}")

        system_msg = get_entity_disambiguation_prompt_template().messages[0]
        await warm_llm(self._disambiguation_llm, system_msg, "entity_recognition:disambiguation")

    # ------------------------------------------------------------------
    # Step 1: Localize entity in image with Qwen-VL, refine with OpenCV
    # ------------------------------------------------------------------

    def _localize_entity(
        self,
        image_id: str,
        description: str,
        media_registry: Any,
        user_id: str,
        thread_id: str,
    ) -> Optional[Image.Image]:
        """Use Qwen-VL to locate the described entity; return a face-refined crop."""
        img = media_registry.get_pil_image(user_id, thread_id, image_id)
        if not img:
            raise ValueError(f"Could not load image {image_id}")
        width, height = img.size

        sas_url = media_registry.get_sas_url(user_id, thread_id, image_id)
        if sas_url:
            image_content = {"url": sas_url}
        else:
            image_b64 = media_registry.get_base_64(user_id, thread_id, image_id)
            image_content = {"url": f"data:image/jpeg;base64,{image_b64}"}

        user_prompt = (
            "Detect ONLY ONE bounding box for the single most prominent person "
            f"that best matches the description. Description: {description}"
        )

        response = self._vl_client.chat.completions.create(
            model=_QWEN_VL_MODEL,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": _LOCALIZE_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image_content},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            temperature=0.1,
            top_p=0.1,
            extra_headers={"X-DashScope-WorkSpace": ""},
        )

        content_str = response.choices[0].message.content.strip()
        content_str = _re.sub(r"^```[a-z]*\n?", "", content_str)
        content_str = _re.sub(r"\n?```$", "", content_str)
        # VLM output is free text — a refusal or error payload must read as
        # "nothing localized", not escalate into a caching-eligible error string.
        try:
            raw_items = json.loads(content_str.strip())
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"[entity_recognition] Qwen-VL returned non-JSON: {content_str[:200]!r}")
            return None
        if not isinstance(raw_items, list):
            raw_items = [raw_items]

        if not raw_items or not isinstance(raw_items[0], dict):
            return None

        bbox = raw_items[0].get("bbox_2d")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(v, (int, float)) for v in bbox)
        ):
            logger.warning(f"[entity_recognition] Qwen-VL returned invalid bbox: {bbox!r}")
            return None

        # Qwen-VL normalises coordinates to 0-1000
        x_min = (bbox[0] / 1000.0) * width
        y_min = (bbox[1] / 1000.0) * height
        x_max = (bbox[2] / 1000.0) * width
        y_max = (bbox[3] / 1000.0) * height

        coarse_crop = img.crop((x_min, y_min, x_max, y_max))
        return self._refine_to_face(coarse_crop)

    def _refine_to_face(self, pil_image: Image.Image, padding: float = 0.20) -> Image.Image:
        """Refine a coarse Qwen-VL crop to the largest detected face. Falls back to the full crop."""
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )

        if len(faces) == 0:
            return pil_image

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        img_w, img_h = pil_image.size
        pad_x, pad_y = int(w * padding), int(h * padding)
        return pil_image.crop((
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(img_w, x + w + pad_x),
            min(img_h, y + h + pad_y),
        ))

    # ------------------------------------------------------------------
    # Step 2: Get face embeddings via InsightFace
    # ------------------------------------------------------------------

    def _get_face_embeddings(self, pil_image: Image.Image) -> List[List[float]]:
        """Send the cropped face image to InsightFace and return the embedding vectors."""
        buf = BytesIO()
        pil_image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {"image": image_b64, "max_faces": 5, "confidence_threshold": 0.60}
        response = requests.post(
            url=self._insight_endpoint_uri,
            headers=self._insight_payload_header,
            json=payload,
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"InsightFace endpoint failed: {response.status_code} — {response.text}"
            )
        data = response.json()
        if not data.get("success"):
            raise RuntimeError(f"InsightFace error: {data.get('error', 'unknown')}")

        return [
            face["embedding"]
            for face in data.get("result", [])
            if face.get("embedding")
        ]

    # ------------------------------------------------------------------
    # Step 3: Search Qdrant with embedding voting + re-ranking
    # ------------------------------------------------------------------

    def _search_qdrant(self, embeddings: List[List[float]]) -> List[tuple]:
        """Search Qdrant for soccer entities matching the given face embeddings.

        Returns (name, composite_score) tuples sorted descending, deduplicated
        across embeddings (best score per unique name is kept).
        """
        collection_name = settings.QDRANT_COLLECTION_NAME
        assert collection_name, "QDRANT_COLLECTION_NAME is not configured"

        best_per_name: Dict[str, float] = {}

        for idx, embedding in enumerate(embeddings):
            points = qdrant_service.search(
                collection_name=collection_name,
                vector=embedding,
                limit=10,
                score_threshold=0.2,
                with_vectors=True,
            )

            candidates: Dict[str, Any] = defaultdict(
                lambda: {"max_score": 0, "count": 0, "num_faces": 1}
            )
            for point in points:
                name = point.payload["NAME"]
                if not point.vector:
                    continue
                match_count = sum(
                    1 for sub_vec in point.vector
                    if float(
                        np.dot(embedding, sub_vec)
                        / (np.linalg.norm(embedding) * np.linalg.norm(sub_vec))
                    ) >= THRESHOLD
                )
                candidates[name]["count"] = match_count
                candidates[name]["num_faces"] = point.payload.get("num_faces", 1)
                candidates[name]["max_score"] = point.score

            ranked = sorted(
                [
                    (name, (0.55 * d["max_score"]) + (0.45 * d["count"] / d["num_faces"]))
                    for name, d in candidates.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            for name, score in ranked:
                if name not in best_per_name or score > best_per_name[name]:
                    best_per_name[name] = score

            if ranked:
                logger.info(f"Face {idx + 1}: top match → {ranked[0][0]} ({ranked[0][1]:.3f})")

        return sorted(best_per_name.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Step 4: VLM disambiguation — pick the correct identity from candidates
    # ------------------------------------------------------------------

    def _disambiguate_entity(self, crop: Image.Image, candidate_names: List[str]) -> str:
        """Feed the cropped face + close candidate names to a VLM to confirm identity."""
        buf = BytesIO()
        crop.save(buf, format="JPEG")
        crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = get_entity_disambiguation_prompt_template()
        messages = prompt.format_messages(
            crop_b64=crop_b64,
            candidate_names="\n".join(candidate_names),
        )
        response = self._disambiguation_llm.invoke(messages)
        return response.content.strip()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _run(
        self,
        query_entity_recognition_task: str,
        image_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        runtime: Optional[ToolRuntime] = None,
    ) -> str:
        run_tree = get_current_run_tree()
        try:
            user_id, thread_id, media_registry = "default_user", "default_thread", None
            if runtime and runtime.config:
                cfg = runtime.config.get("configurable", {})
                user_id = str(cfg.get("user_id", user_id))
                thread_id = str(cfg.get("thread_id", thread_id))
                media_registry = cfg.get("media_registry")
            if not media_registry:
                raise ValueError("MediaRegistryService not found in runtime config")

            langfuse = get_client()
            with langfuse.start_as_current_observation(
                as_type="chain", 
                name="qwenvl-localize-entity") as observation:
                # 1. Localize entity via Qwen-VL + OpenCV face refinement
                crop = self._localize_entity(
                    image_id, query_entity_recognition_task, media_registry, user_id, thread_id
                )
                if crop is None:
                    return (
                        "Could not locate the described entity in the image. "
                        "Try rephrasing the visual description."
                    )

            with langfuse.start_as_current_observation(
                as_type="chain", 
                name="insightface-embedding") as observation:
                # 2. Get face embeddings from crop via InsightFace
                embeddings = self._get_face_embeddings(crop)
                if not embeddings:
                    return (
                        "No faces detected in the localized region. "
                        "Try rephrasing the visual description."
                    )
            
            with langfuse.start_as_current_observation(
                as_type="chain",
                name="search-qdrant") as observation:
                # 3. Search Qdrant — (name, score) sorted desc, deduped across embeddings
                ranked_candidates = self._search_qdrant(embeddings)
                if not ranked_candidates:
                    return "No matching entity found."

            top_score = ranked_candidates[0][1]
            close_candidates = [
                name for name, score in ranked_candidates
                if top_score - score < 0.07
            ]

            if len(close_candidates) == 1:
                logger.info(
                    "InsightFace confident enough; recognized '%s' with score %.3f",
                    close_candidates[0],
                    top_score,
                )
    
                # InsightFace is confident — no VLM needed
                return f"Recognized: {close_candidates[0]}."

            with langfuse.start_as_current_observation(
                as_type="chain",
                name="vlm-disambiguation") as observation:
                logger.info(
                    "Multiple close candidates detected; invoking VLM disambiguation: %s",
                    close_candidates,
                )
                # 4. Multiple candidates within 0.07 of top score — VLM breaks the tie
                identified_name = self._disambiguate_entity(crop, close_candidates)

            if identified_name.lower() == "unknown":
                logger.info("Could not confidently identify the entity.")
                return "Could not confidently identify the entity."
            return f"Recognized: {identified_name}."

        except Exception as e:
            error_msg = f"Error in entity_recognition: {e}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return f"An error occurred. Try rephrasing the description. Error: {error_msg}"
