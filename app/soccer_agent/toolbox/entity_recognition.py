import base64
import json
import logging
import re as _re
from collections import defaultdict
from io import BytesIO
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Type

import cv2
import numpy as np
import pymongo
import requests
from dns import resolver
from langsmith import get_current_run_tree
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import ToolRuntime
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pymongo.server_api import ServerApi
from qdrant_client import QdrantClient

from app.cache.standard_cache import standard_cache
from app.config.config import QDRANT_SEARCH_SCORE_THRESHOLD as THRESHOLD
from app.config.settings import settings
from app.schema.soccerwiki_entities import PlayerSchema, RefereeSchema, TeamSchema, VenueSchema
from app.schema.textual_entity_search import SearchingResult
from app.soccer_agent.toolbox._config_loader import tool_description

logger = logging.getLogger(__name__)


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
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    args_schema: Type[BaseModel] = EntityRecognitionInput

    _vl_client: Any = PrivateAttr(default=None)
    _qdrant_client: Optional[QdrantClient] = PrivateAttr(default=None)
    _mongo_client: Optional[pymongo.MongoClient] = PrivateAttr(default=None)
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

        # Qdrant + MongoDB — prefer preloaded clients from lifespan
        try:
            import main
            if main.qdrant_client is not None:
                self._qdrant_client = main.qdrant_client
            if main.mongo_client is not None:
                self._mongo_client = main.mongo_client
        except (ImportError, AttributeError):
            pass

        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=True,
                check_compatibility=False,
                timeout=20,
            )

        if self._mongo_client is None and settings.MONGO_SRV:
            resolver.default_resolver = resolver.Resolver(configure=False)
            resolver.default_resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
            self._mongo_client = pymongo.MongoClient(
                settings.MONGO_SRV, server_api=ServerApi("1")
            )

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
        logger.info("✅ EntityRecognitionTool clients initialized")

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

        system_prompt = (
            "You are a helpful assistant to detect objects in images. "
            "When asked to detect elements based on a description, "
            'return valid JSON: [{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "placeholder"}]. '
            "Return ONLY ONE bounding box for the single most prominent person matching the description."
        )
        user_prompt = (
            "Detect ONLY ONE bounding box for the single most prominent person "
            f"that best matches the description. Description: {description}"
        )

        response = self._vl_client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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
        raw_items = json.loads(content_str.strip())
        if not isinstance(raw_items, list):
            raw_items = [raw_items]

        if not raw_items:
            return None

        bbox = raw_items[0].get("bbox_2d")
        if not bbox or len(bbox) != 4:
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

    def _search_qdrant(self, embeddings: List[List[float]]) -> List[Dict]:
        """Search Qdrant for soccer entities matching the given face embeddings."""
        collection_name = settings.QDRANT_COLLECTION_NAME
        assert collection_name and self._qdrant_client, "Qdrant client is not initialized"

        soccer_entities: List[Dict] = []
        for idx, embedding in enumerate(embeddings):
            search_result = self._qdrant_client.query_points(
                collection_name=collection_name,
                query=embedding,
                limit=7,
                score_threshold=0.5,
                with_vectors=True,
            )

            candidates: Dict[str, Any] = defaultdict(
                lambda: {"ENTITY_TYPE": None, "max_score": 0, "count": 0, "score_list": []}
            )
            for point in search_result.points:
                name = point.payload["NAME"]
                if not point.vector:
                    continue
                match_count = 0
                for sub_vec in point.vector:
                    score = float(
                        np.dot(embedding, sub_vec)
                        / (np.linalg.norm(embedding) * np.linalg.norm(sub_vec))
                    )
                    if score >= THRESHOLD:
                        candidates[name]["score_list"].append(score)
                        match_count += 1
                candidates[name]["count"] = match_count
                candidates[name]["ENTITY_TYPE"] = point.payload.get("ENTITY_TYPE")
                candidates[name]["max_score"] = point.score

            ranked = sorted(
                [
                    (
                        name,
                        (0.55 * d["max_score"])
                        + (0.45 * d["count"] / 20)
                        + (0.1 * (np.mean(d["score_list"]) if d["score_list"] else 0)),
                        d,
                    )
                    for name, d in candidates.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            if ranked:
                best_name, _, best_data = ranked[0]
                soccer_entities.append(
                    {"ENTITY_TYPE": best_data["ENTITY_TYPE"], "NAME": best_name}
                )
                logger.info(f"Face {idx + 1}: matched → {best_name}")

        return soccer_entities

    # ------------------------------------------------------------------
    # Step 4: Fetch detailed entity info from MongoDB
    # ------------------------------------------------------------------

    @standard_cache.cache(ttl=60 * 60, validatedModel=SearchingResult)
    def _query_database(self, soccer_entities: List[Dict]) -> SearchingResult:
        """Look up detailed entity information in MongoDB."""
        result = SearchingResult()
        if not self._mongo_client:
            logger.error("MongoDB client not initialized")
            return result

        schema_map = {
            "venue": VenueSchema,
            "player": PlayerSchema,
            "team": TeamSchema,
            "referee": RefereeSchema,
        }
        try:
            collection = self._mongo_client.get_database(settings.SOCCER_DB_NAME).get_collection(
                settings.SOCCER_COLLECTION_NAME
            )
            for entity in soccer_entities:
                entity_data = collection.find_one(
                    {"$and": [{"ENTITY_TYPE": entity["ENTITY_TYPE"]}, {"NAME": entity["NAME"]}]}
                )
                if entity_data:
                    entity_data["_id"] = str(entity_data["_id"])
                    schema_cls = schema_map.get(entity["ENTITY_TYPE"])
                    if schema_cls:
                        try:
                            result.found_entities.append(schema_cls(**entity_data))
                            logger.info(f"✅ Found {entity['ENTITY_TYPE']}: {entity['NAME']}")
                        except Exception as e:
                            logger.warning(f"Failed to parse {entity['NAME']}: {e}")
                            result.missing_entities.append(entity["NAME"])
                    else:
                        result.missing_entities.append(entity["NAME"])
                else:
                    result.missing_entities.append(entity["NAME"])
        except Exception as e:
            raise RuntimeError(f"Database query error: {e}") from e

        return result

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _run(
        self,
        query_entity_recognition_task: str,
        image_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        runtime: Optional[ToolRuntime] = None,
    ) -> Tuple[str, SearchingResult]:
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

            # 1. Localize entity via Qwen-VL + OpenCV face refinement
            crop = self._localize_entity(
                image_id, query_entity_recognition_task, media_registry, user_id, thread_id
            )
            if crop is None:
                return (
                    "Could not locate the described entity in the image. "
                    "Try rephrasing the visual description.",
                    SearchingResult(),
                )

            # 2. Get face embeddings from crop via InsightFace
            embeddings = self._get_face_embeddings(crop)
            if not embeddings:
                return (
                    "No faces detected in the localized region. "
                    "Try rephrasing the visual description.",
                    SearchingResult(),
                )

            # 3. Search Qdrant
            entities = self._search_qdrant(embeddings)
            if not entities:
                return "No matching entities found in the database.", SearchingResult()

            # 4. Fetch detailed info from MongoDB
            db_result = self._query_database(entities)

            parts = []
            if db_result.found_entities:
                found = ", ".join(e.NAME for e in db_result.found_entities)
                parts.append(f"Found: {found}.")
            if db_result.missing_entities:
                missing = ", ".join(db_result.missing_entities)
                parts.append(f"Not in database: {missing}.")
            return "Successfully identified soccer entity. " + " ".join(parts), db_result

        except Exception as e:
            error_msg = f"Error in entity_recognition: {e}"
            logger.error(error_msg, exc_info=True)
            if run_tree:
                run_tree.end(error=error_msg)
            return (
                f"An error occurred. Try rephrasing the description. Error: {error_msg}",
                SearchingResult(),
            )
