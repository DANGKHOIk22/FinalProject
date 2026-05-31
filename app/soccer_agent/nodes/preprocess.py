import logging
from urllib.parse import urlparse
from app.schema.soccer_agent.state import AgentState

logger = logging.getLogger(__name__)

def parse_sas_url(sas_url: str) -> dict | None:
    """
    Extracts metadata from an Azure Blob SAS URL.
    Expected format: https://.../<container>/sessions/<thread_id>/<uuid>.<extension>?<sas_token>
    Or: sessions/<thread_id>/<uuid>.<extension>
    """
    try:
        parsed = urlparse(sas_url)
        path = parsed.path.strip("/")
        parts = path.split("/")
        
        if "sessions" in parts:
            idx = parts.index("sessions")
            if len(parts) > idx + 2:
                thread_id = parts[idx + 1]
                filename = parts[idx + 2]
                uuid = filename.split(".")[0]
                blob_path = "/".join(parts[idx:])
                return {
                    "uuid": uuid,
                    "blob_path": blob_path,
                    "thread_id": thread_id,
                    "filename": filename
                }
    except Exception as e:
        logger.error(f"Error parsing SAS URL '{sas_url[:50]}...': {e}")
    return None

class PreprocessNode:
    async def preprocess_multimedia_node(self, state: AgentState) -> dict:
        """
        Entry node in the LangGraph workflow.
        Extracts clean Image IDs (UUIDs) from SAS URLs, populates the media_map,
        and cleans the additional_material array.
        """
        additional_material = state.get("additional_material") or []
        media_map = state.get("media_map") or {}
        messages = state.get("messages") or []

        new_additional = []
        new_media_map = dict(media_map)

        # Parse from last message. If the user just uploaded an image, the SAS URL will be in the last message content.
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content") and isinstance(last_message.content, list):
                for part in last_message.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url_obj = part.get("image_url") or {}
                        sas_url = image_url_obj.get("url")
                        if sas_url and isinstance(sas_url, str) and sas_url.startswith("http"):
                            info = parse_sas_url(sas_url)
                            if info:
                                uuid = info["uuid"]
                                new_media_map[uuid] = sas_url
                                if uuid not in new_additional:
                                    new_additional.append(uuid)
                                logger.info(f"Parsed SAS URL from CopilotKit message content: {uuid} -> {sas_url[:50]}...")

        return {
            "additional_material": new_additional,
            "media_map": new_media_map
        }
