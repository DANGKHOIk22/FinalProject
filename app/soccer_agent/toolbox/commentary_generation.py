import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Literal, Union, Dict
from pydantic import BaseModel, Field, PrivateAttr

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from app.soccer_agent.factory.llm_provider import get_llm
from langsmith import get_current_run_tree

from app.config.config import DEFAULT_MODEL
from app.config.settings import Settings
from app.soccer_agent.prompts.toolbox.commentary_generation import get_commentary_generation_prompt_template
from app.schema.match import Annotation
from app.cache.exact_cache import exact_cache


class CommentaryGenerationInput(BaseModel):
    """Input schema for commentary generation.

    `material` is expected to be one or more local video file paths.
    `query` is optional extra context (often None).
    """

    material: List[str] = Field(
        ..., description="List of local video file paths (e.g., segment clips)."
    )
    query: Optional[str] = Field(
        default=None,
        description="Optional user query/context to guide commentary (usually None).",
    )


class _CommentaryGenerationOutput(BaseModel):
    commentary: str = Field(description="Generated match commentary (~500 words).")
    annotations: List[Annotation] = Field(
        default_factory=list,
        description="List of key-event annotations extracted from the video.",
    )


class CommentaryGenerationTool(BaseTool):
    name: str = "commentary_generation"
    description: str = (
        "Given one or more soccer match video clips (file paths), this tool generates an approximately 500-word "
        "match commentary and extract key events as structured annotations. "
        "After using this tool, you can use the 'game_history_retrieval' tool for answering questions about the match history."
    )

    args_schema: Type[BaseModel] = CommentaryGenerationInput  # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _vlm: Any = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._vlm = get_llm(
            temperature=1.0,
            top_p=0.95
        )

    def _run(
        self,
        material: List[str],
        query: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[Annotation]]:
        run_tree = get_current_run_tree()
        try:
            output = self._cached_generate_commentary(material, query)

            content_msg = (
                "Successfully generated commentary from the provided video material. "
                f"Extracted {len(output.annotations)} annotations. "
                f"This is the short commentary:\n\n{output.commentary}"
            )
            return content_msg, output.annotations

        except Exception as e:
            error_msg = f"Error in CommentaryGenerationTool: {str(e)}"

            # Log the error to LangSmith
            if run_tree:
                run_tree.end(
                    error=error_msg
                )

            return (
                error_msg,
                [],
            )

    @exact_cache.cache(ttl=60 * 60, validatedModel=_CommentaryGenerationOutput)
    def _cached_generate_commentary(self, material: List[str], query: Optional[str] = None) -> _CommentaryGenerationOutput:
        """Internal method to handle the VLM generation with caching."""
        self.validate_tool_input(material)
        media_parts = self.transform_input(material)

        parser = PydanticOutputParser(pydantic_object=_CommentaryGenerationOutput)
        prompt_template = get_commentary_generation_prompt_template()
        
        # Use first video for generation
        prompt_value = prompt_template.invoke(
            {
                "output_format": parser.get_format_instructions(),
                "mime_type": media_parts[0]["mime_type"],
                "video_base64": media_parts[0]["data"],
            }
        )
        response = self._vlm.invoke(prompt_value)
        response_text = response.content[1] if isinstance(response.content, list) else str(response.content)

        output: _CommentaryGenerationOutput = parser.parse(response_text) # type: ignore
        return output

    @staticmethod
    def validate_tool_input(material: Any) -> None:
        """Validate input is a non-empty list of existing video file paths."""
        if not isinstance(material, list) or not material:
            raise ValueError("'material' must be a non-empty list of video file paths.")

        allowed_ext = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
        for raw_path in material:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("Each item in 'material' must be a non-empty string path.")
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                raise ValueError(f"Video file not found: {raw_path}")
            if path.suffix.lower() not in allowed_ext:
                raise ValueError(
                    f"Unsupported video extension '{path.suffix}' for file: {raw_path}. "
                    f"Allowed: {sorted(allowed_ext)}"
                )

    @staticmethod
    def transform_input(material: List[str], max_file_size_mb: int = 100) -> List[dict]:
        """Prepare video clips as base64 inline media parts.
        """

        parts: List[dict] = []
        for raw_path in material:
            path = Path(raw_path)
            size_bytes = path.stat().st_size
            if size_bytes > max_file_size_mb * 1024 * 1024:
                raise ValueError(
                    f"Video file too large ({size_bytes / (1024 * 1024):.1f} MB): {raw_path}. "
                    f"Max allowed is {max_file_size_mb} MB."
                )

            mime_type, _ = mimetypes.guess_type(str(path))
            mime_type = mime_type or "video/mp4"

            with open(path, "rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Multimodal part format used by LangChain's Google GenAI integration.
            parts.append({"type": "media", "source_type": "base64", "mime_type": mime_type, "data": data_b64})

        return parts

