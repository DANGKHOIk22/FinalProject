import base64
import logging
import mimetypes
import os
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, List, Optional, Tuple, Type, Literal, Union, Dict, Annotated
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import InjectedToolArg
from app.soccer_agent.factory.llm_provider import get_llm
from langsmith import get_current_run_tree

from app.soccer_agent.prompts.toolbox.commentary_generation import get_commentary_generation_prompt_template
from app.schema.match import Annotation
from app.cache.standard_cache import standard_cache
from app.soccer_agent.toolbox._config_loader import tool_description
from langgraph.prebuilt import ToolRuntime


class CommentaryGenerationInput(BaseModel):
    """Input schema for commentary generation.

    `material` is expected to be one or more local video file paths.
    `query` is optional extra context (often None).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    video_id: str = Field(
        ..., description="UUID of the video media to generate commentary for."
    )
    query: Optional[str] = Field(
        default=None,
        description="Optional user query/context to guide commentary (usually None).",
    )
    runtime: Annotated[Optional[ToolRuntime], InjectedToolArg] = Field(default=None)


class _CommentaryGenerationOutput(BaseModel):
    commentary: str = Field(description="Generated match commentary (~500 words).")
    annotations: List[Annotation] = Field(
        default_factory=list,
        description="List of key-event annotations extracted from the video.",
    )


class CommentaryGenerationTool(BaseTool):
    name: str = "commentary_generation"
    description: str = (
        "Given a soccer match video clip, this tool generates an approximately 500-word "
        "match commentary and extracts key events as structured annotations. "
        "It is only used at the beginning of the chain to process the initial user-provided video clip. "
        "After using this tool, you must use the 'game_history_retrieval' tool to answer questions about the match history."
    )

    args_schema: Type[BaseModel] = CommentaryGenerationInput  # type: ignore
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _vlm: Any = PrivateAttr()

    def __init__(self):
        super().__init__(description=tool_description("commentary_generation"))
        self._vlm = get_llm("tool-multi-modal")

    async def warmup(self) -> None:
        from app.soccer_agent.factory.llm_provider import warm_llm
        system_msg = get_commentary_generation_prompt_template().messages[0]
        await warm_llm(self._vlm, system_msg, "commentary_generation")

    def _run(
        self,
        video_id: str,
        query: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        runtime: Optional[ToolRuntime] = None,
    ) -> Tuple[str, List[Annotation]]:
        run_tree = get_current_run_tree()
        try:
            user_id = "default_user"
            thread_id = "default_thread"
            media_registry = None
            if runtime and runtime.config:
                configurable = runtime.config.get("configurable", {})
                user_id = str(configurable.get("user_id", "default_user"))
                thread_id = str(configurable.get("thread_id", "default_thread"))
                media_registry = configurable.get("media_registry")

            if not media_registry:
                raise ValueError("MediaRegistryService not found in runtime config")

            output = self._generate_commentary(video_id, query, media_registry=media_registry, user_id=user_id, thread_id=thread_id)

            content_msg = (
                "Successfully generated commentary from the provided video material. "
                f"Extracted {len(output.annotations)} annotations. "
                f"This is the short commentary:\n\n{output.commentary}"
            )
            logger.info(f"✅ commentary_generation: annotations={len(output.annotations)} | commentary={output.commentary[:200]}")
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

    def _generate_commentary(self, media_id: str, query: Optional[str] = None, media_registry: Any = None, user_id: str = "default_user", thread_id: str = "default_thread") -> _CommentaryGenerationOutput:
        """Internal method to handle the VLM generation with caching."""
        self.validate_tool_input(media_id, media_registry, user_id, thread_id)
        
        sas_url = media_registry.get_sas_url(user_id, thread_id, media_id)
        if not sas_url:
            raise ValueError(f"Could not retrieve SAS URL for video media ID: {media_id}")

        parser = PydanticOutputParser(pydantic_object=_CommentaryGenerationOutput)
        prompt_template = get_commentary_generation_prompt_template()
        
        query_context = f"\n### USER QUERY/CONTEXT\nFocus commentary on: {query}\n" if query else ""

        # Use video URL for generation
        prompt_value = prompt_template.invoke(
            {
                "output_format": parser.get_format_instructions(),
                "video_url": sas_url,
                "query_context": query_context,
            }
        )
        response = self._vlm.invoke(prompt_value)
        response_text = response.content[1] if isinstance(response.content, list) else str(response.content)

        output: _CommentaryGenerationOutput = parser.parse(response_text) # type: ignore
        return output

    @staticmethod
    def validate_tool_input(media_id: str, media_registry: Any, user_id: str, thread_id: str) -> None:
        """Validate input is a non-empty list of existing media IDs."""
        if not isinstance(media_id, str) or not media_id.strip():
            raise ValueError("'media_id' must be a non-empty string.")

