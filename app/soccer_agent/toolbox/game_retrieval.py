import json
import os
import logging
from app.schema.match import Annotation
from typing import List, Type,Optional, Literal,Annotated, Union
from pydantic import BaseModel, Field, PrivateAttr

from app.soccer_agent.factory.llm_provider import get_llm
from typing import Any
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.tools import InjectedState, BaseTool
from langsmith import get_current_run_tree

from app.config.settings import Settings
from app.config.config import PROJECT_PATH, GEMINI_2_0_FLASH_LITE, GEMINI_2_5_FLASH_LITE
from app.soccer_agent.prompts.toolbox.game_retrieval import get_game_info_retrieval_prompt_template, get_game_history_retrieval_prompt_template

logger = logging.getLogger(__name__)

# ==========================================
# 1. Định nghĩa Data Models
# ==========================================
class RetrievalInput(BaseModel):
    query: str = Field(description="Câu hỏi hoặc truy vấn của người dùng về trận đấu.")
    execution_agent_state: Annotated[dict, InjectedState] = Field(description="Trạng thái hiện tại của execution agent, bao gồm các artifact từ các công cụ trước đó.")

class ToolOutput(BaseModel):
    answer: str = Field(description="Câu trả lời dựa trên thông tin được truy xuất từ file JSON.")

# ==========================================
# 2. Tool: Game Info Retrieval (Metadata)
# ==========================================
class GameInfoRetrievalTool(BaseTool):
    name: str = "game_info_retrieval"
    description: str = """
    Retrieve structured match metadata and final results from the match database based on the identified match
    (for example: referee, coaches, attendance, formation, venue, lineups,
    and final score). Use this tool after game_search for questions that require static pre- or
    post-match facts. The tool reads the match JSON and returns a concise,
    structured answer based on the metadata.
    """
    args_schema: Type[BaseModel] = RetrievalInput # type: ignore
    
    project_path: str = PROJECT_PATH
    _llm: Any = PrivateAttr()
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    def __init__(self):
        super().__init__()

        self._llm = get_llm(
            temperature=0,
            model=GEMINI_2_0_FLASH_LITE,
        )

    def _get_match_info_json(self, json_file_path: str) -> str:
        """Đọc file JSON và loại bỏ phần annotations để lấy metadata."""
        try:
            full_path = os.path.join(self.project_path, "app", json_file_path)
            with open(full_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Loại bỏ phần annotations (history) để giảm token, chỉ giữ lại thông tin chung
            if "annotations" in data:
                del data["annotations"]
            if "comments" in data:
                del data["comments"]
            
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Error in reading match info JSON file: {str(e)}")

    def _run(self, query: str, execution_agent_state: Annotated[dict, InjectedState],run_manager: Optional[CallbackManagerForToolRun] = None):
        try:
            if not execution_agent_state.get("last_tool_artifact"):
                return "Error: Missing game file information. Please ensure 'game_search' tool has been executed successfully before running this tool.", None

            file_path = execution_agent_state["last_tool_artifact"]
            match_info_context = self._get_match_info_json(file_path)

            if match_info_context.startswith("Error"):
                return f"Failed to retrieve match info. Please try again or stop the execution.", file_path

            prompt = get_game_info_retrieval_prompt_template()
            llm_structured = self._llm.with_structured_output(ToolOutput)
            chain = prompt | llm_structured
            
            response: ToolOutput = chain.invoke({
                "query": query,
                "context": match_info_context
            }) # type: ignore
            
            return response.answer, None
        except Exception as e:
            error_msg = f"Error in game_info_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Send error to LangSmith run tree
            run_tree = get_current_run_tree()
            if run_tree:
                run_tree.end(
                    error=error_msg
                )
            
            return f"An error occurred while retrieving game info. Details: {str(e)}. Please try again or stop the execution.", None

# ==========================================
# 3. Tool: Match History Retrieval (Live Stream/Events)
# ==========================================

class GameHistoryRetrievalTool(BaseTool):
    name: str = "game_history_retrieval"
    description: str = """
    Load the match event timeline or commentary for the identified match and generate answers grounded in
    those events. Use this tool after `game_search` or 'commentary_generation' to answer questions about specific events, timestamps,
    substitutions, goals, cards, and other in-game details. returns a context-aware
    synthesized response.
    """
    args_schema: Type[BaseModel] = RetrievalInput # type: ignore

    project_path: str = PROJECT_PATH
    _llm: Any = PrivateAttr()
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def __init__(self):
        super().__init__()

        self._llm = get_llm(
            temperature=0,
            model = GEMINI_2_5_FLASH_LITE
        )

    def _transform_match_history_artifact_to_str(self, match_history_artifact: Union[str, List[Annotation]]) -> str:
        """
        Transform the match history from JSON file or list of Annotation objects into a string format suitable for prompt context.
        """
        # Check the type of match_history_artifact
        if isinstance(match_history_artifact, str):
            # Assume it's a JSON file path
            return self._read_match_history_from_file(match_history_artifact)
        elif isinstance(match_history_artifact, list):
            # Assume it's a list of Annotation objects
            processed_annotations = [anno.model_dump() for anno in match_history_artifact]
            return json.dumps(processed_annotations, indent=2, ensure_ascii=False)
        else:
            raise ValueError("Unsupported artifact type for match history. Expected file path (str) (which is the output of game_search tool) or list of Annotation objects (which is the output of commentary_generation tool).")

        
    def _read_match_history_from_file(self, json_file_path: str) -> str:
        """
        Read the match history JSON file and convert it into a string
        
        :param json_file_path: Path to the JSON file containing match history
        :type json_file_path: str
        :return: Processed match history as a string
        :rtype: str
        """
        # Check whether the file exists
        full_path = os.path.join(self.project_path, "app", json_file_path)
        if not os.path.exists(full_path):
            # Fallback: thử tìm trực tiếp nếu path đã đầy đủ
            if os.path.exists(json_file_path):
                full_path = json_file_path
            else:
                raise FileNotFoundError(f"Error: File not found at {full_path}")

        with open(full_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        processed_annotations = []
        
        # --- CASE 1: Format 'annotations' (MatchTime) ---
        if "annotations" in data:
            event_list = data.get("annotations", [])
            for event in event_list:
                # Logic lấy timestamp ưu tiên
                timestamp = event.get("contrastive_aligned_gameTime", "")
                if not timestamp:
                    timestamp = event.get("gameTime", "")
                
                anno = Annotation(
                    description=event.get("description", ""),
                    label=event.get("label", "unknown"),
                    gameTime=timestamp
                )
                processed_annotations.append(anno)

        # --- CASE 2: Format 'comments' (SoccerWiki / 1988) ---
        elif "comments" in data:
            comments_list = data.get("comments", [])
            for comment in comments_list:
                # Mapping theo yêu cầu mới:
                # description <= comments_text
                # label <= comments_type
                # gameTime <= half - time_stamp
                
                desc = comment.get("comments_text", "")
                lbl = comment.get("comments_type", "unknown")
                
                half = str(comment.get("half", ""))
                t_stamp = str(comment.get("time_stamp", ""))
                
                # Format gameTime ví dụ: "1 - 15:30"
                g_time = f"{half} - {t_stamp}"

                anno = Annotation(
                    description=desc,
                    label=lbl,
                    gameTime=g_time
                )
                processed_annotations.append(anno)
        
        else:
            raise ValueError("Error: Unsupported JSON format. Match history file must contain 'annotations' or 'comments' key.")

        if not processed_annotations:
            raise ValueError("Match history file doesn't contain any events. Therefore, cannot proceed with retrieval.")
        
        # Chuyển list object thành chuỗi JSON đẹp để đưa vào prompt
        return json.dumps([a.model_dump() for a in processed_annotations], indent=2, ensure_ascii=False)


    def _run(self, query: str, execution_agent_state: Annotated[dict, InjectedState], run_manager: Optional[CallbackManagerForToolRun] = None):
        run_tree = get_current_run_tree()
        try:
            if not execution_agent_state.get("last_tool_artifact"):
                raise ValueError("Missing game file information. Please ensure 'game_search' or 'commentary_generation' tool has been executed successfully before running this tool.")

            match_history_artifact = execution_agent_state["last_tool_artifact"] # It should be a JSON file path (if the previous tool is game_search) or list of Annotation objects (if the previous tool is commentary_generation)
            logger.info(f"📖 Processing Match History from: {match_history_artifact[:500]}...")
            match_history_context = self._transform_match_history_artifact_to_str(match_history_artifact)

            # Nếu quá dài, có thể cắt bớt ở đây, nhưng Gemini Flash context window rất lớn (1M tokens).
            prompt = get_game_history_retrieval_prompt_template()
            llm_structured = self._llm.with_structured_output(ToolOutput)
            chain = prompt | llm_structured
            response: ToolOutput = chain.invoke({
                "query": query,
                "context": match_history_context
            }) # type: ignore
            logger.info(f"Game History Retrieval Response: {response}")
            return response.answer, None
        
        except Exception as e:
            error_msg = f"Error in game_history_retrieval: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Send error to LangSmith run tree
            if run_tree:
                run_tree.end(
                    error=error_msg
                )
            return f"An error occurred while retrieving game history. Details: {str(e)}. Please try again or stop the execution.", None