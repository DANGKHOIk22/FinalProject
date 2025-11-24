import json
import os
import logging
from app.schema.match import Annotation
from pathlib import Path
from typing import List, Type,Optional, Literal,Annotated
from pydantic import BaseModel, Field
from app.config.config import PROJECT_PATH
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.tools import InjectedState, BaseTool
from app.config.settings import Settings


logger = logging.getLogger(__name__)

# ==========================================
# 1. Định nghĩa Data Models
# ==========================================

class RetrievalInput(BaseModel):
    query: str = Field(description="Câu hỏi hoặc truy vấn của người dùng về trận đấu.")
    execution_agent_state: Annotated[dict, InjectedState] = Field(description="Trạng thái hiện tại của execution agent, bao gồm các artifact từ các công cụ trước đó.")

class ToolOutput(BaseModel):
    answer: str = Field(description="Câu trả lời dựa trên thông tin được truy xuất từ file JSON.")
    artifact: str = Field(description="Dữ liệu gốc được sử dụng để tạo câu trả lời (nội dung file JSON).")
# ==========================================
# 2. Tool: Game Info Retrieval (Metadata)
# ==========================================

class GameInfoRetrievalTool(BaseTool):
    name: str = "game_info_retrieval"
    description: str = """
    Retrieves pre-match info (referee, coach, attendance, formation) and final results/scores from the soccer match database JSON file.
    Use this for static game information.
    """
    args_schema: Type[BaseModel] = RetrievalInput
    
    project_path: str = PROJECT_PATH
    llm: ChatGoogleGenerativeAI = None
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    def __init__(self):
        super().__init__()

        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-flash-latest",
            temperature=0,
            google_api_key=Settings.GOOGLE_API_KEY
        )

    def _get_match_info_json(self, json_file_path: str) -> str:
        """Đọc file JSON và loại bỏ phần annotations để lấy metadata."""
        try:
            full_path = Path(self.project_path, json_file_path).as_posix()
            with open(full_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Loại bỏ phần annotations (history) để giảm token, chỉ giữ lại thông tin chung
            if "annotations" in data:
                del data["annotations"]
            if "comments" in data:
                del data["comments"]
            
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _run(self, query: str, execution_agent_state: Annotated[dict, InjectedState],run_manager: Optional[CallbackManagerForToolRun] = None):
        if not execution_agent_state.get("last_tool_artifact"):
            return "Error: No last_tool_artifact provided."

        file_path = execution_agent_state["last_tool_artifact"]
        match_info_context = self._get_match_info_json(file_path)

        if match_info_context.startswith("Error"):
            return match_info_context,file_path

        prompt_template = """
        You are a soccer expert. Answer the question based ONLY on the provided match related information (metadata).

        User Question: "{query}"

        Match Information:
        {context}

        Please provide the answer based on the match related information. Make sure your answer is evidence-based and accurate.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm_structured = self.llm.with_structured_output(ToolOutput)
        chain = prompt | llm_structured
        
        response = chain.invoke({
            "query": query,
            "context": match_info_context
        })
        
        return response.answer, response.artifact

# ==========================================
# 3. Tool: Match History Retrieval (Live Stream/Events)
# ==========================================

class GameHistoryRetrievalTool(BaseTool):
    name: str = "game_history_retrieval"
    description: str = """
    Retrieves the textual live stream/commentary history of the whole game from the JSON file. 
    Use this for questions about specific events, timestamps, plays, or game statistics that happened during the match.
    """
    args_schema: Type[BaseModel] = RetrievalInput

    project_path: str = PROJECT_PATH
    llm: ChatGoogleGenerativeAI = None
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def __init__(self):
        super().__init__()

        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-flash-latest", # Dùng flash cho context dài (lịch sử trận đấu thường dài)
            temperature=0,
            google_api_key=Settings.GOOGLE_API_KEY
        )

    def _process_data(self, json_file_path: str) -> str:
        """
        Đọc file JSON và chuẩn hóa dữ liệu (annotations hoặc comments) 
        thành danh sách các object Annotation.
        """
        try:
            # Xử lý đường dẫn file
            full_path = os.path.join(self.project_path, json_file_path)
            if not os.path.exists(full_path):
                # Fallback: thử tìm trực tiếp nếu path đã đầy đủ
                if os.path.exists(json_file_path):
                    full_path = json_file_path
                else:
                    return f"Error: File not found at {full_path}"

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
                return "Error: Unsupported JSON format. File must contain 'annotations' or 'comments' key."

            if not processed_annotations:
                return "No events found in the file."

            # Chuyển list object thành chuỗi JSON đẹp để đưa vào prompt
            return json.dumps([a.model_dump() for a in processed_annotations], indent=2, ensure_ascii=False)

        except Exception as e:
            return f"Error reading/processing file: {str(e)}"

    def _run(self, query: str, execution_agent_state: Annotated[dict, InjectedState], run_manager: Optional[CallbackManagerForToolRun] = None):
        if not execution_agent_state.get("last_tool_artifact"):
            return "Error: No last_tool_artifact provided."

        file_path = execution_agent_state["last_tool_artifact"]
        if not file_path:
            return "Error: No game file provided. Please run Game Search first.", None

        logger.info(f"📖 Processing Match History from: {file_path}")
        match_history_context = self._process_data(file_path)

        if match_history_context.startswith("Error"):
            return match_history_context,file_path
        
        # Nếu quá dài, có thể cắt bớt ở đây, nhưng Gemini Flash context window rất lớn (1M tokens).
        
        prompt_template = """
        You are a soccer expert. Answer the question based ONLY on the provided match history (live commentary/annotations).
        
        User Question: "{query}"

        Match History (List of Annotations):
        {context}

        Please provide the answer based on the match history information. Think carefully about timestamps and event sequences. Make sure your answer is evidence-based and accurate.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm_structured = self.llm.with_structured_output(ToolOutput)
        chain = prompt | llm_structured
        response = chain.invoke({
            "query": query,
            "context": match_history_context
        })
        logger.info(f"Game History Retrieval Response: {response}")
        return response.answer, response.artifact