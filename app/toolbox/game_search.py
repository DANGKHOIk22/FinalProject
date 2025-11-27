import os
import logging
import pandas as pd
from typing import Type, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import CallbackManagerForToolRun

# Import config và prompts từ project của bạn
from app.config.config import PROJECT_PATH, DEFAULT_MODEL
from app.config.settings import Settings
from app.schema.match import MatchInfo
from app.prompts.toolbox.game_search import get_extraction_prompt_template, get_match_selection_prompt_template

logger = logging.getLogger(__name__)

# --- Input Schema ---
class GameSearchInput(BaseModel):
    query: str = Field(description="Câu truy vấn tự nhiên về trận đấu bóng đá cần tìm kiếm.")

# --- Output Schema (Internal use) ---
class FinalResult(BaseModel):
    """Cấu trúc trả về từ LLM khi chọn trận đấu."""
    path: Optional[str] = Field(description="Đường dẫn file (file_path) của trận đấu nếu tìm thấy chính xác. Nếu không tìm thấy, để null.", default=None)
    response_llm: str = Field(description="Lời giải thích chi tiết về việc tìm thấy hay không tìm thấy trận đấu.")

# --- Tool Definition ---
class GameSearchTool(BaseTool):
    name: str = "game_search"
    description: str = """
    Ability: Given certain information regarding a match, this tool retrieves the corresponding game from the soccer match database. 
    The games pertain to six major European leagues (England Premier, Germany Bundesliga, Italy Serie A, Spain La Liga, France Ligue 1, 
    and the European Champions League) spanning the years 2017-2024.
    Query Input: Simply the original inquiry as the query input here.
    Output: Returns a summary message and the JSON file path of the identified game as an artifact.
    Remark: This tool must be utilized initially to acquire the game's context.
    """
    args_schema: Type[BaseModel] = GameSearchInput
    
    # Cấu hình trả về cả Content (Text) và Artifact (File Path)
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    # Các thuộc tính nội bộ
    project_path: str = PROJECT_PATH
    csv_path: str = ""
    llm: ChatGoogleGenerativeAI = None
    df: pd.DataFrame = None

    def __init__(self):
        super().__init__()
        self.csv_path = os.path.join(self.project_path, "app", "database", "game_database.csv")
        
        # Khởi tạo LLM
        self.llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL, 
            temperature=0,
            google_api_key=Settings.GOOGLE_API_KEY
        )
        
        # Load dữ liệu CSV
        try:
            if os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
            else:
                logger.error(f"CSV file not found at: {self.csv_path}")
                self.df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            self.df = pd.DataFrame()

    def _extract_match_info(self, query: str) -> MatchInfo:
        """Bước 1: Trích xuất thông tin (Bỏ PydanticOutputParser thừa)."""
        prompt = get_extraction_prompt_template()
        
        # Gemini tự động parse ra object MatchInfo
        structured_llm = self.llm.with_structured_output(MatchInfo)
        
        chain = prompt | structured_llm
        return chain.invoke({"question": query})

    def _retrieve_candidates(self, info: MatchInfo):
        """Bước 2: Lọc dữ liệu Pandas."""
        if self.df.empty:
            return None, None

        df = self.df.copy()
        conditions = []

        # Lọc Metadata
        if info.league != "unknown": conditions.append(df["league"] == info.league)
        if info.season != "unknown": conditions.append(df["season"] == info.season)
        if info.year != "unknown": conditions.append(df["year"] == int(info.year))
        if info.month != "unknown": conditions.append(df["month"] == int(info.month.lstrip('0')))
        if info.day != "unknown": conditions.append(df["day"] == int(info.day.lstrip('0')))
        if info.time != "unknown": conditions.append(df["time"] == info.time)

        if conditions:
            combined_condition = pd.concat(conditions, axis=1).all(axis=1)
            initial_filtered_df = df[combined_condition]
        else:
            initial_filtered_df = df

        if initial_filtered_df.empty:
            return initial_filtered_df, None

        # Lọc Team
        final_filtered_df = initial_filtered_df
        team_values = [t for t in [info.team1, info.team2] if t != "unknown"]
        
        if team_values:
            team_conditions = []
            t1 = info.team1.replace(" ", "") if info.team1 != "unknown" else ""
            t2 = info.team2.replace(" ", "") if info.team2 != "unknown" else ""
            target_df = initial_filtered_df

            if t1 and t2:
                mask = (
                    (target_df["home_team"].str.replace(" ", "").str.contains(t1, case=False, na=False) &
                     target_df["away_team"].str.replace(" ", "").str.contains(t2, case=False, na=False)) |
                    (target_df["home_team"].str.replace(" ", "").str.contains(t2, case=False, na=False) &
                     target_df["away_team"].str.replace(" ", "").str.contains(t1, case=False, na=False))
                )
                team_conditions.append(mask)
            elif t1:
                mask = (target_df["home_team"].str.replace(" ", "").str.contains(t1, case=False, na=False) |
                        target_df["away_team"].str.replace(" ", "").str.contains(t1, case=False, na=False))
                team_conditions.append(mask)
            elif t2:
                mask = (target_df["home_team"].str.replace(" ", "").str.contains(t2, case=False, na=False) |
                        target_df["away_team"].str.replace(" ", "").str.contains(t2, case=False, na=False))
                team_conditions.append(mask)

            if team_conditions:
                final_mask = pd.concat(team_conditions, axis=1).any(axis=1)
                final_filtered_df = initial_filtered_df[final_mask]

        if len(final_filtered_df) > 10:
            final_filtered_df = None 

        return initial_filtered_df, final_filtered_df

    def _finalize_candidate_selection(self, candidates, candidates_with_team, info: MatchInfo, question: str) -> Tuple[str, Optional[str]]:
        """Bước 3: Chọn kết quả cuối cùng (Trả về Content và Path)."""
        
        if candidates is None or (isinstance(candidates, pd.DataFrame) and candidates.empty):
             return "We did not find the match you mentioned in the database. Stop the execution and ask user give more specific information.", None

        # Case: Tìm thấy chính xác 1 kết quả
        target_candidates = candidates_with_team if candidates_with_team is not None else candidates
        
        if len(target_candidates) == 1:
            row = target_candidates.iloc[0]
            msg = f"Found match: {row['home_team']} vs {row['away_team']} ({row['date']})."
            return msg, row['file_path']

        # Case: Nhiều kết quả -> Dùng LLM chọn
        if len(target_candidates) > 1:
            candidate_text = ""
            for i, row in target_candidates.iterrows():
                candidate_text += f"""
                Candidate {i + 1}:
                - Date: {row['date']}, League: {row['league']}
                - Match: {row['home_team']} vs {row['away_team']}
                - Score: {row['score']}, File: {row['file_path']}
                """

            prompt = get_match_selection_prompt_template()
            
            # Gemini tự parse ra object FinalResult
            structured_llm = self.llm.with_structured_output(FinalResult)
            
            chain = prompt | structured_llm
            
            response: FinalResult = chain.invoke({
                "question": question,
                "info": info.model_dump_json(),
                "candidates": candidate_text
            }) # type: ignore
            
            return response.response_llm, response.path
        
        return "Could not find a specific match.", None

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Tuple[str, Optional[str]]:
        """
        Hàm thực thi chính.
        Trả về Tuple (Content, Artifact) vì response_format="content_and_artifact".
        """
        logger.info(f"🔎 Game Search Query: {query}")
        
        try:
            # 1. Extract
            info = self._extract_match_info(query)
            logger.debug(f"Extracted Info: {info}")
            
            # 2. Filter
            candidates, candidates_with_team = self._retrieve_candidates(info)
            
            # 3. Finalize
            content, artifact_path = self._finalize_candidate_selection(
                candidates, 
                candidates_with_team, 
                info, 
                query
            )
            
            logger.info(f"✅ Search Result: {content} | Path: {artifact_path}")
            
            # Trả về đúng định dạng (Content, Artifact)
            return content, artifact_path
            
        except Exception as e:
            error_msg = f"Error in Game Search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"An error occurred while searching for the game. Details: {str(e)}. Please check your query or try again.", None