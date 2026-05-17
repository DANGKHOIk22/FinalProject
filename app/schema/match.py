from typing import Literal
from pydantic import BaseModel, Field

class MatchInfo(BaseModel):
    """Thông tin trích xuất từ câu truy vấn về trận đấu bóng đá."""
    league: str = Field(description="Giải đấu (england_epl, germany_bundesliga, europe_uefa-champions-league, italy_serie-a, france_league-1, spain_laliga, hoặc unknown)", default="unknown")
    season: str = Field(description="Mùa giải (ví dụ: 2023-2024), hoặc unknown", default="unknown")
    date: str = Field(description="Ngày cụ thể format YYYY-MM-DD, hoặc unknown", default="unknown")
    year: str = Field(description="Năm diễn ra trận đấu (số), hoặc unknown", default="unknown")
    month: str = Field(description="Tháng diễn ra trận đấu (số), hoặc unknown", default="unknown")
    day: str = Field(description="Ngày diễn ra trận đấu (số), hoặc unknown", default="unknown")
    time: str = Field(description="Giờ thi đấu format HH:MM, hoặc unknown", default="unknown")
    score: str = Field(description="Tỷ số dạng X - X, hoặc unknown", default="unknown")
    team1: str = Field(description="Tên đội 1 (giữ nguyên tên gốc)", default="unknown")
    team2: str = Field(description="Tên đội 2 (giữ nguyên tên gốc, nếu không có để unknown)", default="unknown")
    time_range: Literal["day", "week", "month", "year"] = Field(
        default="month",
        description=(
            "Phạm vi thời gian phù hợp để tìm kiếm tin tức nếu không có trong DB. "
            "day: hôm nay/hôm qua/today/yesterday. "
            "week: tuần này/tuần trước/this week/last week. "
            "month: tháng này/tháng trước/gần đây/this month/last month/recently. "
            "year: mùa giải/năm nay/năm ngoái/this season/last season/this year/last year."
        )
    )

class Annotation(BaseModel):
    """Thông tin chú thích cho một tình huống cụ thể trong trận đấu."""
    description: str = Field(
        default="",
        description="Mô tả chi tiết về tình huống."
    )

    label: str = Field(
        default="",
        description="Nhãn phân loại tình huống (vd: goal, corner, foul, y-card, r-card, substitution, penalty, offside)."
    )

    gameTime: str = Field(
        default="",
        description="Thời gian xảy ra tình huống trong trận đấu (hiệp - phút:giây). VD: '2 - 48:43'."
    )

