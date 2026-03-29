from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage


def get_summarisation_prompt() -> ChatPromptTemplate:
    """Prompt để tóm tắt lịch sử hội thoại bóng đá thành SessionMemory có cấu trúc."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Bạn là chuyên gia phân tích hội thoại bóng đá. "
            "Nhiệm vụ của bạn là đọc lịch sử hội thoại và tạo ra một bản tóm tắt có cấu trúc "
            "để hỗ trợ trả lời các câu hỏi tiếp theo. Chỉ trích xuất thông tin có trong hội thoại, "
            "không thêm thông tin ngoài."
        )),
        HumanMessagePromptTemplate.from_template(
            """Đọc đoạn hội thoại bóng đá dưới đây và tạo ra một SessionMemory có cấu trúc.

## Hội thoại cần tóm tắt:
{conversation_block}

## Hướng dẫn điền từng trường:
- **conversation_state**: Mô tả tiến trình hội thoại ở mức vừa đủ — không chỉ nêu chủ đề mà phải nêu rõ người dùng đang ở giai đoạn nào (mới bắt đầu hỏi, đang đi sâu, hay đang xác nhận thông tin), những thông tin nào đã được cung cấp và phản hồi ra sao. Ví dụ: "Người dùng hỏi về phong độ của Chelsea mùa 2014-15. Hệ thống đã cung cấp số trận thắng và điểm số. Người dùng tiếp tục hỏi chi tiết về Diego Costa."
- **user_context.preferences**: Liệt kê rõ các sở thích thể thao được thể hiện trong hội thoại: đội bóng yêu thích, giải đấu quan tâm, cầu thủ theo dõi, mùa giải cụ thể, phong cách chơi hoặc số liệu mà người dùng hay hỏi tới. Nếu người dùng hỏi nhiều lần về cùng một chủ đề thì đó là dấu hiệu sở thích rõ ràng. Để trống nếu không có đủ thông tin.
- **user_context.goals**: Mô tả mục tiêu thông tin cụ thể của người dùng trong cuộc hội thoại này — họ muốn biết điều gì, phục vụ mục đích gì (tra cứu, so sánh, phân tích, tranh luận…). Ví dụ: "Muốn xác nhận Diego Costa có phải là chân sút hàng đầu EPL 2014-15 không, có vẻ để so sánh với Sergio Agüero." Để trống nếu mục tiêu chưa rõ.
- **shared_context**: QUAN TRỌNG — Đây là trường cốt lõi. Liệt kê tất cả thực thể bóng đá đã được nhắc đến kèm thông tin cụ thể đã được xác nhận hoặc tìm thấy trong hội thoại. Mỗi entry phải đủ thông tin để tái sử dụng: tên thực thể, vai trò/loại thực thể, các số liệu hoặc sự kiện cụ thể. Format: "Tên thực thể (loại) — thông tin cụ thể". Ví dụ: ["Chelsea FC (đội bóng) — EPL 2014-15, vô địch với 87 điểm, 26 trận thắng", "Diego Costa (tiền đạo, Chelsea) — 20 bàn thắng mùa EPL 2014-15, dẫn đầu danh sách ghi bàn"]. Để trống nếu chưa có thông tin cụ thể nào được xác nhận.
- **open_discussion_threads**: Liệt kê các câu hỏi chưa được trả lời đầy đủ, chủ đề đang bỏ lửng, hoặc điều người dùng có vẻ muốn hỏi tiếp. Bao gồm cả những câu hỏi mà hệ thống trả lời chưa chính xác hoặc cần làm rõ thêm. Để trống nếu không có.
- **scope**: Mô tả ngắn gọn phạm vi bao quát của cuộc hội thoại — kết hợp chủ thể chính (đội, giải, cầu thủ) và khung thời gian nếu có. Ví dụ: "Thống kê và phong độ của Chelsea và Diego Costa tại EPL mùa 2014-15". Để trống nếu hội thoại quá rộng hoặc chưa có hướng rõ ràng.

## Output Format:
{format_instructions}
"""
        )
    ])
