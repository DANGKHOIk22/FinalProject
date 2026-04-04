from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage


def get_summarisation_prompt() -> ChatPromptTemplate:
    """
    Prompt để tóm tắt lịch sử hội thoại bóng đá thành SessionMemory có cấu trúc.

    Lịch sử đầu vào có format tool-enriched:
        User: <câu hỏi>
        Assistant:
        [Tool Usage]
          Step 1: tool_name(arg1=val1, ...)
            Response: <nội dung response>
            Artifact: <dữ liệu artifact nếu có>
        [Final Response]
        <câu trả lời cuối>
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Bạn là chuyên gia phân tích hội thoại bóng đá. "
            "Nhiệm vụ của bạn là đọc lịch sử hội thoại (bao gồm kết quả từ các tool calls) "
            "và tổng hợp thành một SessionMemory có cấu trúc, giữ lại đầy đủ thông tin quan trọng. "
            "Chỉ trích xuất thông tin có trong hội thoại, không thêm thông tin ngoài."
        )),
        HumanMessagePromptTemplate.from_template(
            """Đọc đoạn hội thoại bóng đá dưới đây và tạo ra một SessionMemory có cấu trúc.

## Lịch sử hội thoại cần tóm tắt:
{conversation_block}

---

## Hướng dẫn điền từng trường:

### conversation_state
Mô tả tiến trình hội thoại — người dùng đang hỏi về gì, đã giải quyết được gì, đang ở giai đoạn nào.
Ví dụ: "Người dùng hỏi về phong độ Chelsea mùa 2014-15. Hệ thống đã cung cấp thống kê trận thắng. Người dùng tiếp tục hỏi về Diego Costa."

### user_context.preferences
Sở thích thể thao được thể hiện rõ: đội bóng, giải đấu, cầu thủ, mùa giải, loại số liệu hay hỏi.

### user_context.goals
Mục tiêu thông tin cụ thể — người dùng muốn biết gì, phục vụ mục đích gì (tra cứu, so sánh, phân tích).

### confirmed_entities
Liệt kê các thực thể bóng đá đã được xác nhận với thông tin cụ thể đã tìm được.
Format: "Tên thực thể (loại) — thông tin chính"
Ví dụ: ["Chelsea FC (đội bóng) — EPL 2014-15, vô địch với 87 điểm, 26 trận thắng"]
Dùng chủ yếu để nhận diện thực thể khi người dùng dùng đại từ ở các turn sau.

### tool_findings
QUAN TRỌNG — Đây là trường cốt lõi. Trích xuất kết quả từ TỪNG tool call trong [Tool Usage].
Mỗi entry gồm:
- tool_name: tên tool đã gọi (vd: "textual_entity_search")
- input_summary: tóm tắt ngắn query/input của tool call đó
- key_facts: danh sách các thông tin/sự kiện quan trọng từ Response và Artifact của tool.
  Giữ đủ chi tiết để có thể tái sử dụng thông tin này mà không cần gọi lại tool.
  Mỗi fact là một câu ngắn gọn, độc lập, đầy đủ thông tin.
  Ví dụ: ["Diego Costa ghi 20 bàn mùa EPL 2014-15, dẫn đầu danh sách ghi bàn",
           "Chelsea sử dụng sơ đồ 4-2-3-1 với Hazard đóng vai trò số 10"]

### open_discussion_threads
Câu hỏi chưa trả lời đầy đủ, chủ đề đang bỏ lửng, điều người dùng có vẻ muốn hỏi tiếp.

### scope
Phạm vi bao quát ngắn gọn của cuộc hội thoại (chủ thể + khung thời gian nếu có).

---

## Output Format:
{format_instructions}
"""
        )
    ])


def get_update_prompt() -> ChatPromptTemplate:
    """
    Prompt để cập nhật incremental một SessionMemory đã có với thông tin từ turn mới.

    Đầu vào:
    - existing_memory: SessionMemory hiện tại (JSON)
    - turn_tool_summary: nội dung [Tool Usage] + [Final Response] của turn vừa xong
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Bạn là chuyên gia cập nhật bộ nhớ hội thoại bóng đá. "
            "Nhiệm vụ của bạn là nhận một SessionMemory đã có và thông tin từ một turn mới, "
            "sau đó trả về SessionMemory đã được cập nhật với thông tin mới được merge vào. "
            "Giữ nguyên thông tin cũ trừ khi thông tin mới thay thế hoặc mâu thuẫn với nó."
        )),
        HumanMessagePromptTemplate.from_template(
            """## SessionMemory hiện tại:
```json
{existing_memory}
```

## Thông tin từ turn mới:
{turn_tool_summary}

---

## Nhiệm vụ:
Cập nhật SessionMemory với thông tin mới theo các quy tắc sau:

1. **tool_findings**: Thêm các ToolFinding mới từ [Tool Usage] của turn này vào list.
   - Mỗi Step trong [Tool Usage] → 1 ToolFinding entry.
   - Trích xuất key_facts từ cả Response lẫn Artifact (nếu có).
   - Giữ facts đủ chi tiết để không cần gọi lại tool.
   - Không thêm duplicate nếu tool call giống hệt đã có trong tool_findings.

2. **confirmed_entities**: Thêm/cập nhật các thực thể mới được xác nhận từ tool results.
   - Format: "Tên thực thể (loại) — thông tin chính"
   - Cập nhật entry hiện có nếu có thêm thông tin mới cho cùng thực thể.

3. **conversation_state**: Cập nhật để phản ánh turn mới nhất.

4. **open_discussion_threads**:
   - Thêm câu hỏi/chủ đề mới nếu có.
   - Xóa thread đã được giải quyết trong turn này.

5. **user_context, scope**: Cập nhật nếu có thêm thông tin mới.

## Output Format:
{format_instructions}
"""
        )
    ])
