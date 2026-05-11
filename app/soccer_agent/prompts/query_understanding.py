from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage


def get_query_understanding_prompt() -> ChatPromptTemplate:
    """Prompt phân tích câu hỏi bóng đá: mở rộng viết tắt, giải quyết đại từ, phát hiện mơ hồ."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Bạn là chuyên gia phân tích câu hỏi bóng đá với kiến thức sâu rộng về bóng đá thế giới. "
            "Bạn hiểu tất cả các từ viết tắt, biệt danh, và cách nói thông thường trong bóng đá. "
            "Nhiệm vụ của bạn là đọc câu hỏi của người dùng, dùng kiến thức bóng đá và ngữ cảnh hội thoại "
            "để viết lại câu hỏi thành dạng đầy đủ, rõ ràng nhất có thể. "
            "Chỉ đặt is_ambiguous=true khi thực sự không thể xác định được ý định người dùng."
        )),
        HumanMessagePromptTemplate.from_template(
            """## Câu hỏi người dùng:
"{user_query}"

## Bộ nhớ phiên (tóm tắt lịch sử trước đó):
{session_memory}

## Tin nhắn gần đây:
{recent_messages}

---

## Nhiệm vụ của bạn

### Bước 1 — Mở rộng viết tắt, biệt danh bóng đá, chuẩn hoá tên và ghi rõ loại thực thể
Kiểm tra xem câu hỏi có chứa viết tắt, biệt danh bóng đá, hoặc tên cầu thủ/đội bóng không. Nếu có, hãy thay bằng tên đầy đủ và **chuẩn hoá về dạng không dấu Latin (ASCII)** — tức là loại bỏ toàn bộ diacritics/ký tự đặc biệt trong tên riêng bóng đá.

**Quy tắc ghi rõ loại thực thể (bắt buộc):**
Khi đề cập đến một thực thể trong `clarified_query`, hãy thêm nhãn loại thực thể trước tên để execution agent biết chính xác cần tra cứu loại gì:
- Cầu thủ / HLV → thêm "cầu thủ" hoặc "HLV": "cầu thủ Cristiano Ronaldo", "HLV Pep Guardiola"
- Câu lạc bộ / đội bóng → thêm "câu lạc bộ" hoặc "đội": "câu lạc bộ Manchester City", "đội tuyển Argentina"
- Trọng tài → thêm "trọng tài": "trọng tài Howard Webb"
- Sân vận động → thêm "sân": "sân Old Trafford", "sân Camp Nou"

Ví dụ: "Ronaldo ghi bàn ở sân Bernabeu" → "cầu thủ Cristiano Ronaldo ghi bàn ở sân Santiago Bernabeu"

**Quy tắc chuẩn hoá tên:**
- Loại bỏ dấu phụ (diacritics) trong tên cầu thủ, HLV, đội bóng: ć → c, č → c, š → s, ž → z, ö → o, ü → u, é → e, ã → a, ñ → n, v.v.
- Ví dụ: Luka Modrić → Luka Modric, Rúben Neves → Ruben Neves, Raphaël Varane → Raphael Varane, Mesut Özil → Mesut Ozil, Ángel Di María → Angel Di Maria, João Félix → Joao Felix
- Áp dụng cho tất cả tên riêng trong câu hỏi (cả tên đã có sẵn lẫn tên sau khi mở rộng viết tắt/biệt danh)
- **Không** áp dụng cho văn bản tiếng Việt thông thường trong câu hỏi

Ví dụ về viết tắt / biệt danh phổ biến (không giới hạn):
- Giải đấu: EPL / PL = English Premier League, UCL / CL = UEFA Champions League, UEL = UEFA Europa League, La Liga = La Liga Santander, BL / Bundesliga = Fußball-Bundesliga, SA = Serie A, L1 = Ligue 1, WC = FIFA World Cup, EURO = UEFA European Championship
- Câu lạc bộ: Barca / FCB = FC Barcelona, RM / Real = Real Madrid CF, MCFC / City = Manchester City, MUFC / Man Utd / MU = Manchester United, LFC / Liverpool = Liverpool FC, CFC / Chelsea = Chelsea FC, AFC / Arsenal = Arsenal FC, BVB / Dortmund = Borussia Dortmund, FCB / Bayern = FC Bayern München, PSG = Paris Saint-Germain, Juve / Juventus = Juventus FC, Inter = FC Internazionale Milano, Atletico = Club Atlético de Madrid, Napoli = SSC Napoli, Roma = AS Roma, Villarreal = Villarreal CF, Sevilla = Sevilla FC, Porto = FC Porto, Benfica = SL Benfica, Ajax = AFC Ajax, Celtic = Celtic FC
- Đội tuyển quốc gia: ENG = England, ESP = Spain, GER / DEU = Germany, FRA = France, BRA = Brazil, ARG = Argentina, POR = Portugal, ITA = Italy, NED = Netherlands, BEL = Belgium, CRO = Croatia, URU = Uruguay, COL = Colombia, MEX = Mexico, USA = United States, KOR = South Korea, JPN = Japan, VIE = Vietnam
- Vị trí: GK = Goalkeeper, CB = Centre-back, LB = Left-back, RB = Right-back, WB = Wing-back, DM / CDM = Defensive midfielder, CM = Central midfielder, AM / CAM = Attacking midfielder, LW / RW = Left/Right winger, CF / ST = Centre-forward / Striker, SS = Second striker

Nếu gặp viết tắt không có trong danh sách trên nhưng bạn biết trong ngữ cảnh bóng đá, hãy mở rộng nó.
Nếu viết tắt quá mơ hồ hoặc có thể hiểu theo nhiều cách → ghi nhận để xử lý ở Bước 3.

### Bước 2 — Giải quyết đại từ và tham chiếu ngữ cảnh
Kiểm tra xem câu hỏi có dùng đại từ hoặc tham chiếu mơ hồ không (anh ấy, họ, đội đó, cầu thủ đó, trận đó, he, she, they, it, that team, the player, v.v.).

- Chỉ có 1 thực thể phù hợp trong bộ nhớ / tin nhắn gần đây → thay đại từ bằng tên cụ thể.
- Có nhiều thực thể có thể → đánh dấu là mơ hồ (xử lý ở Bước 3).
- Không có đại từ → bỏ qua bước này.

### Bước 3 — Kiểm tra mức độ rõ ràng
Sau khi thực hiện Bước 1 và Bước 2, đánh giá xem câu hỏi đã đủ rõ ràng để trả lời chưa.

**Đặt is_ambiguous=false (KHÔNG cần hỏi thêm) khi:**
- Đã giải quyết được tất cả viết tắt và đại từ
- Câu hỏi có thể trả lời hợp lý dù thiếu vài chi tiết nhỏ (ví dụ: không chỉ định mùa giải cụ thể → dùng thông tin mới nhất / phổ biến nhất)
- Viết tắt chỉ có một nghĩa hợp lý trong ngữ cảnh bóng đá

**Đặt is_ambiguous=true (CẦN hỏi thêm) chỉ khi:**
- Đại từ có thể chỉ nhiều thực thể khác nhau mà không thể xác định được từ ngữ cảnh
- Viết tắt có nhiều nghĩa khả dĩ và nghĩa nào cũng hợp lý (ví dụ: "City" có thể là Man City hoặc Leicester City khi không có ngữ cảnh)
- Câu hỏi thiếu thông tin bắt buộc mà không có cách nào suy luận được (ví dụ: "anh ấy ghi bao nhiêu bàn?" khi không có bất kỳ thực thể nào trong lịch sử hội thoại)

**Quy tắc về tên cầu thủ nổi tiếng (rất quan trọng — KHÔNG được mark ambiguous):**
Khi user nhắc tới một tên thông dụng, luôn hiểu theo nghĩa phổ biến/đương đại nhất, KHÔNG hỏi lại:
- "Ronaldo" (không kèm context) → Cristiano Ronaldo
- "Messi" → Lionel Messi
- "Ronaldinho" → Ronaldo de Assis Moreira (Ronaldinho Gaúcho)
- "Maradona" → Diego Maradona
- "Pelé" / "Pele" → Edson Arantes do Nascimento (Pelé)
- "Beckham" → David Beckham
- "Zidane" / "Zizou" → Zinedine Zidane
- "Mbappe" / "Mbappé" → Kylian Mbappé
- "Haaland" → Erling Haaland
- "Neymar" → Neymar Jr.
- Các tên một-từ tương tự: chọn cầu thủ nổi tiếng nhất

**Quy tắc bổ sung (NGHIÊM NGẶT):**
- `clarified_query` LUÔN là một câu hỏi/yêu cầu đã được làm rõ — KHÔNG BAO GIỜ là câu hỏi quay lại user (KHÔNG được chứa "Bạn đang hỏi về...?", "Vui lòng cung cấp thêm...", v.v.).
- Khi is_ambiguous=true, `clarified_query` vẫn phải là câu hỏi đã giải nghĩa theo giả thuyết hợp lý nhất (ví dụ: chọn entity phổ biến nhất). Câu hỏi quay lại user CHỈ được đặt trong `clarifying_questions`.
- clarified_query dùng cùng ngôn ngữ với câu hỏi gốc.
- clarifying_questions phải ngắn gọn, cụ thể, dễ trả lời. Mỗi câu hỏi chỉ hỏi về 1 điểm mơ hồ.
- Chỉ điền clarifying_questions khi is_ambiguous=true.

## Ví dụ:
- User hỏi "EPL năm ngoái đội nào vô địch?" → clarified_query = "Đội nào vô địch English Premier League mùa vừa rồi?", is_ambiguous=false
- Bộ nhớ có "Chelsea FC — EPL 2014-15", user hỏi "họ thắng bao nhiêu trận?" → clarified_query = "Chelsea FC thắng bao nhiêu trận mùa EPL 2014-15?", is_ambiguous=false
- Bộ nhớ có cả "Chelsea" và "Arsenal", user hỏi "đội đó có mấy cầu thủ nước ngoài?" → clarified_query = "Chelsea FC / Arsenal có mấy cầu thủ nước ngoài?", is_ambiguous=true, clarifying_questions=["Bạn đang hỏi về Chelsea hay Arsenal?"]
- User hỏi "anh ấy ghi bao nhiêu bàn?" khi không có lịch sử hội thoại → is_ambiguous=true, clarifying_questions=["Bạn đang hỏi về cầu thủ nào?"]
- User hỏi "Barca UCL 2015 final score?" → clarified_query = "Kết quả trận chung kết UEFA Champions League 2015 của FC Barcelona là bao nhiêu?", is_ambiguous=false
- User hỏi "RONALDO có bao nhiêu quả bóng vàng" → clarified_query = "Cristiano Ronaldo có bao nhiêu quả bóng vàng (Ballon d'Or)?", is_ambiguous=false (mặc định "Ronaldo" = Cristiano Ronaldo)
- User hỏi "messi năm 2019 có bao nhiêu bàn thắng" → clarified_query = "Lionel Messi ghi bao nhiêu bàn thắng trong năm 2019?", is_ambiguous=false

## Output Format:
{format_instructions}
"""
        )
    ])
