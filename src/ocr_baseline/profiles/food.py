"""Profile: nhãn bao bì thực phẩm (giống dataset ViFoodLabel).

Prompt dưới đây là "prompt v2" đã đánh giá trên 100 ảnh ViFoodLabel:
Vision + prompt v2 đạt F1 tổng thể 83.7% (xem OCR/docs/metrics/2026-11-14-llm-ocr-local-vs-vision.md).
Các quy ước phân cột (mục 5-9) rút ra từ 500 nhãn của dataset. Sửa quy ước nào thì nên đo lại.
"""
from .base import KIND_LIST, KIND_NUTRITION, KIND_TEXT, Field, Profile

FIELDS = (
    Field("product_name", "Tên sản phẩm", KIND_TEXT),
    Field("ingredient", "Thành phần", KIND_LIST),
    Field("additive", "Phụ gia", KIND_LIST),
    Field("warning", "Cảnh báo / dị ứng", KIND_LIST),
    Field("nutrition", "Giá trị dinh dưỡng", KIND_NUTRITION),
    Field("origin", "Xuất xứ", KIND_TEXT),
    Field("net_weight", "Khối lượng tịnh", KIND_TEXT),
    Field("mfg_date", "Ngày sản xuất", KIND_TEXT),
    Field("expiry_date", "Hạn sử dụng", KIND_TEXT),
)

SYSTEM_PROMPT = """Bạn TRÍCH XUẤT thông tin từ văn bản OCR của nhãn bao bì thực phẩm tại Việt Nam.

Quy tắc:
1. CHỈ lấy thông tin có trong văn bản, không suy luận, không thêm kiến thức bên ngoài.
2. Không có thông tin thì để null (trường chuỗi) hoặc [] (trường danh sách).
3. Văn bản OCR có thể sai chính tả, mất dấu, ngắt dòng lung tung, lẫn dòng của cột bên cạnh.
   Được sửa lỗi OCR hiển nhiên, nhưng KHÔNG được đổi con số, đơn vị.
4. Trả về đúng một JSON object:
{
  "product_name": tên sản phẩm,
  "ingredient": [từng thành phần nguyên liệu, giữ phần ngoặc và tỉ lệ nếu có],
  "additive": [từng phụ gia: chất bảo quản, chất điều vị, chất tạo màu, mã INS...; KHÔNG lặp lại trong ingredient],
  "warning": [từng câu cảnh báo / thông tin dị ứng],
  "nutrition": [{"name": tên chất dinh dưỡng, "value": giá trị kèm đơn vị}],
  "origin": xuất xứ,
  "net_weight": khối lượng tịnh / thể tích thực,
  "mfg_date": ngày sản xuất (hoặc chỉ dẫn, vd "Xem trên bao bì"),
  "expiry_date": hạn sử dụng (hoặc chỉ dẫn)
}

Quy ước phân cột (bắt buộc tuân theo):
5. PHỤ GIA (additive) gồm mọi chất có chức năng công nghệ: chất ổn định, chất nhũ hóa, chất điều chỉnh độ acid,
   chất điều vị, chất bảo quản, chất chống oxy hóa, chất làm dày, chất tạo xốp, chất làm ẩm, chất tạo màu / phẩm màu,
   chất tạo ngọt (tổng hợp hoặc có mã INS), chất chống đông vón, chất xử lý bột, chất làm bóng, và MỌI LOẠI HƯƠNG LIỆU
   (hương liệu tổng hợp / tự nhiên / giống tự nhiên). Các chất này KHÔNG đưa vào ingredient.
6. Mỗi nhóm phụ gia là MỘT mục, giữ nguyên tên nhóm và toàn bộ mã trong ngoặc, KHÔNG tách từng mã.
   Ví dụ: "Chất nhũ hóa (471, 433, 322(i))" là 1 mục; "Hương liệu tổng hợp (vani, sữa)" là 1 mục.
7. THÀNH PHẦN (ingredient): mỗi nguyên liệu cấp ngoài cùng là một mục, giữ tỉ lệ % / g/kg.
   Nguyên liệu có ngoặc chứa thành phần con giữ nguyên thành MỘT mục, vd "Shortening thực vật (chứa chất nhũ hóa (471))";
   phụ gia nằm trong ngoặc đó vẫn liệt kê thêm ở additive. Muối, vitamin, khoáng chất là ingredient.
8. CẢNH BÁO (warning): mỗi câu cảnh báo / thông tin dị ứng giữ nguyên cả câu là một mục, không tách danh sách chất dị ứng.
9. Dinh dưỡng: name ghi như trên bao bì; value gồm con số và đơn vị (vd "6,6 g"), lấy đơn vị từ tiêu đề cột nếu cần."""

PROFILE = Profile(
    key="food",
    label="Bao bì thực phẩm",
    fields=FIELDS,
    system_prompt=SYSTEM_PROMPT,
    default_prompt_name="Thực phẩm - prompt v2 (gốc, đã đánh giá)",
    title_field="product_name",
)
