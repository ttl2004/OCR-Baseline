# OCR Baseline

Hệ thống trích xuất thông tin từ ảnh bao bì sản phẩm / tờ hướng dẫn sử dụng thuốc:
**OCR** (lấy văn bản từ ảnh) → **LLM** (tách văn bản thành các trường dữ liệu) → **người dùng kiểm tra, chỉnh sửa**
→ **lưu SQLite**.

## Mô hình và API sử dụng

| Thành phần | Công nghệ | Ghi chú |
|---|---|---|
| OCR qua API | **Google Cloud Vision API** (`DOCUMENT_TEXT_DETECTION`) | Cách thực hiện mặc định, cần mạng |
| OCR cục bộ: phát hiện vùng chữ | **PaddleOCR** `PP-OCRv5_server_det` | Chạy trên CPU, không cần GPU |
| OCR cục bộ: nhận dạng chữ | **VietOCR** `vgg_transformer` | Hỗ trợ đầy đủ tiếng Việt có dấu |
| OCR cục bộ: tự xoay ảnh | PaddleOCR `PP-LCNet_x1_0_doc_ori` | Nhận biết ảnh xoay 0/90/180/270° |
| Tách thông tin | **DeepSeek** (`deepseek-chat`, API tương thích OpenAI) | Chế độ JSON, temperature 0 |
| Giao diện | Streamlit | |
| Lưu trữ | SQLite | |

## Kiến trúc

```
                ┌───────────────────────────────────────────────┐
   Ảnh ───────► │ OCR (chọn cách thực hiện)                     │ ──► Văn bản
                │  • Google Cloud Vision API                    │
                │  • OCR cục bộ: tiền xử lý → PaddleOCR         │
                │    (phát hiện) → thứ tự đọc → VietOCR         │
                └───────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌───────────────────────────────────────────────┐
                │ LLM (DeepSeek) + prompt theo loại tài liệu    │ ──► Các trường (JSON)
                │ + kiểm tra con số không có trong văn bản OCR  │
                └───────────────────────────────────────────────┘
                                       │
                                       ▼
                    Người dùng kiểm tra / chỉnh sửa ──► SQLite
```

- Mọi engine OCR có cùng giao diện `run(ảnh) -> văn bản`, nên đổi cách OCR không ảnh hưởng tới bước LLM và lưu trữ.
- Không gọi được Google Cloud Vision (mất mạng, sai key, hết quota)? Hệ thống hiện thông báo kèm nút
  **chuyển sang OCR cục bộ**.
- **Loại tài liệu (profile)** quy định danh sách trường cần lấy và prompt gốc. Hiện có:
  - bao bì thực phẩm: tên, thành phần, phụ gia, cảnh báo, dinh dưỡng, xuất xứ, khối lượng, ngày sản xuất, hạn dùng;
  - tờ hướng dẫn sử dụng thuốc: 16 trường, gồm số đăng ký, hoạt chất, chỉ định, liều dùng, chống chỉ định...

## Luồng xử lý

### Chế độ 1: Tải ảnh lên

Dùng cho một sản phẩm. Có thể tải nhiều ảnh, tương ứng các mặt hoặc các trang của cùng sản phẩm.

1. Chọn ảnh → **OCR**, xem văn bản từng ảnh và các dòng có độ tin cậy thấp.
2. Sửa văn bản nếu cần → chọn prompt → **gửi LLM** để tách trường.
3. Kiểm tra, sửa các trường (kèm cảnh báo con số đáng ngờ) → **lưu**. Mã bản ghi có dạng `UP-xxxxxx`.

### Chế độ 2: Chạy thư mục ảnh (dataset)

Dùng cho nhiều sản phẩm. Mỗi ảnh trong thư mục là một sản phẩm.

1. Nhập đường dẫn thư mục, đặt tên dataset, chọn prompt cho dataset.
2. Hệ thống tự chạy OCR + LLM cho từng ảnh và lưu ngay khi xong. Mã bản ghi có dạng `DS-xxxxxx`, kèm tên dataset.
3. Dừng giữa chừng vẫn **chạy tiếp được**: ảnh đã xử lý sẽ được bỏ qua.
   - Ảnh lỗi được đánh dấu riêng để xử lý lại.
   - Mất kết nối Google Vision: có thể chuyển sang OCR cục bộ và chạy nốt.
4. Kiểm tra, chỉnh sửa kết quả ở trang **Xem & chỉnh sửa dữ liệu**.

### Xem & chỉnh sửa dữ liệu

- Lọc theo chế độ, dataset, trạng thái, loại tài liệu, prompt.
- Xem ảnh, sửa các trường hoặc văn bản OCR rồi lưu. Bản ghi chuyển sang trạng thái *Đã duyệt*.
- Chạy lại LLM (có thể chọn prompt khác) hoặc chạy lại cả OCR + LLM. Xoá bản ghi, xuất CSV.

### Quản lý prompt

- Prompt quản lý theo **phiên bản**: mỗi lần sửa tạo phiên bản mới, không sửa đè.
- Mỗi dataset gắn một prompt. Mỗi bản ghi lưu rõ prompt và model LLM đã tạo ra nó.
- Prompt gốc lấy từ code, không sửa được trên giao diện. Muốn thay đổi thì tạo prompt mới từ prompt gốc.
- Có thể thử prompt trên một bản ghi có sẵn trước khi dùng.

## Dữ liệu (SQLite)

| Bảng | Nội dung chính |
|---|---|
| `records` | Mỗi dòng là một sản phẩm gồm:<br>• mã bản ghi (`UP-`/`DS-`), chế độ (`UPLOAD`/`DATASET`), tên dataset (chỉ chế độ 2);<br>• loại tài liệu, cách OCR, prompt và model LLM đã dùng;<br>• văn bản OCR, các trường (JSON), cảnh báo, trạng thái (*chưa duyệt* / *đã duyệt* / *lỗi*). |
| `datasets` | Tên dataset, thư mục, loại tài liệu, prompt đang gắn |
| `prompts` | Tên, phiên bản, loại tài liệu, nội dung, ghi chú, prompt gốc / đã ẩn |

## Cấu trúc thư mục

```
app.py                       Điểm vào Streamlit (chỉ chọn trang)
ui/                          Giao diện
  sidebar.py                 Chọn chế độ, cách thực hiện, loại tài liệu
  page_upload.py             Chế độ 1
  page_dataset.py            Chế độ 2
  page_records.py            Xem & chỉnh sửa dữ liệu
  page_prompts.py            Quản lý prompt
  components.py, resources.py, state.py
src/ocr_baseline/            Logic xử lý (độc lập với giao diện)
  config.py                  Đọc cấu hình từ .env
  constants.py               Mã chế độ, cách thực hiện, trạng thái
  pipeline.py                Các bước dùng chung: OCR ảnh, ghép văn bản, trích xuất
  prompts.py                 Prompt có phiên bản
  ocr/
    vision.py                Google Cloud Vision
    local/                   OCR cục bộ: tiền xử lý, PaddleOCR, VietOCR, thứ tự đọc
  profiles/                  Loại tài liệu: danh sách trường + prompt gốc (food.py, drug.py)
  extraction/                Gọi LLM, kiểm tra con số
  services/                  Chạy thư mục ảnh (batch), đọc / lưu ảnh
  storage/                   SQLite: định nghĩa bảng, đọc / ghi
tests/                       Kiểm thử tự động (không gọi API)
```

## Cài đặt và chạy

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --no-deps vietocr==0.3.13
.venv/bin/pip install -e ".[dev]"
cp .env.example .env          # điền DEEPSEEK_API_KEY, GOOGLE_VISION_API_KEY

.venv/bin/streamlit run app.py
.venv/bin/python -m pytest
```

- Lần đầu dùng OCR cục bộ, hệ thống tự tải model (PaddleOCR vào `~/.paddlex/`, VietOCR vào `MODELS_DIR`).
- Các tuỳ chọn khác như số luồng CPU, độ phân giải phát hiện chữ, cách thực hiện và loại tài liệu mặc định
  được mô tả trong `.env.example`.

## Mở rộng

| Muốn | Sửa ở |
|---|---|
| Sửa quy ước tách trường | Trang **Quản lý prompt** (không cần sửa code) |
| Thêm / bớt trường | `src/ocr_baseline/profiles/` |
| Thêm loại tài liệu | Thêm file trong `profiles/`, đăng ký trong `profiles/__init__.py` |
| Thêm cách OCR | Class có `run(ảnh) -> OcrResult` trong `ocr/`, thêm mã vào `constants.py` và `ocr/__init__.py` |

## Lưu ý

- Google Cloud Vision gửi ảnh lên máy chủ Google và cần mạng. OCR cục bộ chạy hoàn toàn trên máy.
- Thông tin do hệ thống trích xuất cần được người dùng kiểm tra trước khi sử dụng.
