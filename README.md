# OCR Baseline

Ảnh bao bì / tờ thuốc → **OCR** → **LLM tách thành các cột** → **SQLite** → người dùng xem và chỉnh sửa.

Hai cách thực hiện, chọn trên giao diện:

| Cách thực hiện | OCR | Kết quả đánh giá (100 ảnh ViFoodLabel) |
|---|---|---|
| **Google Cloud Vision + LLM** (mặc định) | Google Cloud Vision API, cần mạng | OCR 95.7%, F1 tổng thể **83.7%**, ~3 s/ảnh |
| **OCR local + LLM** | PaddleOCR + VietOCR chạy CPU, không gửi ảnh ra ngoài | OCR 82.6%, F1 tổng thể 69.9%, ~11 s/ảnh |

Chi tiết đánh giá: `../OCR/docs/metrics/2026-11-14-llm-ocr-local-vs-vision.md`.
LLM trong cả 2 cách là DeepSeek (`deepseek-chat`). Nếu không gọi được Google Vision (mất mạng, sai key,
hết quota), app hiện thông báo kèm nút **"Chuyển sang OCR local"**.

## Hai chế độ

| | Chế độ 1 - Tải ảnh lên | Chế độ 2 - Thư mục ảnh (dataset) |
|---|---|---|
| Đầu vào | Một hoặc nhiều ảnh của **cùng một sản phẩm** (nhiều mặt) | Thư mục, **mỗi ảnh là một sản phẩm** |
| Xử lý | Từng bước: OCR → sửa text → LLM → sửa cột → lưu | Tự chạy hết thư mục, lưu từng ảnh ngay khi xong |
| Prompt | Chọn trước khi gửi LLM | Chọn cho dataset (lần sau mặc định chọn lại prompt của dataset) |
| Mã bản ghi | `UP-000001` | `DS-000002` |
| Cột `mode` | `UPLOAD` | `DATASET` |
| Cột `dataset` | `NULL` | tên dataset (bắt buộc) |
| Trạng thái khi lưu | Đã duyệt | Chưa duyệt → chỉnh ở trang "Xem & chỉnh sửa dữ liệu" |

Chế độ 2:

- **Chạy tiếp được.** Dừng giữa chừng (bấm nút khác, đóng trang) rồi bấm "Bắt đầu / chạy tiếp"; ảnh đã xử lý được bỏ qua.
- **Lỗi của một ảnh** (ảnh hỏng, LLM trả lỗi) được ghi trạng thái "Lỗi" và app chạy tiếp ảnh khác.
  Tick "Xử lý lại các ảnh bị lỗi" để chạy lại riêng những ảnh này.
- **Mất kết nối Google Vision** thì app dừng ngay. Nút "Chuyển sang OCR local và chạy tiếp" chạy nốt các ảnh còn lại bằng OCR local.
- **Tên dataset** gắn với một thư mục và một loại tài liệu. Muốn chạy lại cùng thư mục với cấu hình khác thì đặt tên dataset khác.
- **Mỗi dataset gắn một prompt** (`datasets.prompt_id`). Nếu chạy tiếp với prompt khác, app cảnh báo và yêu cầu
  xác nhận. Bản ghi cũ giữ prompt cũ; ảnh xử lý tiếp dùng prompt mới. Muốn so sánh 2 prompt thì chạy cùng
  thư mục với 2 tên dataset khác nhau.

Trang **Xem & chỉnh sửa dữ liệu**:

- lọc theo chế độ, dataset, trạng thái, loại tài liệu;
- chọn một dòng để xem ảnh, sửa các cột hoặc sửa text OCR, rồi bấm **Lưu** (bản ghi chuyển sang Đã duyệt);
- **Chạy lại LLM** (dùng text OCR đã sửa) hoặc **Chạy lại OCR + LLM** (dùng cách thực hiện đang chọn), đều được chọn prompt.
  Mặc định là prompt của dataset; bản ghi được cập nhật sang prompt vừa dùng;
- lọc theo prompt; tab "Prompt đã dùng" hiện đúng nội dung prompt đã tạo ra bản ghi;
- xoá bản ghi, xuất CSV các bản ghi đang lọc.

## Quản lý prompt

Trang **Quản lý prompt** (theo loại tài liệu đang chọn ở thanh bên):

- **Prompt có phiên bản, không bị sửa đè.** Mỗi lần "Lưu" tạo một phiên bản mới: giữ tên thì version + 1,
  đổi tên thì thành prompt mới. Bản ghi lưu `prompt_id` nên luôn biết **chính xác nội dung prompt** đã tạo ra nó.
- **Prompt gốc** lấy từ code (`profiles/*.py`); prompt thực phẩm gốc là prompt v2 đã đánh giá. Prompt gốc không
  sửa được trên giao diện: muốn thay đổi thì lưu với tên khác. Nếu sửa prompt gốc trong code, khi khởi động app
  tự thêm phiên bản mới.
- **Thử prompt**: chạy trên text OCR của một bản ghi có sẵn (so sánh với kết quả đã lưu) hoặc text tự dán.
  Kết quả thử không được lưu.
- **Kiểm tra cột**: app cảnh báo nếu prompt không nhắc tới cột nào của loại tài liệu.
- **Ẩn** prompt khỏi danh sách chọn. **Xoá** chỉ được với prompt chưa có bản ghi hay dataset nào dùng.

## Cài đặt và chạy

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --no-deps vietocr==0.3.13      # deps gốc của vietocr ghim bản cũ
.venv/bin/pip install -e ".[dev]"
cp .env.example .env                                 # điền DEEPSEEK_API_KEY, GOOGLE_VISION_API_KEY

.venv/bin/streamlit run app.py
.venv/bin/python -m pytest                           # test (không gọi API, không cần model)
```

Trên máy dev hiện tại có thể dùng luôn môi trường của project OCR (đã cài sẵn Paddle, torch, VietOCR):

```bash
../OCR/.venv/bin/pip install -e . --no-deps     # đã chạy
../OCR/.venv/bin/streamlit run app.py
```

Lần đầu dùng OCR local, PaddleOCR sẽ tải model về `~/.paddlex/`. VietOCR dùng thư mục `MODELS_DIR`
(file `.env` hiện trỏ tới `../OCR/models`).

## Cấu trúc

```
app.py                       Điểm vào Streamlit: chỉ chọn trang
ui/                          Giao diện (không chứa logic xử lý)
  sidebar.py                 Chọn chế độ, cách thực hiện, loại tài liệu; trạng thái cấu hình
  page_upload.py             Chế độ 1
  page_dataset.py            Chế độ 2
  page_records.py            Xem & chỉnh sửa dữ liệu
  page_prompts.py            Quản lý prompt
  components.py              Ô chọn prompt, form sửa các cột, cảnh báo con số, thông báo chuyển sang OCR local
  resources.py               Engine OCR, Repository (tạo một lần, dùng lại)
  state.py                   Key session_state dùng chung, callback của nút
src/ocr_baseline/            Logic xử lý (dùng được cả ngoài giao diện, vd viết API sau này)
  config.py                  Đọc .env -> get_settings()
  constants.py               Mã chế độ (UPLOAD/DATASET), cách thực hiện, trạng thái
  pipeline.py                Bước dùng chung: ocr_image, join_pages, extract(text, profile, prompt)
  prompts.py                 Prompt (có phiên bản), prompt gốc từ profile
  ocr/
    __init__.py              create_engine(method, settings)
    vision.py                Google Cloud Vision
    local/                   OCR local: engine.py, preprocess.py, vietocr_rec.py, layout.py, geometry.py
    errors.py                OcrUnavailableError (dừng, gợi ý đổi cách) / OcrImageError (lỗi riêng ảnh)
    types.py                 OcrLine, OcrResult
  profiles/                  Loại tài liệu = danh sách cột + prompt gốc
    food.py                  Bao bì thực phẩm (prompt gốc = prompt v2 đã đánh giá)
    drug.py                  Tờ hướng dẫn sử dụng thuốc (16 cột)
  extraction/
    llm.py                   Gọi LLM, chuẩn hoá output theo profile
    checks.py                Phát hiện con số LLM đưa ra mà không có trong text OCR
  services/
    batch.py                 Chế độ 2: lập kế hoạch, chạy từng ảnh, chạy tiếp, ghi lỗi
    images.py                Đọc ảnh, liệt kê thư mục, lưu bản sao ảnh tải lên
  storage/
    database.py              Định nghĩa bảng SQLite + tự nâng cấp DB cũ (thêm cột)
    repository.py            Đọc / ghi (mọi câu SQL nằm ở đây)
tests/                       Test (engine và LLM giả, không gọi API)
data/                        baseline.db + uploads/ (không commit)
```

## Cơ sở dữ liệu (`data/baseline.db`)

Bảng `records`: mỗi dòng là một sản phẩm.

| Cột | Ý nghĩa |
|---|---|
| `code` | Mã bản ghi: `UP-xxxxxx` (chế độ 1) / `DS-xxxxxx` (chế độ 2). Số = id, tăng dần chung |
| `mode` | `UPLOAD` / `DATASET` |
| `dataset` | Tên dataset (chỉ chế độ 2; DB ràng buộc: DATASET bắt buộc có, UPLOAD phải NULL) |
| `source_key` | Chế độ 2: tên file ảnh trong thư mục. Mỗi ảnh chỉ có một bản ghi trong một dataset |
| `image_paths` | JSON: đường dẫn ảnh (chế độ 1: bản sao trong `data/uploads/`) |
| `profile` | Loại tài liệu: `food` / `drug` |
| `ocr_method` | Cách OCR đã dùng: `vision` / `local` |
| `prompt_id` | Prompt đã tạo ra các cột (→ bảng `prompts`). NULL nếu chưa chạy được LLM (lỗi OCR) |
| `llm_model` | Model LLM đã dùng |
| `title` | Tên sản phẩm / tên thuốc |
| `ocr_text` | Text OCR (đã sửa nếu người dùng sửa) |
| `fields_json` | JSON: các cột thông tin theo profile |
| `warnings_json` | JSON: con số LLM đưa ra mà không có trong text OCR |
| `status` | `auto` (chưa duyệt) / `reviewed` (đã duyệt) / `error` |
| `error_message`, `ocr_seconds`, `llm_seconds`, `created_at`, `updated_at` | |

Bảng `datasets`: `name`, `folder`, `profile`, `prompt_id` (prompt gắn với dataset), `created_at`.

Bảng `prompts`: `id`, `name`, `version` (UNIQUE(name, version)), `profile`, `content`, `note`,
`is_builtin` (prompt gốc từ code), `archived` (ẩn), `created_at`. Không bao giờ sửa đè nội dung.

## Sửa / mở rộng thường gặp

| Muốn | Sửa ở |
|---|---|
| Sửa quy ước tách cột (prompt) | Trang **Quản lý prompt** (tạo phiên bản mới), không cần sửa code |
| Thêm / bớt cột | `profiles/food.py` hoặc `profiles/drug.py` (DB không cần migrate vì cột lưu dạng JSON). Nhớ sửa prompt cho khớp |
| Thêm loại tài liệu mới | Thêm file trong `profiles/`, đăng ký trong `profiles/__init__.py` |
| Thêm cách OCR mới | Class có `run(img) -> OcrResult` trong `ocr/`, thêm mã vào `constants.py`, thêm nhánh trong `ocr/__init__.py` |
| Đổi model LLM, số luồng, độ phân giải detection | `.env` |
| Thêm API (FastAPI...) | Gọi `create_engine`, `pipeline.ocr_image`, `pipeline.extract`, `Repository` giống `ui/` |

## Lưu ý

- Prompt thực phẩm gốc giống hệt "prompt v2" đã đánh giá (F1 83.7%). Prompt mới tạo trên giao diện chưa được đo;
  nên đo lại bằng `../OCR/evaluation/` trước khi dùng cho dữ liệu thật.
- Profile `drug` chưa có dataset có nhãn để đo độ chính xác.
- Google Cloud Vision gửi ảnh lên máy chủ Google. Giá: 1000 ảnh/tháng miễn phí, sau đó khoảng 1.5 USD/1000 ảnh.
- OCR local: đặt `OCR_THREADS` bằng số nhân thật của máy. Nếu Paddle chạy rất chậm (CPU có nhân E),
  đặt `OCR_ENABLE_MKLDNN=false`.
