"""Các mã dùng chung trong toàn bộ project (lưu trong DB, hiển thị trên giao diện)."""

# Chế độ nhập dữ liệu -> cột `mode` trong bảng records
MODE_UPLOAD = "UPLOAD"     # Chế độ 1: người dùng tự tải ảnh lên (nhiều ảnh = nhiều mặt của 1 sản phẩm)
MODE_DATASET = "DATASET"   # Chế độ 2: chạy cả thư mục, mỗi ảnh là 1 sản phẩm

MODE_LABELS = {
    MODE_UPLOAD: "Chế độ 1 - Tải ảnh lên",
    MODE_DATASET: "Chế độ 2 - Thư mục ảnh (dataset)",
}

# Tiền tố mã bản ghi: UP-000001, DS-000002
CODE_PREFIX = {
    MODE_UPLOAD: "UP",
    MODE_DATASET: "DS",
}

# Cách thực hiện (hướng OCR) -> cột `ocr_method`
METHOD_VISION = "vision"   # Google Cloud Vision + LLM
METHOD_LOCAL = "local"     # OCR local (PaddleOCR + VietOCR, CPU) + LLM

METHOD_LABELS = {
    METHOD_VISION: "Google Cloud Vision + LLM (chính xác nhất, cần mạng)",
    METHOD_LOCAL: "OCR local + LLM (chạy trên máy, không gửi ảnh ra ngoài)",
}

# Trạng thái bản ghi -> cột `status`
STATUS_AUTO = "auto"           # máy tự xử lý, chưa có người kiểm tra
STATUS_REVIEWED = "reviewed"   # người dùng đã kiểm tra / chỉnh sửa
STATUS_ERROR = "error"         # xử lý lỗi (xem cột error_message)

STATUS_LABELS = {
    STATUS_AUTO: "Chưa duyệt",
    STATUS_REVIEWED: "Đã duyệt",
    STATUS_ERROR: "Lỗi",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
