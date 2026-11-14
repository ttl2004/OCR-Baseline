"""Lỗi của bước OCR.

Phân biệt 2 loại vì cách xử lý khác nhau:
- OcrUnavailableError: dịch vụ OCR không dùng được (mất mạng, sai/thiếu key, hết quota...).
  Ảnh nào cũng sẽ lỗi -> dừng lại, báo người dùng chuyển sang OCR local.
- OcrImageError: chỉ ảnh này lỗi (ảnh hỏng, không đọc được) -> ghi lỗi cho ảnh đó, chạy tiếp ảnh khác.
"""


class OcrError(RuntimeError):
    pass


class OcrUnavailableError(OcrError):
    pass


class OcrImageError(OcrError):
    pass
