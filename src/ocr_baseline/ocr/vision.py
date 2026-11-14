"""OCR bằng Google Cloud Vision API (DOCUMENT_TEXT_DETECTION).

Gọi REST bằng `requests` với API key, không cần thư viện của Google.
Đánh giá (100 ảnh ViFoodLabel): char acc 0.957, ~0.5 s/ảnh.
"""
import base64

import cv2
import numpy as np
import requests

from .errors import OcrImageError, OcrUnavailableError
from .types import OcrLine, OcrResult

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Loại ngắt sau một ký tự (symbol.property.detectedBreak.type)
SPACE_BREAKS = {"SPACE", "SURE_SPACE"}
LINE_BREAKS = {"EOL_SURE_SPACE", "LINE_BREAK", "HYPHEN"}

# HTTP status cho biết dịch vụ không dùng được (không phải lỗi của riêng ảnh)
UNAVAILABLE_STATUS = {
    400: "API key không hợp lệ hoặc request sai",
    401: "API key không hợp lệ",
    403: "Không có quyền (chưa bật Cloud Vision API / chưa bật Billing / key bị giới hạn)",
    429: "Vượt quota hoặc gọi quá nhanh",
    500: "Lỗi máy chủ Google",
    503: "Dịch vụ Google tạm thời không khả dụng",
}


class VisionEngine:
    name = "Google Cloud Vision"

    def __init__(self, api_key: str, timeout: float = 60, max_side: int = 3000,
                 language_hints: tuple[str, ...] = ("vi", "en")):
        if not api_key:
            raise OcrUnavailableError("Chưa có GOOGLE_VISION_API_KEY trong file .env")
        self.api_key = api_key
        self.timeout = timeout
        self.max_side = max_side
        self.language_hints = list(language_hints)
        self.session = requests.Session()

    def run(self, img: np.ndarray) -> OcrResult:
        return parse_response(self.annotate(img))

    def annotate(self, img: np.ndarray) -> dict:
        """Gọi API, trả về JSON gốc. Mỗi lần gọi = 1 đơn vị tính tiền."""
        body = {"requests": [{
            "image": {"content": encode_image(img, self.max_side)},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": self.language_hints},
        }]}

        try:
            resp = self.session.post(ENDPOINT, params={"key": self.api_key}, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise OcrUnavailableError(f"Không kết nối được Google Cloud Vision: {e}") from e

        if resp.status_code != 200:
            reason = UNAVAILABLE_STATUS.get(resp.status_code, "Lỗi không xác định")
            raise OcrUnavailableError(f"Google Cloud Vision trả về HTTP {resp.status_code}: {reason}. "
                                      f"Chi tiết: {resp.text[:300]}")

        result = resp.json()["responses"][0]
        if "error" in result:  # lỗi của riêng ảnh này (ảnh hỏng, quá lớn...)
            raise OcrImageError(result["error"].get("message", str(result["error"])))
        return result


def encode_image(img: np.ndarray, max_side: int) -> str:
    """Thu nhỏ ảnh quá lớn (giới hạn request ~10 MB), mã hoá JPEG rồi base64."""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OcrImageError("Không mã hoá được ảnh")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parse_response(result: dict) -> OcrResult:
    """Chuyển fullTextAnnotation thành OcrResult: mỗi dòng chữ là một OcrLine.

    Giữ thứ tự đọc của Google (block -> paragraph -> word). Box của dòng là hình chữ nhật bao
    các từ; độ tin cậy của dòng là trung bình confidence các từ.
    """
    lines: list[OcrLine] = []
    pages = (result.get("fullTextAnnotation") or {}).get("pages", [])
    for page in pages:
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                lines.extend(_paragraph_lines(paragraph, start_id=len(lines)))
    return OcrResult(lines=lines)


def _paragraph_lines(paragraph: dict, start_id: int) -> list[OcrLine]:
    lines = []
    text, points, confidences = "", [], []

    for word in paragraph.get("words", []):
        for v in word["boundingBox"].get("vertices", []):
            points.append((v.get("x", 0), v.get("y", 0)))
        confidences.append(word.get("confidence", 0.0))

        for symbol in word.get("symbols", []):
            text += symbol.get("text", "")
            brk = ((symbol.get("property") or {}).get("detectedBreak") or {}).get("type")
            if brk in SPACE_BREAKS:
                text += " "
            elif brk == "HYPHEN":
                text += "-"
            if brk in LINE_BREAKS:
                if text.strip():
                    lines.append(_make_line(text, points, confidences, start_id + len(lines)))
                text, points, confidences = "", [], []

    if text.strip():
        lines.append(_make_line(text, points, confidences, start_id + len(lines)))
    return lines


def _make_line(text: str, points: list, confidences: list, line_id: int) -> OcrLine:
    xs = [p[0] for p in points] or [0]
    ys = [p[1] for p in points] or [0]
    box = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
    score = float(np.mean(confidences)) if confidences else 0.0
    return OcrLine(text=text.strip(), score=score, box=box, line_id=line_id)
