"""OCR local chạy CPU: tiền xử lý -> PaddleOCR (phát hiện vùng chữ) -> VietOCR (nhận dạng tiếng Việt).

Cấu hình v3 đã đánh giá (100 ảnh ViFoodLabel): char acc 0.826, ~8.7 s/ảnh trên i5-12450H 4 luồng.
Model nhận dạng của PaddleOCR thiếu ký tự tiếng Việt (ạ ả ố ự...) nên phần nhận dạng dùng VietOCR.
"""
import re
import threading
from pathlib import Path

import numpy as np

from ..errors import OcrImageError
from ..types import OcrLine, OcrResult
from .geometry import crop_line
from .layout import sort_reading_order
from .preprocess import PreprocessOptions, preprocess
from .vietocr_rec import VietOcrRecognizer


class LocalOcrEngine:
    name = "OCR local (PaddleOCR + VietOCR)"

    def __init__(self, models_dir: Path, threads: int = 4, enable_mkldnn: bool = True,
                 det_limit_side_len: int = 1600, det_model: str = "PP-OCRv5_server_det",
                 rec_model: str = "vgg_transformer", crop_pad: float = 0.04,
                 options: PreprocessOptions | None = None):
        import torch
        from paddleocr import TextDetection

        torch.set_num_threads(threads)
        # Đánh giá: detection 1600px + crop_pad 0.04 tốt hơn mặc định của Paddle (960, 0.15) 4.6 điểm.
        # Box rộng hơn (crop_pad 0.25) làm GIẢM độ chính xác: VietOCR cần crop sát chữ.
        self.detector = TextDetection(model_name=det_model, device="cpu", cpu_threads=threads,
                                      enable_mkldnn=enable_mkldnn, limit_side_len=det_limit_side_len,
                                      limit_type="max")
        self.recognizer = VietOcrRecognizer(rec_model, models_dir)
        self.crop_pad = crop_pad
        self.options = options or PreprocessOptions()
        # Model không an toàn khi nhiều luồng gọi cùng lúc -> các lần gọi xếp hàng.
        self.lock = threading.Lock()

    def run(self, img: np.ndarray) -> OcrResult:
        if img is None or img.size == 0:
            raise OcrImageError("Ảnh rỗng hoặc không đọc được")

        with self.lock:
            img, _ = preprocess(img, self.options)
            polys = [np.asarray(p, dtype=np.float32)
                     for res in self.detector.predict(img) for p in res["dt_polys"]]
            if not polys:
                return OcrResult()

            # Mỗi dòng hiển thị có thể gồm nhiều box (vd: "Metronidazol ....... 125mg")
            visual_lines = sort_reading_order(polys)
            ordered = []
            for line_id, line in enumerate(visual_lines):
                for poly in line:
                    ordered.append((line_id, poly))
            crops = [crop_line(img, poly, self.crop_pad) for _, poly in ordered]
            predictions = self.recognizer.recognize(crops)

        lines = []
        for (line_id, poly), (text, prob) in zip(ordered, predictions):
            text = fix_text(text)
            if text:
                lines.append(OcrLine(text=text, score=prob, box=poly.tolist(), line_id=line_id))
        return OcrResult(lines=lines)


def fix_text(text: str) -> str:
    text = str(text).strip()
    # VietOCR không có ký tự "°" -> nhận thành "?" (vd: 30?C)
    return re.sub(r"(\d)\s?\?\s?([CF])\b", r"\1°\2", text)
