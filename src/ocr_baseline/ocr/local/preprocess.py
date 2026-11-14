"""Tiền xử lý ảnh trước khi OCR."""
import threading
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@dataclass
class PreprocessOptions:
    auto_rotate: bool = True     # tự xoay 0/90/180/270 độ theo hướng chữ (model PP-LCNet doc_ori)
    # Chỉ xoay khi model chắc chắn. Đánh giá ViFoodLabel: mọi lần xoay với score < 0.9 đều SAI
    # (ảnh đứng thẳng bị xoay ngang -> OCR ra rác), các lần score >= 0.92 đều đúng.
    rotate_min_score: float = 0.9
    resize: bool = True          # đưa cạnh dài về khoảng [min_long, max_long]
    min_long: int = 1600
    max_long: int = 3000
    deskew: bool = True          # xoay thẳng ảnh bị nghiêng nhẹ
    denoise: bool = False        # khử nhiễu (chậm hơn, dùng cho ảnh chụp tối/nhiễu)
    # CLAHE tăng tương phản. Đánh giá ViFoodLabel (600 ảnh): BẬT làm giảm 1.5 điểm độ chính xác,
    # nên mặc định tắt; chỉ bật thủ công cho ảnh chụp quá tối.
    enhance_contrast: bool = False


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được file ảnh")
    return img


# nhãn của model = góc ảnh đang bị xoay -> phép xoay để đưa về 0
_UNROTATE = {"90": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180, "270": cv2.ROTATE_90_CLOCKWISE}


_orientation_lock = threading.Lock()


def _mkldnn_enabled() -> bool:
    from ...config import get_settings

    return get_settings().ocr_enable_mkldnn


@lru_cache(maxsize=1)
def _orientation_model(threads: int = 2, enable_mkldnn: bool = True):
    from paddleocr import DocImgOrientationClassification

    return DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori", device="cpu",
                                           cpu_threads=threads, enable_mkldnn=enable_mkldnn)


def _auto_rotate(img: np.ndarray, min_score: float = 0.9) -> tuple[np.ndarray, str]:
    with _orientation_lock:
        res = next(iter(_orientation_model(enable_mkldnn=_mkldnn_enabled()).predict(img)))
    label, score = str(res["label_names"][0]), float(res["scores"][0])
    if label in _UNROTATE and score >= min_score:
        return cv2.rotate(img, _UNROTATE[label]), label
    return img, "0"


def _resize(img: np.ndarray, min_long: int, max_long: int) -> np.ndarray:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side < min_long:
        scale = min_long / long_side
        interp = cv2.INTER_CUBIC
    elif long_side > max_long:
        scale = max_long / long_side
        interp = cv2.INTER_AREA
    else:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)


def _estimate_skew(gray: np.ndarray) -> float:
    """Ước lượng góc nghiêng (độ) dựa trên các dòng chữ."""
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # nối các ký tự trong cùng dòng thành vệt ngang
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 60), 3))
    lines = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles, weights = [], []
    min_width = gray.shape[1] * 0.15
    for c in contours:
        (_, _), (w, h), angle = cv2.minAreaRect(c)
        if w < h:
            w, h = h, w
            angle = angle - 90
        if w < min_width or h == 0 or w / h < 5:
            continue
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        angles.append(angle)
        weights.append(w)
    if not angles:
        return 0.0
    return float(np.average(angles, weights=weights))


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def preprocess(img: np.ndarray, opts: PreprocessOptions | None = None) -> tuple[np.ndarray, dict]:
    """Trả về (ảnh BGR đã xử lý, thông tin các bước đã áp dụng)."""
    opts = opts or PreprocessOptions()
    info: dict = {"original_size": img.shape[1::-1]}

    if opts.auto_rotate:
        img, label = _auto_rotate(img, opts.rotate_min_score)
        if label != "0":
            info["rotated_from"] = int(label)

    if opts.resize:
        img = _resize(img, opts.min_long, opts.max_long)

    if opts.denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    if opts.enhance_contrast:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if opts.deskew:
        angle = _estimate_skew(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        # chỉ xoay khi nghiêng rõ nhưng không quá lớn (góc lớn để PaddleOCR tự xử lý)
        if 0.5 <= abs(angle) <= 15:
            img = _rotate(img, angle)
            info["deskew_angle"] = round(angle, 2)

    info["final_size"] = img.shape[1::-1]
    return img, info
