"""Hàm hình học cho box chữ."""
import cv2
import numpy as np

Box = np.ndarray  # polygon (N, 2)


def bbox(p: Box) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) của polygon."""
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def crop_line(img: np.ndarray, poly: Box, pad_ratio: float = 0.04) -> np.ndarray:
    """Cắt và nắn thẳng một dòng chữ theo polygon."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(poly.astype(np.float32))
    if w < h:  # đưa về dạng dòng nằm ngang
        w, h = h, w
        angle += 90
    pad = h * pad_ratio
    w, h = w + 2 * pad, h + 2 * pad
    src = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    # boxPoints trả về theo thứ tự bất kỳ -> sắp lại: trên-trái, trên-phải, dưới-phải, dưới-trái
    s, d = src.sum(1), np.diff(src, axis=1).ravel()
    ordered = np.array([src[s.argmin()], src[d.argmin()], src[s.argmax()], src[d.argmax()]], np.float32)
    W, H = max(int(round(w)), 1), max(int(round(h)), 1)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    m = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
