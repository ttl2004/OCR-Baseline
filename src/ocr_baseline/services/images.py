"""Đọc ảnh, liệt kê ảnh trong thư mục, lưu bản sao ảnh tải lên."""
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..constants import IMAGE_EXTENSIONS


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được file ảnh")
    return img


def read_image(path: Path) -> np.ndarray:
    # np.fromfile + imdecode thay cho cv2.imread để đọc được đường dẫn có dấu tiếng Việt
    return decode_image(np.fromfile(str(path), dtype=np.uint8).tobytes())


def list_images(folder: Path, recursive: bool = False) -> list[Path]:
    """Các file ảnh trong thư mục, sắp xếp theo tên."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Không tìm thấy thư mục: {folder}")
    pattern = "**/*" if recursive else "*"
    images = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def save_uploads(files: list[tuple[str, bytes]], uploads_dir: Path) -> list[str]:
    """Lưu bản sao ảnh tải lên vào uploads_dir/<thời gian>_<mã ngẫu nhiên>/ để xem lại sau."""
    folder = Path(uploads_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (name, data) in enumerate(files, start=1):
        path = folder / f"{index:02d}_{Path(name).name}"
        path.write_bytes(data)
        paths.append(str(path))
    return paths
