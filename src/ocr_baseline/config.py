"""Cấu hình đọc từ file .env.

Mọi nơi trong code lấy cấu hình qua `get_settings()`, không đọc os.environ trực tiếp.
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]


def _default_threads() -> int:
    # Dùng hết số luồng logic thường CHẬM hơn (hyper-threading, nhân E).
    # Mặc định: một nửa số luồng, tối đa 4.
    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count // 2))


def _to_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # LLM (DeepSeek, API tương thích OpenAI)
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    llm_timeout: float

    # Google Cloud Vision
    google_vision_api_key: str
    vision_timeout: float

    # Đường dẫn
    db_path: Path
    uploads_dir: Path      # nơi lưu bản sao ảnh tải lên (chế độ 1) để xem lại khi chỉnh sửa
    models_dir: Path       # model VietOCR

    # OCR local
    ocr_threads: int
    ocr_enable_mkldnn: bool
    ocr_det_limit_side: int

    # Mặc định trên giao diện
    default_method: str    # vision | local
    default_profile: str   # food | drug


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    env = os.getenv
    data_dir = Path(env("DATA_DIR", str(ROOT_DIR / "data")))
    return Settings(
        deepseek_api_key=env("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=env("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        deepseek_model=env("DEEPSEEK_MODEL", "deepseek-chat").strip(),
        llm_timeout=float(env("LLM_TIMEOUT", "180")),
        google_vision_api_key=env("GOOGLE_VISION_API_KEY", "").strip(),
        vision_timeout=float(env("VISION_TIMEOUT", "60")),
        db_path=Path(env("DB_PATH", str(data_dir / "baseline.db"))),
        uploads_dir=Path(env("UPLOADS_DIR", str(data_dir / "uploads"))),
        models_dir=Path(env("MODELS_DIR", str(ROOT_DIR / "models"))),
        ocr_threads=int(env("OCR_THREADS", "0")) or _default_threads(),
        ocr_enable_mkldnn=_to_bool(env("OCR_ENABLE_MKLDNN", "true")),
        ocr_det_limit_side=int(env("OCR_DET_LIMIT_SIDE", "1600")),
        default_method=env("DEFAULT_METHOD", "vision").strip(),
        default_profile=env("DEFAULT_PROFILE", "food").strip(),
    )
