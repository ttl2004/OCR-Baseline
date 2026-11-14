"""Tài nguyên dùng chung, tạo một lần và giữ lại giữa các lần vẽ lại trang (st.cache_resource)."""
import streamlit as st

from ocr_baseline.config import get_settings
from ocr_baseline.ocr import create_engine
from ocr_baseline.prompts import builtin_prompts
from ocr_baseline.storage import Repository


@st.cache_resource(show_spinner="Đang khởi tạo OCR (OCR local lần đầu sẽ tải model, mất vài chục giây)...")
def get_engine(method: str):
    """Engine OCR theo cách thực hiện. Lỗi (vd thiếu key Vision) không được cache, lần sau sẽ thử lại."""
    return create_engine(method, get_settings())


@st.cache_resource
def get_repo() -> Repository:
    repo = Repository(get_settings().db_path)
    repo.ensure_builtin_prompts(builtin_prompts())  # nạp prompt gốc từ code vào bảng prompts
    return repo
