"""Thanh bên: chọn trang (chế độ), cách thực hiện, loại tài liệu + trạng thái cấu hình."""
import streamlit as st

from ocr_baseline.config import Settings
from ocr_baseline.constants import METHOD_LABELS, METHOD_VISION
from ocr_baseline.profiles import PROFILES

from . import state


def render_sidebar(settings: Settings) -> str:
    """Vẽ thanh bên, trả về trang đang chọn."""
    with st.sidebar:
        st.title("OCR Baseline")

        page = st.radio("Chế độ", list(state.PAGE_LABELS), format_func=state.PAGE_LABELS.get, key=state.PAGE)

        st.divider()
        st.radio("Cách thực hiện", list(METHOD_LABELS), format_func=METHOD_LABELS.get, key=state.METHOD)
        st.selectbox("Loại tài liệu", list(PROFILES), format_func=lambda k: PROFILES[k].label,
                     key=state.PROFILE)

        st.divider()
        st.caption("**Cấu hình**")
        st.caption(_key_status("Google Vision API key", settings.google_vision_api_key))
        st.caption(_key_status("DeepSeek API key (LLM)", settings.deepseek_api_key))
        st.caption(f"LLM: `{settings.deepseek_model}`")
        st.caption(f"OCR local: {settings.ocr_threads} luồng, detection {settings.ocr_det_limit_side}px")
        st.caption(f"DB: `{settings.db_path}`")

        if st.session_state[state.METHOD] == METHOD_VISION and not settings.google_vision_api_key:
            st.warning("Chưa có GOOGLE_VISION_API_KEY trong .env. Hãy thêm key hoặc chọn OCR local.")
    return page


def _key_status(name: str, value: str) -> str:
    return f"{name}: {'✅ đã có' if value else '❌ chưa có'}"
