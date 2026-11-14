"""Giao diện OCR Baseline.

Chạy:  streamlit run app.py
File này chỉ chọn trang; nội dung từng trang nằm trong thư mục ui/.
"""
import streamlit as st

from ocr_baseline.config import get_settings
from ocr_baseline.profiles import get_profile
from ui import page_dataset, page_prompts, page_records, page_upload, state
from ui.sidebar import render_sidebar

st.set_page_config(page_title="OCR Baseline", layout="wide")

settings = get_settings()
state.init_state(settings.default_method, settings.default_profile)
page = render_sidebar(settings)

method = st.session_state[state.METHOD]
profile = get_profile(st.session_state[state.PROFILE])

if page == state.PAGE_UPLOAD:
    page_upload.render(settings, method, profile)
elif page == state.PAGE_DATASET:
    page_dataset.render(settings, method, profile)
elif page == state.PAGE_RECORDS:
    page_records.render(settings, method)
else:
    page_prompts.render(settings, profile)
