"""Chế độ 1: người dùng tải ảnh lên (nhiều ảnh = nhiều mặt của CÙNG một sản phẩm).

Luồng: chọn ảnh -> 1. OCR -> (sửa text) -> 2. LLM tách cột -> review, sửa -> 3. Lưu (mã UP-xxxxxx).
"""
import streamlit as st

from ocr_baseline.config import Settings
from ocr_baseline.constants import MODE_UPLOAD, STATUS_REVIEWED
from ocr_baseline.extraction import LlmError, unverified_numbers
from ocr_baseline.ocr import OcrImageError, OcrUnavailableError
from ocr_baseline.pipeline import extract, join_pages, ocr_image
from ocr_baseline.profiles import Profile
from ocr_baseline.prompts import Prompt
from ocr_baseline.services.images import decode_image, save_uploads

from . import components
from .resources import get_engine, get_repo

# Key session_state của trang này (tiền tố "up_")
FILES = "up_files"            # [(tên file, bytes)]
PAGES = "up_pages"            # [PageOcr]
TEXT = "up_text"              # text gửi LLM (người dùng có thể sửa)
METHOD_USED = "up_method_used"
EXTRACTION = "up_extraction"  # Extraction
PROFILE_USED = "up_profile_used"
SAVED = "up_saved"            # thông báo đã lưu
VERSION = "up_version"        # đổi khi có kết quả mới -> làm mới các ô chỉnh sửa


def _reset() -> None:
    for key in (PAGES, TEXT, METHOD_USED, EXTRACTION, PROFILE_USED, SAVED):
        st.session_state.pop(key, None)


def render(settings: Settings, method: str, profile: Profile) -> None:
    st.header("Chế độ 1 - Tải ảnh lên")
    st.caption("Chọn một hoặc nhiều ảnh của **cùng một sản phẩm** (nhiều mặt / nhiều trang, theo đúng thứ tự). "
               "Kết quả lưu với mã `UP-xxxxxx`.")

    uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
                                accept_multiple_files=True, on_change=_reset)
    if not uploaded:
        return

    if st.button("1. OCR", type="primary"):
        _run_ocr(uploaded, method)

    if PAGES not in st.session_state:
        return

    _show_ocr_result()

    st.session_state[TEXT] = st.text_area("Text gửi cho LLM (có thể sửa tay trước khi gửi)",
                                          st.session_state[TEXT], height=250)

    prompt = components.prompt_selector(get_repo(), profile, key="up_prompt")
    if prompt and st.button("2. Gửi LLM để tách cột", type="primary"):
        _run_llm(settings, profile, prompt)

    if EXTRACTION in st.session_state:
        _show_review(settings)


def _run_ocr(uploaded, method: str) -> None:
    _reset()
    files = [(f.name, f.getvalue()) for f in uploaded]
    try:
        engine = get_engine(method)
        pages = []
        progress = st.progress(0.0)
        for index, (name, data) in enumerate(files):
            with st.spinner(f"OCR {name} ({index + 1}/{len(files)})..."):
                pages.append(ocr_image(engine, decode_image(data), name))
            progress.progress((index + 1) / len(files))
    except OcrUnavailableError as e:
        components.show_ocr_unavailable(e, method)
        return
    except (OcrImageError, ValueError) as e:
        st.error(f"Không OCR được ảnh: {e}")
        return

    st.session_state[FILES] = files
    st.session_state[PAGES] = pages
    st.session_state[TEXT] = join_pages(pages)
    st.session_state[METHOD_USED] = method


def _show_ocr_result() -> None:
    pages = st.session_state[PAGES]
    files = dict(st.session_state[FILES])
    st.subheader(f"Kết quả OCR ({st.session_state[METHOD_USED]})")
    for page in pages:
        title = f"{page.name} - {page.n_lines} dòng, độ tin cậy TB {page.mean_score}, {page.seconds}s"
        with st.expander(title, expanded=len(pages) == 1):
            left, right = st.columns(2)
            left.image(files[page.name], width="stretch")
            right.text_area("Text OCR", page.text, height=450, disabled=True, key=f"up_raw_{page.name}")
            if page.low_confidence:
                right.caption(f"⚠️ {len(page.low_confidence)} dòng có độ tin cậy < 0.8:")
                right.dataframe([{"text": ln.text, "score": round(ln.score, 2)} for ln in page.low_confidence],
                                hide_index=True, width="stretch")


def _run_llm(settings: Settings, profile: Profile, prompt: Prompt) -> None:
    try:
        with st.spinner("LLM đang xử lý..."):
            st.session_state[EXTRACTION] = extract(st.session_state[TEXT], profile, prompt, settings)
        st.session_state[PROFILE_USED] = profile
        st.session_state[VERSION] = st.session_state.get(VERSION, 0) + 1
        st.session_state.pop(SAVED, None)
    except LlmError as e:  # gồm cả LlmUnavailableError
        st.error(f"Lỗi LLM: {e}")


def _show_review(settings: Settings) -> None:
    extraction = st.session_state[EXTRACTION]
    profile = st.session_state[PROFILE_USED]
    prompt = get_repo().get_prompt(extraction.prompt_id)
    st.subheader(f"Review kết quả - {profile.label} ({extraction.seconds}s)")
    st.caption(f"Prompt đã dùng: **{prompt.label if prompt else '-'}** · model `{extraction.llm_model}`")
    st.caption("Sửa trực tiếp các ô bên dưới rồi bấm Lưu.")

    edited = components.fields_editor(profile, extraction.fields, key=f"up_edit_{st.session_state[VERSION]}")
    warnings = unverified_numbers(edited, st.session_state[TEXT])
    components.warnings_box(profile, warnings)

    if st.button("3. Duyệt & lưu vào DB", type="primary"):
        if not profile.title_of(edited):
            st.error(f"Cần có ít nhất '{_title_label(profile)}' trước khi lưu.")
        else:
            image_paths = save_uploads(st.session_state[FILES], settings.uploads_dir)
            _, code = get_repo().add_record(
                mode=MODE_UPLOAD, image_paths=image_paths, profile=profile.key,
                ocr_method=st.session_state[METHOD_USED], ocr_text=st.session_state[TEXT], fields=edited,
                warnings=warnings, status=STATUS_REVIEWED, title=profile.title_of(edited),
                prompt_id=extraction.prompt_id, llm_model=extraction.llm_model,
                ocr_seconds=round(sum(p.seconds for p in st.session_state[PAGES]), 2),
                llm_seconds=extraction.seconds,
            )
            st.session_state[SAVED] = f"Đã lưu bản ghi {code}"

    if SAVED in st.session_state:
        st.success(st.session_state[SAVED])


def _title_label(profile: Profile) -> str:
    for field in profile.fields:
        if field.key == profile.title_field:
            return field.label
    return profile.title_field
