"""Xem & chỉnh sửa dữ liệu đã lưu (cả 2 chế độ).

Lọc theo chế độ / dataset / trạng thái / loại tài liệu / prompt -> chọn một bản ghi -> sửa các cột hoặc
text OCR -> Lưu (Đã duyệt). Có thể chạy lại LLM (chọn prompt) hoặc chạy lại cả OCR + LLM.
"""
import pandas as pd
import streamlit as st

from ocr_baseline.config import Settings
from ocr_baseline.constants import (MODE_DATASET, MODE_LABELS, STATUS_AUTO, STATUS_LABELS,
                                    STATUS_REVIEWED)
from ocr_baseline.extraction import LlmError, unverified_numbers
from ocr_baseline.ocr import OcrImageError, OcrUnavailableError
from ocr_baseline.pipeline import extract, join_pages, ocr_image
from ocr_baseline.profiles import PROFILES, get_profile
from ocr_baseline.prompts import Prompt
from ocr_baseline.services.images import read_image

from . import components, state
from .resources import get_engine, get_repo

ALL = "(tất cả)"


def render(settings: Settings, method: str) -> None:
    st.header("Xem & chỉnh sửa dữ liệu")
    repo = get_repo()

    records = repo.list_records(**_filters(repo))
    st.caption(f"{len(records)} bản ghi")
    if not records:
        return

    selected_id = _records_table(records)
    _export_button(records)

    if selected_id is None:
        st.info("Chọn một dòng trong bảng để xem và chỉnh sửa.")
        return
    _record_detail(selected_id, settings, method)


def _prompt_text(name, version) -> str:
    return f"{name} · v{int(version)}" if isinstance(name, str) and name else ""


# ------------------------------------------------------------------ lọc + bảng
def _filters(repo) -> dict:
    datasets = [d["name"] for d in repo.list_datasets()]
    preset = st.session_state.pop(state.RECORDS_DATASET_FILTER, None)
    if preset in datasets:  # được mở từ trang chế độ 2
        st.session_state["rec_mode"] = MODE_DATASET
        st.session_state["rec_dataset"] = preset

    col1, col2, col3, col4, col5 = st.columns(5)
    mode = col1.selectbox("Chế độ", [ALL] + list(MODE_LABELS),
                          format_func=lambda m: MODE_LABELS.get(m, m), key="rec_mode")
    dataset = ALL
    if mode == MODE_DATASET:
        dataset = col2.selectbox("Dataset", [ALL] + datasets, key="rec_dataset")
    status = col3.selectbox("Trạng thái", [ALL] + list(STATUS_LABELS),
                            format_func=lambda s: STATUS_LABELS.get(s, s), key="rec_status")
    profile = col4.selectbox("Loại tài liệu", [ALL] + list(PROFILES),
                             format_func=lambda p: PROFILES[p].label if p in PROFILES else p, key="rec_profile")

    prompts = {p.id: p for p in repo.list_prompts(include_archived=True)
               if profile == ALL or p.profile == profile}
    prompt_id = col5.selectbox("Prompt", [ALL] + list(prompts),
                               format_func=lambda i: prompts[i].label if i in prompts else i, key="rec_prompt")

    def value(v):
        return None if v == ALL else v

    return {"mode": value(mode), "dataset": value(dataset), "status": value(status),
            "profile": value(profile), "prompt_id": value(prompt_id)}


def _records_table(records: list[dict]) -> int | None:
    df = pd.DataFrame(records)
    view = pd.DataFrame({
        "Mã": df["code"],
        "Chế độ": df["mode"],
        "Dataset": df["dataset"].fillna(""),
        "Ảnh": df["source_key"].fillna(""),
        "Tên": df["title"].fillna(""),
        "Trạng thái": df["status"].map(STATUS_LABELS),
        "Prompt": [_prompt_text(n, v) for n, v in zip(df["prompt_name"], df["prompt_version"])],
        "Cách OCR": df["ocr_method"],
        "Loại": df["profile"],
        "Cập nhật": df["updated_at"],
    })
    event = st.dataframe(view, hide_index=True, width="stretch", height=300,
                         on_select="rerun", selection_mode="single-row", key="rec_table")
    rows = event.selection.rows
    if not rows:
        return None
    return int(df.iloc[rows[0]]["id"])


def _export_button(records: list[dict]) -> None:
    prompt_labels = {r["id"]: _prompt_text(r["prompt_name"], r["prompt_version"]) for r in records}
    rows = []
    for r in get_repo().export_records([r["id"] for r in records]):
        row = {"Mã": r["code"], "Chế độ": r["mode"], "Dataset": r["dataset"] or "", "Ảnh": r["source_key"] or "",
               "Loại": r["profile"], "Cách OCR": r["ocr_method"], "Prompt": prompt_labels.get(r["id"], ""),
               "Prompt ID": r["prompt_id"], "Model LLM": r["llm_model"] or "",
               "Trạng thái": STATUS_LABELS[r["status"]]}
        if r["profile"] in PROFILES:
            row.update(components.fields_to_row(PROFILES[r["profile"]], r["fields"]))
        rows.append(row)
    csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Xuất CSV các bản ghi đang lọc", csv, file_name="records.csv", mime="text/csv")


# ------------------------------------------------------------------ chi tiết một bản ghi
def _record_detail(record_id: int, settings: Settings, method: str) -> None:
    repo = get_repo()
    record = repo.get_record(record_id)
    if record is None:
        st.warning("Bản ghi không còn tồn tại.")
        return
    profile = get_profile(record["profile"])
    used_prompt = repo.get_prompt(record["prompt_id"]) if record["prompt_id"] else None

    version_key = f"rec_version_{record_id}"   # tăng khi dữ liệu đổi -> làm mới các ô chỉnh sửa
    version = st.session_state.get(version_key, 0)
    widget_key = f"rec_{record_id}_{version}"

    st.divider()
    where = f" - dataset `{record['dataset']}` / `{record['source_key']}`" if record["dataset"] else ""
    st.subheader(f"{record['code']}{where}")
    st.caption(f"{MODE_LABELS[record['mode']]} · {profile.label} · OCR: {record['ocr_method']} · "
               f"Prompt: **{used_prompt.label if used_prompt else 'chưa chạy LLM'}** · "
               f"Model: {record['llm_model'] or '-'} · Trạng thái: **{STATUS_LABELS[record['status']]}** · "
               f"OCR {record['ocr_seconds'] or '-'}s, LLM {record['llm_seconds'] or '-'}s")
    if record["error_message"]:
        st.error(record["error_message"])

    left, right = st.columns([2, 3])
    with left:
        components.image_gallery(record["image_paths"])
    with right:
        tab_fields, tab_text, tab_prompt = st.tabs(["Thông tin", "Text OCR", "Prompt đã dùng"])
        with tab_text:
            ocr_text = st.text_area("Text OCR (sửa xong có thể bấm 'Chạy lại LLM')", record["ocr_text"] or "",
                                    height=500, key=f"{widget_key}_ocr")
        with tab_fields:
            edited = components.fields_editor(profile, record["fields"], key=widget_key)
        with tab_prompt:
            if used_prompt:
                st.caption(f"{used_prompt.label} (id {used_prompt.id}), tạo lúc {used_prompt.created_at}")
                st.code(used_prompt.content, language=None, wrap_lines=True)
            else:
                st.info("Bản ghi chưa được LLM xử lý.")
        warnings = unverified_numbers(edited, ocr_text)
        components.warnings_box(profile, warnings)

    # Prompt dùng khi chạy lại LLM: mặc định prompt của dataset (chế độ 2) hoặc prompt bản ghi đã dùng
    default_prompt_id = record["prompt_id"]
    if record["dataset"]:
        dataset = repo.get_dataset(record["dataset"])
        default_prompt_id = (dataset or {}).get("prompt_id") or default_prompt_id
    rerun_prompt = components.prompt_selector(repo, profile, key=f"rec_prompt_{record_id}",
                                              default_id=default_prompt_id, label="Prompt khi chạy lại LLM")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("✅ Lưu (Đã duyệt)", type="primary", key=f"{widget_key}_save"):
        repo.update_record(record_id, fields=edited, ocr_text=ocr_text, warnings=warnings,
                           title=profile.title_of(edited), status=STATUS_REVIEWED, error_message=None)
        st.session_state[version_key] = version + 1
        st.toast(f"Đã lưu {record['code']}")
        st.rerun()

    if rerun_prompt and col2.button("Chạy lại LLM", key=f"{widget_key}_llm",
                                    help="Dùng text OCR hiện tại (kể cả phần đã sửa) và prompt đã chọn ở trên"):
        _rerun_llm(record_id, ocr_text, profile, rerun_prompt, settings, version_key)

    if rerun_prompt and col3.button("Chạy lại OCR + LLM", key=f"{widget_key}_all",
                                    help=f"Dùng cách thực hiện đang chọn ({method}) và prompt đã chọn ở trên"):
        _rerun_all(record, profile, rerun_prompt, settings, method, version_key)

    with col4.popover("🗑️ Xoá"):
        st.write(f"Xoá bản ghi {record['code']}?")
        if st.button("Xác nhận xoá", key=f"{widget_key}_delete"):
            repo.delete_record(record_id)
            st.rerun()


def _rerun_llm(record_id: int, ocr_text: str, profile, prompt: Prompt, settings: Settings,
               version_key: str) -> None:
    try:
        with st.spinner("LLM đang xử lý..."):
            result = extract(ocr_text, profile, prompt, settings)
    except LlmError as e:
        st.error(f"Lỗi LLM: {e}")
        return
    get_repo().update_record(record_id, ocr_text=ocr_text, fields=result.fields, warnings=result.warnings,
                             title=profile.title_of(result.fields), llm_seconds=result.seconds,
                             prompt_id=result.prompt_id, llm_model=result.llm_model,
                             status=STATUS_AUTO, error_message=None)
    st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
    st.rerun()


def _rerun_all(record: dict, profile, prompt: Prompt, settings: Settings, method: str, version_key: str) -> None:
    try:
        engine = get_engine(method)
        with st.spinner("Đang OCR..."):
            pages = [ocr_image(engine, read_image(p), p) for p in record["image_paths"]]
    except OcrUnavailableError as e:
        components.show_ocr_unavailable(e, method)
        return
    except (OcrImageError, ValueError) as e:
        st.error(f"Không OCR được ảnh: {e}")
        return

    ocr_text = join_pages(pages)
    get_repo().update_record(record["id"], ocr_method=method, ocr_text=ocr_text,
                             ocr_seconds=round(sum(p.seconds for p in pages), 2))
    _rerun_llm(record["id"], ocr_text, profile, prompt, settings, version_key)
