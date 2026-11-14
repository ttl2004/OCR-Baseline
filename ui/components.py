"""Các thành phần giao diện dùng lại ở nhiều trang."""
from pathlib import Path

import pandas as pd
import streamlit as st

from ocr_baseline.constants import METHOD_VISION
from ocr_baseline.profiles import KIND_LIST, KIND_NUTRITION, KIND_TEXT, Profile
from ocr_baseline.prompts import Prompt

from . import state


# ------------------------------------------------------------------ thông báo lỗi OCR
def show_ocr_unavailable(error: Exception, method: str, continue_batch: bool = False) -> None:
    """Báo dịch vụ OCR không dùng được. Với Google Vision: gợi ý chuyển sang OCR local."""
    if method != METHOD_VISION:
        st.error(f"OCR local lỗi: {error}")
        return

    st.error(f"Không gọi được Google Cloud Vision.\n\n{error}")
    st.info("Bạn có thể chuyển sang **OCR local** (chạy trên máy, không cần mạng; chậm hơn và độ chính xác "
            "thấp hơn) để tiếp tục.")
    if continue_batch:
        st.button("Chuyển sang OCR local và chạy tiếp", type="primary",
                  on_click=state.switch_to_local_and_continue)
    else:
        st.button("Chuyển sang OCR local", type="primary", on_click=state.switch_to_local)


# ------------------------------------------------------------------ chọn prompt
def prompt_selector(repo, profile: Profile, key: str, default_id: int | None = None,
                    label: str = "Prompt", context: str = "") -> Prompt | None:
    """Ô chọn prompt của loại tài liệu `profile`. Trả về prompt đã chọn.

    default_id: prompt chọn sẵn (vd prompt của dataset). Lựa chọn chỉ được đặt lại về default_id khi
    `context` đổi (vd người dùng nhập tên dataset khác) hoặc lựa chọn hiện tại không còn hợp lệ,
    để không ghi đè lựa chọn của người dùng.
    """
    prompts = repo.list_prompts(profile=profile.key)
    if not prompts:
        st.error(f"Chưa có prompt nào cho '{profile.label}'. Tạo ở trang Quản lý prompt.")
        return None

    ids = [p.id for p in prompts]
    by_id = {p.id: p for p in prompts}
    if default_id not in by_id:
        default_id = ids[0]
    context = f"{profile.key}|{context}"
    if st.session_state.get(f"{key}_context") != context or st.session_state.get(key) not in by_id:
        st.session_state[f"{key}_context"] = context
        st.session_state[key] = default_id

    def fmt(prompt_id: int) -> str:
        p = by_id[prompt_id]
        return p.label + (" (gốc)" if p.is_builtin else "")

    selected = by_id[st.selectbox(label, ids, format_func=fmt, key=key)]
    with st.expander(f"Xem nội dung prompt: {selected.label}"):
        if selected.note:
            st.caption(f"Ghi chú: {selected.note}")
        st.code(selected.content, language=None, wrap_lines=True)
    return selected


# ------------------------------------------------------------------ chỉnh sửa các cột
def fields_editor(profile: Profile, fields: dict, key: str) -> dict:
    """Form chỉnh sửa các cột theo profile. `key` phải khác nhau giữa các bản ghi.

    Trả về giá trị sau khi người dùng sửa (cùng cấu trúc với `fields`).
    """
    edited = {}
    for field in profile.fields:
        value = fields.get(field.key)
        widget_key = f"{key}_{field.key}"

        if field.kind == KIND_TEXT:
            text = value or ""
            height = 68 if len(text) < 80 else 150
            edited[field.key] = st.text_area(field.label, text, height=height, key=widget_key).strip()

        elif field.kind == KIND_LIST:
            items = value or []
            text = st.text_area(f"{field.label} (mỗi mục một dòng)", "\n".join(items),
                                height=max(68, min(300, 28 * len(items) + 20)), key=widget_key)
            edited[field.key] = [line.strip() for line in text.split("\n") if line.strip()]

        elif field.kind == KIND_NUTRITION:
            st.markdown(f"**{field.label}**")
            df = pd.DataFrame(value or [], columns=["name", "value"])
            table = st.data_editor(
                df, num_rows="dynamic", width="stretch", key=widget_key,
                column_config={"name": st.column_config.TextColumn("Tên chất"),
                               "value": st.column_config.TextColumn("Giá trị (kèm đơn vị)")},
            )
            rows = []
            for row in table.fillna("").to_dict("records"):
                name, amount = str(row["name"]).strip(), str(row["value"]).strip()
                if name or amount:
                    rows.append({"name": name, "value": amount})
            edited[field.key] = rows

    return edited


def warnings_box(profile: Profile, warnings: dict) -> None:
    """Cảnh báo con số LLM đưa ra mà không có trong text OCR."""
    if not warnings:
        return
    labels = {f.key: f.label for f in profile.fields}
    lines = [f"- **{labels.get(k, k)}**: {', '.join(v)}" for k, v in warnings.items()]
    st.warning("Con số có trong kết quả nhưng KHÔNG có trong text OCR, cần so lại với ảnh:\n\n" + "\n".join(lines))


def image_gallery(paths: list[str]) -> None:
    for path in paths:
        if Path(path).exists():
            st.image(path, caption=Path(path).name, width="stretch")
        else:
            st.warning(f"Không tìm thấy ảnh: {path}")


# ------------------------------------------------------------------ xuất CSV
def fields_to_row(profile: Profile, fields: dict) -> dict:
    """Chuyển các cột thành chuỗi phẳng để xuất CSV."""
    row = {}
    for field in profile.fields:
        value = fields.get(field.key)
        if field.kind == KIND_LIST:
            value = " | ".join(value or [])
        elif field.kind == KIND_NUTRITION:
            value = "; ".join(f"{n.get('name', '')}: {n.get('value', '')}" for n in value or [])
        row[field.label] = value or ""
    return row
