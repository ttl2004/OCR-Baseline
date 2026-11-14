"""Quản lý prompt: xem, tạo phiên bản mới / prompt mới, thử prompt, ẩn, xoá.

Prompt không bị sửa đè: "Lưu" luôn tạo một phiên bản mới (cùng tên -> version + 1, tên mới -> version 1),
nên các bản ghi đã tạo vẫn trỏ đúng nội dung prompt đã dùng.
Prompt gốc (lấy từ code) không sửa được; muốn thay đổi thì lưu với tên khác.
"""
import json

import pandas as pd
import streamlit as st

from ocr_baseline.config import Settings
from ocr_baseline.extraction import LlmError, extract_fields
from ocr_baseline.profiles import Profile
from ocr_baseline.prompts import Prompt, missing_field_keys

from .resources import get_repo


def render(settings: Settings, profile: Profile) -> None:
    st.header("Quản lý prompt")
    st.caption(f"Loại tài liệu: **{profile.label}** (đổi ở thanh bên). Prompt phải yêu cầu LLM trả về JSON có đúng "
               f"các cột: `{', '.join(profile.field_keys())}`.")
    repo = get_repo()

    show_archived = st.checkbox("Hiện cả prompt đã ẩn")
    prompts = repo.list_prompts(profile=profile.key, include_archived=show_archived)
    usage = repo.prompt_usage()

    selected = _prompts_table(prompts, usage)
    if selected is None:
        st.info("Chọn một prompt trong bảng để xem, sửa (tạo phiên bản mới) hoặc thử.")
        return

    st.divider()
    st.subheader(selected.label + (" (gốc)" if selected.is_builtin else ""))
    used = usage.get(selected.id, {"records": 0, "datasets": []})
    st.caption(f"ID {selected.id} · tạo lúc {selected.created_at} · {used['records']} bản ghi đã dùng · "
               f"dataset đang gắn: {', '.join(used['datasets']) or 'không có'}")
    if selected.note:
        st.caption(f"Ghi chú: {selected.note}")

    tab_edit, tab_test, tab_manage = st.tabs(["Sửa / tạo phiên bản mới", "Thử prompt", "Ẩn / xoá"])
    with tab_edit:
        _edit_form(repo, profile, selected)
    with tab_test:
        _test_prompt(repo, profile, selected, settings)
    with tab_manage:
        _manage(repo, selected, used)


def _prompts_table(prompts: list[Prompt], usage: dict) -> Prompt | None:
    if not prompts:
        st.warning("Chưa có prompt nào.")
        return None
    view = pd.DataFrame([{
        "ID": p.id,
        "Tên": p.name,
        "Phiên bản": p.version,
        "Gốc": "✓" if p.is_builtin else "",
        "Ẩn": "✓" if p.archived else "",
        "Số bản ghi": usage.get(p.id, {}).get("records", 0),
        "Dataset đang gắn": ", ".join(usage.get(p.id, {}).get("datasets", [])),
        "Ghi chú": p.note,
        "Tạo lúc": p.created_at,
    } for p in prompts])
    event = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="prompt_table")
    if not event.selection.rows:
        return None
    return prompts[event.selection.rows[0]]


def _edit_form(repo, profile: Profile, selected: Prompt) -> None:
    if selected.is_builtin:
        st.info("Đây là prompt gốc, không sửa được. Lưu với **tên khác** để tạo prompt mới từ prompt này.")
        default_name = f"{selected.name} - bản sửa"
    else:
        st.caption("Giữ nguyên tên -> tạo phiên bản tiếp theo. Đổi tên -> tạo prompt mới.")
        default_name = selected.name

    with st.form(f"prompt_form_{selected.id}"):
        name = st.text_input("Tên prompt", default_name)
        content = st.text_area("Nội dung prompt", selected.content, height=450)
        note = st.text_input("Ghi chú thay đổi", placeholder="vd: thêm quy ước tách thành phần")
        submitted = st.form_submit_button("Lưu thành phiên bản mới", type="primary")

    missing = missing_field_keys(profile, content)
    if missing:
        st.warning(f"Prompt không nhắc tới các cột: `{', '.join(missing)}`. LLM có thể không trả về các cột này.")

    if submitted:
        try:
            saved = repo.save_prompt_version(name, profile.key, content, note)
        except ValueError as e:
            st.error(str(e))
            return
        st.success(f"Đã lưu **{saved.label}** (ID {saved.id}). Chọn prompt này ở chế độ 1 / chế độ 2 để dùng.")


def _test_prompt(repo, profile: Profile, selected: Prompt, settings: Settings) -> None:
    st.caption("Chạy thử prompt trên text OCR của một bản ghi có sẵn hoặc text tự dán. Kết quả KHÔNG được lưu.")
    content = st.text_area("Prompt dùng để thử (có thể sửa tạm)", selected.content, height=250,
                           key=f"test_prompt_{selected.id}")

    records = repo.list_records(profile=profile.key)[:200]
    options = [None] + [r["id"] for r in records]
    labels = {r["id"]: f"{r['code']} - {r['title'] or r['source_key'] or ''}" for r in records}
    record_id = st.selectbox("Lấy text OCR từ bản ghi", options,
                             format_func=lambda i: "(tự dán text)" if i is None else labels[i],
                             key=f"test_record_{selected.id}")
    default_text = repo.get_record(record_id)["ocr_text"] if record_id else ""
    ocr_text = st.text_area("Text OCR", default_text or "", height=200, key=f"test_text_{selected.id}_{record_id}")

    if st.button("Chạy thử", key=f"test_run_{selected.id}"):
        try:
            with st.spinner("LLM đang xử lý..."):
                fields, info = extract_fields(ocr_text, profile, content, settings)
        except LlmError as e:
            st.error(f"Lỗi LLM: {e}")
            return
        st.caption(f"{info['seconds']}s · {info['prompt_tokens']} token vào, {info['completion_tokens']} token ra")
        if record_id:
            old = repo.get_record(record_id)["fields"]
            left, right = st.columns(2)
            left.markdown("**Kết quả đã lưu của bản ghi**")
            left.code(json.dumps(old, ensure_ascii=False, indent=1), language="json", wrap_lines=True)
            right.markdown("**Kết quả với prompt đang thử**")
            right.code(json.dumps(fields, ensure_ascii=False, indent=1), language="json", wrap_lines=True)
        else:
            st.code(json.dumps(fields, ensure_ascii=False, indent=1), language="json", wrap_lines=True)


def _manage(repo, selected: Prompt, used: dict) -> None:
    if selected.archived:
        if st.button("Hiện lại prompt này", key=f"unarchive_{selected.id}"):
            repo.set_prompt_archived(selected.id, False)
            st.rerun()
    elif st.button("Ẩn prompt này (không hiện trong danh sách chọn)", key=f"archive_{selected.id}"):
        repo.set_prompt_archived(selected.id, True)
        st.rerun()

    can_delete = not selected.is_builtin and not used["records"] and not used["datasets"]
    if not can_delete:
        st.caption("Chỉ xoá được prompt không phải prompt gốc và chưa được bản ghi / dataset nào dùng.")
    if st.button("🗑️ Xoá prompt", disabled=not can_delete, key=f"delete_{selected.id}"):
        try:
            repo.delete_prompt(selected.id)
        except ValueError as e:
            st.error(str(e))
            return
        st.rerun()
