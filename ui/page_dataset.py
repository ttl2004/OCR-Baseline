"""Chế độ 2: chọn thư mục ảnh (mỗi ảnh = 1 sản phẩm) + prompt cho dataset -> tự chạy OCR + LLM -> lưu SQLite.

- Mã bản ghi DS-xxxxxx, cột dataset = tên dataset, cột prompt_id = prompt đã dùng.
- Mỗi dataset gắn với một prompt (mặc định chọn lại prompt đó khi chạy tiếp).
- Có thể dừng giữa chừng (bấm nút bất kỳ / đóng trang) và chạy tiếp: ảnh đã xử lý sẽ được bỏ qua.
"""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ocr_baseline.config import Settings
from ocr_baseline.constants import METHOD_VISION, STATUS_ERROR, STATUS_LABELS
from ocr_baseline.extraction import LlmUnavailableError
from ocr_baseline.ocr import OcrUnavailableError
from ocr_baseline.profiles import PROFILES, Profile
from ocr_baseline.prompts import Prompt
from ocr_baseline.services.batch import plan_dataset, run_dataset

from . import components, state
from .resources import get_engine, get_repo

# Thời gian ước tính mỗi ảnh (OCR + LLM), đo trên máy dev
SECONDS_PER_IMAGE = {METHOD_VISION: 3, "local": 11}


def render(settings: Settings, method: str, profile: Profile) -> None:
    st.header("Chế độ 2 - Chạy thư mục ảnh (dataset)")
    st.caption("Mỗi ảnh trong thư mục là **một sản phẩm**. Kết quả lưu với mã `DS-xxxxxx`, tên dataset và prompt đã "
               "dùng; sau đó xem và chỉnh sửa ở trang **Xem & chỉnh sửa dữ liệu**.")
    repo = get_repo()

    folder_text = st.text_input("Đường dẫn thư mục ảnh (trên máy chạy app)",
                                placeholder="/home/ten/anh/bao_bi")
    folder = Path(folder_text.strip()).expanduser() if folder_text.strip() else None

    col1, col2 = st.columns(2)
    dataset = col1.text_input("Tên dataset (dùng để phân biệt dữ liệu)", value=folder.name if folder else "",
                              help="Muốn so sánh 2 prompt trên cùng thư mục: chạy 2 lần với 2 tên dataset khác nhau.")
    dataset = dataset.strip()
    recursive = col2.checkbox("Lấy cả ảnh trong thư mục con")
    retry_errors = col2.checkbox("Xử lý lại các ảnh bị lỗi lần trước")

    existing = repo.get_dataset(dataset) if dataset else None
    if existing and existing["profile"] != profile.key:
        st.error(f"Dataset '{dataset}' là loại tài liệu '{PROFILES[existing['profile']].label}'. "
                 f"Hãy chọn đúng loại tài liệu ở thanh bên hoặc đặt tên dataset khác.")
        return

    prompt = components.prompt_selector(repo, profile, key="ds_prompt",
                                        default_id=existing["prompt_id"] if existing else None,
                                        label="Prompt cho dataset này", context=dataset)
    if prompt is None:
        return
    prompt_changed = _prompt_change_notice(repo, dataset, existing, prompt)

    _show_existing_datasets(repo)

    if not folder or not dataset:
        return
    if not folder.is_dir():
        st.error(f"Không tìm thấy thư mục: {folder}")
        return

    try:
        plan = plan_dataset(folder, dataset, repo, retry_errors, recursive)
    except ValueError as e:
        st.error(str(e))
        return

    eta = len(plan.todo) * SECONDS_PER_IMAGE.get(method, 10) / 60
    st.info(f"Thư mục có **{len(plan.images)} ảnh**: đã xử lý {plan.skipped}, cần xử lý **{len(plan.todo)}** "
            f"(ước tính ~{eta:.0f} phút). Loại tài liệu: **{profile.label}** · Prompt: **{prompt.label}**.")
    if not plan.todo:
        st.button("Xem & chỉnh sửa dataset này", on_click=state.open_records_of_dataset, args=(dataset,))
        return

    confirmed = True
    if prompt_changed:
        confirmed = st.checkbox("Tôi đồng ý dùng prompt mới cho các ảnh còn lại của dataset này")

    autostart = st.session_state.pop(state.DATASET_AUTOSTART, False)
    clicked = st.button("Bắt đầu / chạy tiếp", type="primary", disabled=not confirmed)
    if (clicked or autostart) and confirmed:
        _run(plan, folder, dataset, method, profile, prompt, settings)


def _prompt_change_notice(repo, dataset: str, existing: dict | None, prompt: Prompt) -> bool:
    """Cảnh báo khi chọn prompt khác với prompt dataset đang dùng. Trả về True nếu có đổi prompt."""
    if not existing or existing["prompt_id"] in (None, prompt.id):
        return False
    old = repo.get_prompt(existing["prompt_id"])
    st.warning(f"Dataset **{dataset}** đang dùng prompt **{old.label if old else existing['prompt_id']}**. "
               f"Các bản ghi đã có giữ nguyên prompt cũ (lưu trong cột prompt); các ảnh xử lý tiếp sẽ dùng "
               f"**{prompt.label}** và dataset sẽ chuyển sang prompt mới. "
               f"Muốn so sánh 2 prompt, nên chạy với một tên dataset khác.")
    return True


def _show_existing_datasets(repo) -> None:
    datasets = repo.list_datasets()
    if not datasets:
        return
    with st.expander(f"Các dataset đã có ({len(datasets)})"):
        df = pd.DataFrame(datasets).drop(columns=["prompt_id"])
        df["profile"] = df["profile"].map(lambda k: PROFILES[k].label if k in PROFILES else k)
        st.dataframe(df.rename(columns={
            "name": "Dataset", "folder": "Thư mục", "profile": "Loại tài liệu", "prompt": "Prompt",
            "created_at": "Tạo lúc", "total": "Tổng", "auto": "Chưa duyệt", "reviewed": "Đã duyệt",
            "error": "Lỗi"}), hide_index=True, width="stretch")


def _run(plan, folder: Path, dataset: str, method: str, profile: Profile, prompt: Prompt,
         settings: Settings) -> None:
    st.caption("Đang chạy... Bấm nút bất kỳ hoặc đổi trang sẽ dừng; bấm **Bắt đầu / chạy tiếp** để chạy tiếp.")
    progress = st.progress(0.0)
    status_line = st.empty()
    table = st.empty()
    rows = []
    start = time.perf_counter()

    try:
        engine = get_engine(method)
        items = run_dataset(plan, folder, dataset, profile, prompt, method, engine, get_repo(), settings)
        for item in items:
            progress.progress(item.index / item.total)
            elapsed = time.perf_counter() - start
            remaining = elapsed / item.index * (item.total - item.index)
            status_line.write(f"{item.index}/{item.total} - `{item.source_key}` - "
                              f"còn khoảng {remaining / 60:.1f} phút")
            rows.append({"Mã": item.code, "Ảnh": item.source_key, "Trạng thái": STATUS_LABELS[item.status],
                         "Kết quả / lỗi": item.message, "Giây": item.seconds})
            table.dataframe(pd.DataFrame(rows[::-1]), hide_index=True, width="stretch")

    except OcrUnavailableError as e:
        components.show_ocr_unavailable(e, method, continue_batch=True)
        _summary(rows, dataset, finished=False)
        return
    except LlmUnavailableError as e:
        st.error(f"Không gọi được LLM, đã dừng: {e}")
        _summary(rows, dataset, finished=False)
        return
    except ValueError as e:  # vd: tên dataset đã dùng cho thư mục khác
        st.error(str(e))
        return

    _summary(rows, dataset, finished=True)


def _summary(rows: list[dict], dataset: str, finished: bool) -> None:
    errors = sum(1 for r in rows if r["Trạng thái"] == STATUS_LABELS[STATUS_ERROR])
    message = f"Đã xử lý {len(rows)} ảnh, {errors} lỗi."
    if finished:
        st.success("Hoàn thành. " + message)
    else:
        st.warning("Đã dừng. " + message + " Các ảnh đã xử lý vẫn được lưu.")
    st.button("Xem & chỉnh sửa dataset này", on_click=state.open_records_of_dataset, args=(dataset,),
              key="open_records_after_run")
