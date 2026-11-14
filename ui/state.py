"""Tên các key trong st.session_state dùng chung giữa các trang + hàm khởi tạo."""
import streamlit as st

from ocr_baseline.constants import METHOD_LOCAL

# Lựa chọn trên thanh bên
PAGE = "page"
METHOD = "ocr_method"
PROFILE = "profile"

# Chế độ 2: tự chạy tiếp sau khi chuyển sang OCR local
DATASET_AUTOSTART = "dataset_autostart"
# Trang dữ liệu: dataset đang lọc (được đặt từ trang chế độ 2)
RECORDS_DATASET_FILTER = "records_dataset_filter"

PAGE_UPLOAD = "upload"
PAGE_DATASET = "dataset"
PAGE_RECORDS = "records"
PAGE_PROMPTS = "prompts"

PAGE_LABELS = {
    PAGE_UPLOAD: "Chế độ 1 - Tải ảnh lên",
    PAGE_DATASET: "Chế độ 2 - Chạy thư mục ảnh (dataset)",
    PAGE_RECORDS: "Xem & chỉnh sửa dữ liệu",
    PAGE_PROMPTS: "Quản lý prompt",
}


def init_state(default_method: str, default_profile: str) -> None:
    st.session_state.setdefault(PAGE, PAGE_UPLOAD)
    st.session_state.setdefault(METHOD, default_method)
    st.session_state.setdefault(PROFILE, default_profile)


# ---- callback cho nút bấm (chạy trước khi trang vẽ lại, nên được phép đổi giá trị widget) ----
def switch_to_local() -> None:
    st.session_state[METHOD] = METHOD_LOCAL


def switch_to_local_and_continue() -> None:
    st.session_state[METHOD] = METHOD_LOCAL
    st.session_state[DATASET_AUTOSTART] = True


def open_records_of_dataset(dataset: str) -> None:
    st.session_state[PAGE] = PAGE_RECORDS
    st.session_state[RECORDS_DATASET_FILTER] = dataset
