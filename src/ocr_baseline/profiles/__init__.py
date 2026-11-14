"""Các loại tài liệu (profile). Mỗi profile = danh sách cột + prompt LLM.

Thêm loại tài liệu mới:
    1. Tạo file mới trong thư mục này (copy food.py hoặc drug.py), sửa FIELDS và SYSTEM_PROMPT.
    2. Thêm vào dict PROFILES bên dưới.
Giao diện và DB tự dùng được, không phải sửa chỗ khác (các cột được lưu dạng JSON).
"""
from . import drug, food
from .base import KIND_LIST, KIND_NUTRITION, KIND_TEXT, Field, Profile

PROFILES: dict[str, Profile] = {
    food.PROFILE.key: food.PROFILE,
    drug.PROFILE.key: drug.PROFILE,
}


def get_profile(key: str) -> Profile:
    if key not in PROFILES:
        raise ValueError(f"Loại tài liệu không hợp lệ: {key}. Có: {', '.join(PROFILES)}")
    return PROFILES[key]


__all__ = ["KIND_LIST", "KIND_NUTRITION", "KIND_TEXT", "PROFILES", "Field", "Profile", "get_profile"]
