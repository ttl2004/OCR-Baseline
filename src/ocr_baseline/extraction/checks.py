"""Kiểm tra con số: LLM có đưa ra con số nào KHÔNG có trong text OCR không.

Các số này có thể là LLM sửa đúng lỗi OCR, nhưng cũng có thể là LLM tự đoán.
Với thông tin quan trọng (liều dùng, hàm lượng) cần người review so lại với ảnh.
"""
import json
import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> set[str]:
    """Các con số trong text, chuẩn hoá dấu thập phân "," -> "." (vd "6,6" -> "6.6")."""
    return {m.replace(",", ".") for m in _NUMBER.findall(text)}


def _parts(number: str) -> list[str]:
    """Tách dãy mã viết liền: "434.903" -> ["434", "903"].

    Chỉ tách khi mọi phần đều có >= 3 chữ số (kiểu mã INS 434, 903). Số thập phân như "6.6"
    hay "0.25" không bị tách, để vẫn bắt được lỗi đổi số thập phân.
    """
    parts = number.split(".")
    if len(parts) > 1 and all(len(p) >= 3 for p in parts):
        return parts
    return [number]


def _known_numbers(ocr_text: str) -> set[str]:
    known = set()
    for number in numbers_in(ocr_text):
        known.add(number)
        known.update(_parts(number))
    return known


def unverified_numbers(fields: dict, ocr_text: str) -> dict[str, list[str]]:
    """{tên cột: [các số không có trong text OCR]}. Cột không có số đáng ngờ thì không xuất hiện."""
    known = _known_numbers(ocr_text)
    result = {}
    for key, value in fields.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        missing = []
        for number in sorted(numbers_in(text)):
            if number in known:
                continue
            if all(part in known for part in _parts(number)):
                continue  # "434,903" trong output, OCR có "434" và "903" riêng
            missing.append(number)
        if missing:
            result[key] = missing
    return result
