"""Định nghĩa chung cho một "loại tài liệu" (profile): danh sách cột + prompt cho LLM."""
from dataclasses import dataclass

# Kiểu giá trị của một cột
KIND_TEXT = "text"            # một chuỗi
KIND_LIST = "list"            # danh sách chuỗi (mỗi mục một dòng khi hiển thị)
KIND_NUTRITION = "nutrition"  # danh sách {"name": ..., "value": ...}


@dataclass(frozen=True)
class Field:
    key: str           # tên cột trong JSON (tiếng Anh/không dấu)
    label: str         # tên hiển thị trên giao diện
    kind: str = KIND_TEXT


@dataclass(frozen=True)
class Profile:
    key: str                  # mã lưu trong DB (cột `profile`)
    label: str                # tên hiển thị
    fields: tuple[Field, ...]
    system_prompt: str        # prompt GỐC gửi cho LLM (được nạp vào bảng prompts khi khởi động)
    title_field: str          # cột dùng làm tên bản ghi (vd tên sản phẩm)
    default_prompt_name: str  # tên của prompt gốc trong bảng prompts

    def field_keys(self) -> list[str]:
        return [f.key for f in self.fields]

    def empty_fields(self) -> dict:
        return {f.key: empty_value(f.kind) for f in self.fields}

    def normalize(self, raw: dict) -> dict:
        """Đưa output của LLM (có thể thiếu key, sai kiểu) về đúng cấu trúc của profile."""
        result = {}
        for f in self.fields:
            result[f.key] = normalize_value(raw.get(f.key), f.kind)
        return result

    def title_of(self, fields: dict) -> str:
        return str(fields.get(self.title_field) or "").strip()


def empty_value(kind: str):
    if kind == KIND_TEXT:
        return ""
    return []


def normalize_value(value, kind: str):
    if kind == KIND_TEXT:
        return to_text(value)
    if kind == KIND_LIST:
        return to_list(value)
    if kind == KIND_NUTRITION:
        return to_nutrition(value)
    raise ValueError(f"Kiểu cột không hợp lệ: {kind}")


def to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(to_text(v) for v in value if to_text(v))
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value).strip()


def to_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if not isinstance(value, list):
        value = str(value).split("\n")
    items = []
    for v in value:
        text = to_text(v)
        if text:
            items.append(text)
    return items


def to_nutrition(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if isinstance(item, dict):
            name = to_text(item.get("name"))
            amount = to_text(item.get("value"))
            if name or amount:
                rows.append({"name": name, "value": amount})
    return rows
