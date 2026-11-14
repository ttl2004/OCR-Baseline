"""Prompt cho LLM, quản lý theo phiên bản.

- Mỗi prompt thuộc một loại tài liệu (profile): prompt phải yêu cầu LLM trả về đúng các cột của profile đó.
- Prompt KHÔNG bị sửa đè: mỗi lần sửa tạo phiên bản mới (cùng tên, version + 1). Bản ghi lưu prompt_id
  nên luôn biết chính xác nội dung prompt đã tạo ra bản ghi đó.
- Prompt gốc (is_builtin) lấy từ code trong profiles/, không sửa được trên giao diện.
"""
from dataclasses import dataclass

from .profiles import PROFILES, Profile


@dataclass(frozen=True)
class Prompt:
    id: int
    name: str
    version: int
    profile: str
    content: str
    note: str = ""
    is_builtin: bool = False
    archived: bool = False
    created_at: str = ""

    @property
    def label(self) -> str:
        """Tên hiển thị, vd "Thực phẩm - prompt v2 (gốc) · v1"."""
        return f"{self.name} · v{self.version}"


def builtin_prompts() -> list[tuple[str, str, str]]:
    """Prompt gốc của mỗi profile: [(tên, profile, nội dung)]."""
    return [(p.default_prompt_name, p.key, p.system_prompt) for p in PROFILES.values()]


def missing_field_keys(profile: Profile, content: str) -> list[str]:
    """Các cột của profile không được nhắc tới trong prompt -> LLM có thể không trả về cột đó."""
    return [key for key in profile.field_keys() if key not in content]
