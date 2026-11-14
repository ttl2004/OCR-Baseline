"""Đọc / ghi dữ liệu. Giao diện và service chỉ dùng class Repository, không viết SQL ở chỗ khác."""
import json
from datetime import datetime
from pathlib import Path

from ..constants import CODE_PREFIX, MODE_DATASET, MODE_UPLOAD
from ..prompts import Prompt
from .database import connect, init_db

# Các cột trả về khi liệt kê bản ghi (không lấy text dài để nhẹ), kèm tên + phiên bản prompt
LIST_SQL = """
    SELECT r.id, r.code, r.mode, r.dataset, r.source_key, r.profile, r.ocr_method, r.prompt_id,
           p.name AS prompt_name, p.version AS prompt_version, r.llm_model, r.title, r.status,
           r.error_message, r.ocr_seconds, r.llm_seconds, r.created_at, r.updated_at
    FROM records r LEFT JOIN prompts p ON p.id = r.prompt_id
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _row_to_prompt(row) -> Prompt:
    return Prompt(id=row["id"], name=row["name"], version=row["version"], profile=row["profile"],
                  content=row["content"], note=row["note"], is_builtin=bool(row["is_builtin"]),
                  archived=bool(row["archived"]), created_at=row["created_at"])


class Repository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        init_db(self.db_path)

    # ================================================================== prompts
    def ensure_builtin_prompts(self, builtins: list[tuple[str, str, str]]) -> None:
        """Nạp prompt gốc từ code. Nếu nội dung trong code đổi -> thêm phiên bản mới (không sửa đè)."""
        for name, profile, content in builtins:
            latest = self.latest_prompt_by_name(name)
            if latest is None or latest.content != content:
                self._insert_prompt(name, profile, content, note="Prompt gốc từ code", is_builtin=True)

    def latest_prompt_by_name(self, name: str) -> Prompt | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM prompts WHERE name = ? ORDER BY version DESC LIMIT 1",
                               (name,)).fetchone()
        return _row_to_prompt(row) if row else None

    def save_prompt_version(self, name: str, profile: str, content: str, note: str = "") -> Prompt:
        """Lưu prompt: tên mới -> phiên bản 1; tên đã có -> phiên bản tiếp theo.

        Prompt gốc (từ code) không thêm phiên bản được: hãy lưu với tên khác.
        """
        name, content = name.strip(), content.strip()
        if not name or not content:
            raise ValueError("Tên và nội dung prompt không được để trống")
        latest = self.latest_prompt_by_name(name)
        if latest is not None:
            if latest.is_builtin:
                raise ValueError(f"'{name}' là prompt gốc, không sửa được. Hãy lưu với tên khác.")
            if latest.profile != profile:
                raise ValueError(f"Tên '{name}' đã dùng cho loại tài liệu '{latest.profile}'.")
            if latest.content == content:
                raise ValueError("Nội dung giống hệt phiên bản mới nhất, không cần lưu.")
        return self._insert_prompt(name, profile, content, note=note, is_builtin=False)

    def _insert_prompt(self, name: str, profile: str, content: str, note: str, is_builtin: bool) -> Prompt:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT MAX(version) AS v FROM prompts WHERE name = ?", (name,)).fetchone()
            version = (row["v"] or 0) + 1
            cur = conn.execute(
                "INSERT INTO prompts (name, version, profile, content, note, is_builtin, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, version, profile, content, note, int(is_builtin), _now()))
            prompt_id = cur.lastrowid
        return self.get_prompt(prompt_id)

    def get_prompt(self, prompt_id: int) -> Prompt | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
        return _row_to_prompt(row) if row else None

    def list_prompts(self, profile: str | None = None, include_archived: bool = False) -> list[Prompt]:
        """Mới nhất trước. Mặc định bỏ prompt đã ẩn."""
        sql, params = "SELECT * FROM prompts WHERE 1 = 1", []
        if profile:
            sql += " AND profile = ?"
            params.append(profile)
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY is_builtin DESC, name, version DESC"
        with connect(self.db_path) as conn:
            return [_row_to_prompt(row) for row in conn.execute(sql, params)]

    def prompt_usage(self) -> dict[int, dict]:
        """{prompt_id: {"records": số bản ghi, "datasets": [tên dataset đang gắn prompt]}}."""
        usage: dict[int, dict] = {}
        with connect(self.db_path) as conn:
            for row in conn.execute("SELECT prompt_id, COUNT(*) AS n FROM records "
                                    "WHERE prompt_id IS NOT NULL GROUP BY prompt_id"):
                usage.setdefault(row["prompt_id"], {"records": 0, "datasets": []})["records"] = row["n"]
            for row in conn.execute("SELECT name, prompt_id FROM datasets WHERE prompt_id IS NOT NULL"):
                usage.setdefault(row["prompt_id"], {"records": 0, "datasets": []})["datasets"].append(row["name"])
        return usage

    def set_prompt_archived(self, prompt_id: int, archived: bool) -> None:
        with connect(self.db_path) as conn:
            conn.execute("UPDATE prompts SET archived = ? WHERE id = ?", (int(archived), prompt_id))

    def delete_prompt(self, prompt_id: int) -> None:
        """Chỉ xoá được prompt chưa được bản ghi / dataset nào dùng và không phải prompt gốc."""
        prompt = self.get_prompt(prompt_id)
        if prompt is None:
            return
        if prompt.is_builtin:
            raise ValueError("Không xoá được prompt gốc")
        if prompt_id in self.prompt_usage():
            raise ValueError("Prompt đã được dùng, không xoá được (có thể ẩn đi)")
        with connect(self.db_path) as conn:
            conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))

    # ================================================================== datasets
    def ensure_dataset(self, name: str, folder: str, profile: str, prompt_id: int | None = None) -> None:
        """Tạo dataset nếu chưa có. Nếu đã có thì phải cùng thư mục và cùng loại tài liệu.

        prompt_id: prompt gắn với dataset (dùng cho các ảnh xử lý tiếp). Dataset đã có thì cập nhật.
        """
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT folder, profile FROM datasets WHERE name = ?", (name,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO datasets (name, folder, profile, prompt_id, created_at) "
                             "VALUES (?, ?, ?, ?, ?)", (name, folder, profile, prompt_id, _now()))
                return
            if row["folder"] != folder or row["profile"] != profile:
                raise ValueError(
                    f"Dataset '{name}' đã tồn tại với thư mục '{row['folder']}' và loại tài liệu "
                    f"'{row['profile']}'. Hãy đặt tên dataset khác.")
            if prompt_id is not None:
                conn.execute("UPDATE datasets SET prompt_id = ? WHERE name = ?", (prompt_id, name))

    def get_dataset(self, name: str) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM datasets WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def list_datasets(self) -> list[dict]:
        sql = """
            SELECT d.name, d.folder, d.profile, d.prompt_id,
                   p.name || ' · v' || p.version AS prompt,
                   d.created_at,
                   COUNT(r.id) AS total,
                   SUM(r.status = 'auto') AS auto,
                   SUM(r.status = 'reviewed') AS reviewed,
                   SUM(r.status = 'error') AS error
            FROM datasets d
            LEFT JOIN prompts p ON p.id = d.prompt_id
            LEFT JOIN records r ON r.dataset = d.name
            GROUP BY d.name ORDER BY d.created_at DESC
        """
        with connect(self.db_path) as conn:
            return [dict(row) for row in conn.execute(sql)]

    def dataset_prompt_ids(self, dataset: str) -> set[int]:
        """Các prompt đã được dùng cho bản ghi trong dataset."""
        with connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT prompt_id FROM records "
                                "WHERE dataset = ? AND prompt_id IS NOT NULL", (dataset,))
            return {row["prompt_id"] for row in rows}

    def processed_sources(self, dataset: str, include_errors: bool = True) -> set[str]:
        """Tên các ảnh của dataset đã có bản ghi (để chạy tiếp thì bỏ qua)."""
        sql = "SELECT source_key FROM records WHERE dataset = ?"
        if not include_errors:
            sql += " AND status != 'error'"
        with connect(self.db_path) as conn:
            return {row["source_key"] for row in conn.execute(sql, (dataset,))}

    # ================================================================== records
    def add_record(self, *, mode: str, image_paths: list[str], profile: str, ocr_method: str,
                   ocr_text: str, fields: dict, warnings: dict, status: str, title: str = "",
                   prompt_id: int | None = None, llm_model: str | None = None,
                   dataset: str | None = None, source_key: str | None = None,
                   error_message: str | None = None, ocr_seconds: float | None = None,
                   llm_seconds: float | None = None) -> tuple[int, str]:
        """Thêm bản ghi mới. Trả về (id, mã bản ghi)."""
        if mode not in CODE_PREFIX:
            raise ValueError(f"Chế độ không hợp lệ: {mode}")
        if mode == MODE_DATASET and not dataset:
            raise ValueError("Chế độ DATASET bắt buộc có tên dataset")
        if mode == MODE_UPLOAD:
            dataset, source_key = None, None

        now = _now()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO records (mode, dataset, source_key, image_paths, profile, ocr_method, prompt_id,
                       llm_model, title, ocr_text, fields_json, warnings_json, status, error_message,
                       ocr_seconds, llm_seconds, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mode, dataset, source_key, _to_json(image_paths), profile, ocr_method, prompt_id,
                 llm_model, title, ocr_text, _to_json(fields), _to_json(warnings), status, error_message,
                 ocr_seconds, llm_seconds, now, now),
            )
            record_id = cur.lastrowid
            code = f"{CODE_PREFIX[mode]}-{record_id:06d}"
            conn.execute("UPDATE records SET code = ? WHERE id = ?", (code, record_id))
        return record_id, code

    def find_dataset_record(self, dataset: str, source_key: str) -> int | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT id FROM records WHERE dataset = ? AND source_key = ?",
                               (dataset, source_key)).fetchone()
        return row["id"] if row else None

    def update_record(self, record_id: int, **values) -> None:
        """Cập nhật một số cột. Các cột dict/list (fields, warnings, image_paths) tự chuyển sang JSON."""
        json_columns = {"fields": "fields_json", "warnings": "warnings_json", "image_paths": "image_paths"}
        allowed = {"ocr_method", "prompt_id", "llm_model", "title", "ocr_text", "status", "error_message",
                   "ocr_seconds", "llm_seconds"}

        columns, params = [], []
        for name, value in values.items():
            if name in json_columns:
                columns.append(f"{json_columns[name]} = ?")
                params.append(_to_json(value))
            elif name in allowed:
                columns.append(f"{name} = ?")
                params.append(value)
            else:
                raise ValueError(f"Không cập nhật được cột: {name}")
        columns.append("updated_at = ?")
        params.append(_now())
        params.append(record_id)

        with connect(self.db_path) as conn:
            conn.execute(f"UPDATE records SET {', '.join(columns)} WHERE id = ?", params)

    def get_record(self, record_id: int) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["image_paths"] = json.loads(record["image_paths"] or "[]")
        record["fields"] = json.loads(record.pop("fields_json") or "{}")
        record["warnings"] = json.loads(record.pop("warnings_json") or "{}")
        return record

    def list_records(self, mode: str | None = None, dataset: str | None = None, status: str | None = None,
                     profile: str | None = None, prompt_id: int | None = None) -> list[dict]:
        conditions, params = [], []
        filters = (("r.mode", mode), ("r.dataset", dataset), ("r.status", status), ("r.profile", profile),
                   ("r.prompt_id", prompt_id))
        for column, value in filters:
            if value:
                conditions.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with connect(self.db_path) as conn:
            rows = conn.execute(f"{LIST_SQL} {where} ORDER BY r.id DESC", params)
            return [dict(row) for row in rows]

    def export_records(self, record_ids: list[int]) -> list[dict]:
        """Bản ghi đầy đủ (kèm fields) để xuất CSV."""
        return [r for r in (self.get_record(i) for i in record_ids) if r]

    def delete_record(self, record_id: int) -> None:
        with connect(self.db_path) as conn:
            conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
