"""Kết nối SQLite và định nghĩa bảng.

Bảng `prompts`: prompt cho LLM, có phiên bản. Mỗi lần sửa = một dòng mới (cùng name, version + 1),
    không sửa đè -> bản ghi trỏ tới prompt nào thì biết chính xác nội dung prompt đó.
Bảng `datasets`: mỗi lần chạy thư mục ở chế độ 2, kèm prompt gắn với dataset.
Bảng `records`: mỗi dòng là một sản phẩm đã xử lý.
    - mode     : UPLOAD (chế độ 1) | DATASET (chế độ 2)
    - code     : mã bản ghi, UP-000001 / DS-000002 (tiền tố theo chế độ)
    - dataset  : tên dataset, CHỈ có ở chế độ DATASET (chế độ UPLOAD để NULL)
    - prompt_id: prompt đã dùng để tạo các cột của bản ghi
    - fields_json: các cột thông tin dạng JSON -> dùng chung cho mọi loại tài liệu (profile)
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    profile     TEXT NOT NULL,             -- loại tài liệu: food | drug ...
    content     TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',  -- ghi chú thay đổi
    is_builtin  INTEGER NOT NULL DEFAULT 0, -- 1 = prompt gốc lấy từ code, không sửa trên giao diện
    archived    INTEGER NOT NULL DEFAULT 0, -- 1 = ẩn khỏi danh sách chọn
    created_at  TEXT NOT NULL,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS datasets (
    name        TEXT PRIMARY KEY,
    folder      TEXT NOT NULL,
    profile     TEXT NOT NULL,
    prompt_id   INTEGER REFERENCES prompts(id), -- prompt gắn với dataset (dùng cho các ảnh chạy tiếp)
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    code          TEXT UNIQUE,
    mode          TEXT NOT NULL CHECK (mode IN ('UPLOAD', 'DATASET')),
    dataset       TEXT REFERENCES datasets(name),
    source_key    TEXT,          -- chế độ DATASET: tên file ảnh trong thư mục
    image_paths   TEXT NOT NULL, -- JSON: danh sách đường dẫn ảnh
    profile       TEXT NOT NULL, -- loại tài liệu: food | drug ...
    ocr_method    TEXT NOT NULL, -- cách thực hiện: vision | local
    prompt_id     INTEGER REFERENCES prompts(id), -- prompt đã dùng (NULL nếu chưa chạy được LLM)
    llm_model     TEXT,          -- model LLM đã dùng
    title         TEXT,          -- tên sản phẩm / tên thuốc (để tìm kiếm, hiển thị)
    ocr_text      TEXT,
    fields_json   TEXT,          -- JSON: các cột thông tin
    warnings_json TEXT,          -- JSON: con số LLM đưa ra mà không có trong text OCR
    status        TEXT NOT NULL, -- auto | reviewed | error
    error_message TEXT,
    ocr_seconds   REAL,
    llm_seconds   REAL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    CHECK ((mode = 'DATASET' AND dataset IS NOT NULL AND source_key IS NOT NULL)
        OR (mode = 'UPLOAD' AND dataset IS NULL))
);

-- Mỗi ảnh trong một dataset chỉ có một bản ghi (để chạy tiếp mà không xử lý lại)
CREATE UNIQUE INDEX IF NOT EXISTS idx_records_dataset_source
    ON records(dataset, source_key) WHERE dataset IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_records_mode ON records(mode);
CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
"""

# Cột thêm sau phiên bản đầu: (bảng, cột, kiểu). DB cũ sẽ được thêm cột khi khởi động.
MIGRATIONS = [
    ("datasets", "prompt_id", "INTEGER REFERENCES prompts(id)"),
    ("records", "prompt_id", "INTEGER REFERENCES prompts(id)"),
    ("records", "llm_model", "TEXT"),
]


@contextmanager
def connect(db_path: Path):
    """Mở kết nối, tự commit khi thành công, rollback khi lỗi, luôn đóng kết nối."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for table, column, column_type in MIGRATIONS:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_prompt ON records(prompt_id)")
