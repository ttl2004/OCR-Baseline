import sqlite3

import pytest

from ocr_baseline.constants import MODE_UPLOAD, STATUS_REVIEWED
from ocr_baseline.profiles import get_profile
from ocr_baseline.prompts import builtin_prompts, missing_field_keys
from ocr_baseline.storage import Repository


@pytest.fixture
def repo(tmp_path):
    repo = Repository(tmp_path / "p.db")
    repo.ensure_builtin_prompts(builtin_prompts())
    return repo


def test_builtin_prompts_seeded_once(repo):
    repo.ensure_builtin_prompts(builtin_prompts())  # chạy lại không tạo trùng
    food = repo.list_prompts(profile="food")
    assert len(food) == 1 and food[0].is_builtin and food[0].version == 1
    assert food[0].content == get_profile("food").system_prompt
    assert len(repo.list_prompts(profile="drug")) == 1


def test_builtin_changed_in_code_adds_version(repo):
    name, profile, content = builtin_prompts()[0]
    repo.ensure_builtin_prompts([(name, profile, content + "\n(sửa trong code)")])
    versions = [p.version for p in repo.list_prompts(profile=profile) if p.name == name]
    assert versions == [2, 1]


def test_save_versions_never_overwrite(repo):
    v1 = repo.save_prompt_version("Prompt A", "food", "nội dung 1", "lần đầu")
    v2 = repo.save_prompt_version("Prompt A", "food", "nội dung 2")
    assert (v1.version, v2.version) == (1, 2)
    assert repo.get_prompt(v1.id).content == "nội dung 1"   # bản cũ giữ nguyên
    assert v2.label == "Prompt A · v2"


def test_save_rules(repo):
    builtin = repo.list_prompts(profile="food")[0]
    with pytest.raises(ValueError, match="gốc"):
        repo.save_prompt_version(builtin.name, "food", "sửa prompt gốc")
    repo.save_prompt_version("Prompt A", "food", "x")
    with pytest.raises(ValueError, match="giống hệt"):
        repo.save_prompt_version("Prompt A", "food", "x")
    with pytest.raises(ValueError, match="loại tài liệu"):
        repo.save_prompt_version("Prompt A", "drug", "y")
    with pytest.raises(ValueError):
        repo.save_prompt_version("", "food", "z")


def test_delete_only_unused_non_builtin(repo):
    builtin = repo.list_prompts(profile="food")[0]
    with pytest.raises(ValueError):
        repo.delete_prompt(builtin.id)

    used = repo.save_prompt_version("Đã dùng", "food", "a")
    repo.add_record(mode=MODE_UPLOAD, image_paths=[], profile="food", ocr_method="vision", ocr_text="",
                    fields={}, warnings={}, status=STATUS_REVIEWED, prompt_id=used.id)
    with pytest.raises(ValueError):
        repo.delete_prompt(used.id)
    assert repo.prompt_usage()[used.id]["records"] == 1

    unused = repo.save_prompt_version("Chưa dùng", "food", "b")
    repo.delete_prompt(unused.id)
    assert repo.get_prompt(unused.id) is None


def test_archive_hides_from_list(repo):
    p = repo.save_prompt_version("Prompt A", "food", "x")
    repo.set_prompt_archived(p.id, True)
    assert p.id not in [x.id for x in repo.list_prompts(profile="food")]
    assert p.id in [x.id for x in repo.list_prompts(profile="food", include_archived=True)]


def test_missing_field_keys():
    food = get_profile("food")
    assert missing_field_keys(food, food.system_prompt) == []
    assert "nutrition" in missing_field_keys(food, "chỉ có product_name")


def test_old_database_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE datasets (name TEXT PRIMARY KEY, folder TEXT NOT NULL, profile TEXT NOT NULL,
                               created_at TEXT NOT NULL);
        CREATE TABLE records (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT UNIQUE, mode TEXT NOT NULL,
            dataset TEXT, source_key TEXT, image_paths TEXT NOT NULL, profile TEXT NOT NULL,
            ocr_method TEXT NOT NULL, title TEXT, ocr_text TEXT, fields_json TEXT, warnings_json TEXT,
            status TEXT NOT NULL, error_message TEXT, ocr_seconds REAL, llm_seconds REAL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        INSERT INTO records (code, mode, image_paths, profile, ocr_method, status, created_at, updated_at)
            VALUES ('UP-000001', 'UPLOAD', '[]', 'food', 'vision', 'reviewed', 'x', 'x');
    """)
    conn.close()

    repo = Repository(db)  # tự thêm cột prompt_id, llm_model
    records = repo.list_records()
    assert records[0]["code"] == "UP-000001" and records[0]["prompt_id"] is None
