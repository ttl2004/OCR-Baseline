import sqlite3

import pytest

from ocr_baseline.constants import MODE_DATASET, MODE_UPLOAD, STATUS_AUTO, STATUS_ERROR, STATUS_REVIEWED
from ocr_baseline.services.batch import save_dataset_record
from ocr_baseline.storage import Repository


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / "test.db")


def _record(**overrides):
    record = {"mode": MODE_UPLOAD, "image_paths": ["a.jpg"], "profile": "food", "ocr_method": "vision",
              "ocr_text": "Sữa tươi 100ml", "fields": {"product_name": "Sữa"}, "warnings": {},
              "status": STATUS_REVIEWED, "title": "Sữa"}
    record.update(overrides)
    return record


def test_codes_have_prefix_by_mode(repo):
    _, up = repo.add_record(**_record())
    repo.ensure_dataset("ds1", "/tmp/ds1", "food")
    _, ds = repo.add_record(**_record(mode=MODE_DATASET, dataset="ds1", source_key="1.jpg"))
    assert up.startswith("UP-") and ds.startswith("DS-")


def test_upload_record_has_no_dataset(repo):
    record_id, _ = repo.add_record(**_record(dataset="khong_duoc_luu", source_key="x"))
    record = repo.get_record(record_id)
    assert record["dataset"] is None and record["mode"] == MODE_UPLOAD
    assert record["fields"] == {"product_name": "Sữa"}


def test_dataset_record_requires_dataset(repo):
    with pytest.raises(ValueError):
        repo.add_record(**_record(mode=MODE_DATASET))


def test_same_image_in_dataset_is_unique(repo):
    repo.ensure_dataset("ds1", "/tmp/ds1", "food")
    repo.add_record(**_record(mode=MODE_DATASET, dataset="ds1", source_key="1.jpg"))
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_record(**_record(mode=MODE_DATASET, dataset="ds1", source_key="1.jpg"))


def test_dataset_name_cannot_point_to_other_folder(repo):
    repo.ensure_dataset("ds1", "/tmp/ds1", "food")
    repo.ensure_dataset("ds1", "/tmp/ds1", "food")  # cùng thư mục: OK
    with pytest.raises(ValueError):
        repo.ensure_dataset("ds1", "/tmp/khac", "food")


def test_resume_and_retry_errors(repo):
    repo.ensure_dataset("ds1", "/tmp/ds1", "food")
    ok = _record(mode=MODE_DATASET, dataset="ds1", source_key="1.jpg", status=STATUS_AUTO)
    bad = _record(mode=MODE_DATASET, dataset="ds1", source_key="2.jpg", status=STATUS_ERROR,
                  error_message="Lỗi OCR")
    save_dataset_record(repo, {**ok, "error_message": None, "ocr_seconds": 1, "llm_seconds": 1})
    code = save_dataset_record(repo, {**bad, "ocr_seconds": None, "llm_seconds": None})

    assert repo.processed_sources("ds1") == {"1.jpg", "2.jpg"}
    assert repo.processed_sources("ds1", include_errors=False) == {"1.jpg"}

    # chạy lại ảnh lỗi -> ghi đè bản ghi cũ, giữ nguyên mã
    fixed = {**bad, "status": STATUS_AUTO, "error_message": None, "ocr_seconds": 1, "llm_seconds": 1}
    assert save_dataset_record(repo, fixed) == code
    assert len(repo.list_records(dataset="ds1")) == 2
    assert repo.list_records(dataset="ds1", status=STATUS_ERROR) == []


def test_update_and_filter(repo):
    record_id, _ = repo.add_record(**_record(status=STATUS_AUTO))
    repo.update_record(record_id, fields={"product_name": "Sữa chua"}, status=STATUS_REVIEWED)
    assert repo.get_record(record_id)["fields"]["product_name"] == "Sữa chua"
    assert len(repo.list_records(mode=MODE_UPLOAD, status=STATUS_REVIEWED)) == 1
    with pytest.raises(ValueError):
        repo.update_record(record_id, code="XX")  # cột không cho phép sửa
