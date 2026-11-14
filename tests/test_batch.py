"""Chế độ 2 với engine giả và LLM giả (không gọi model / API thật)."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from ocr_baseline.config import get_settings
from ocr_baseline.constants import STATUS_AUTO, STATUS_ERROR
from ocr_baseline.ocr import OcrLine, OcrResult, OcrUnavailableError
from ocr_baseline.pipeline import Extraction
from ocr_baseline.profiles import get_profile
from ocr_baseline.prompts import builtin_prompts
from ocr_baseline.services.batch import plan_dataset, run_dataset
from ocr_baseline.storage import Repository


class FakeEngine:
    name = "fake"

    def __init__(self, fail_on=None, unavailable_on=None):
        self.fail_on, self.unavailable_on, self.calls = fail_on, unavailable_on, 0

    def run(self, img):
        self.calls += 1
        if self.calls == self.unavailable_on:
            raise OcrUnavailableError("mất mạng")
        return OcrResult([OcrLine(text="Sữa tươi 180ml", score=0.9, box=[])])


def fake_extract(ocr_text, profile, prompt, settings):
    fields = profile.normalize({"product_name": "Sữa tươi", "net_weight": "180ml"})
    return Extraction(fields=fields, warnings={}, seconds=0.1, prompt_id=prompt.id, llm_model="fake")


@pytest.fixture
def folder(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        cv2.imwrite(str(images / name), np.full((30, 30, 3), 255, np.uint8))
    (images / "hong.jpg").write_bytes(b"khong phai anh")
    (images / "ghi_chu.txt").write_text("bỏ qua")
    return images


def _repo(tmp_path):
    repo = Repository(tmp_path / "t.db")
    repo.ensure_builtin_prompts(builtin_prompts())
    return repo


def _run(plan, folder, repo, engine, prompt=None):
    prompt = prompt or repo.list_prompts(profile="food")[0]
    with patch("ocr_baseline.services.batch.extract", fake_extract):
        return list(run_dataset(plan, folder, "ds", get_profile("food"), prompt, "vision", engine, repo,
                                get_settings()))


def test_runs_saves_and_resumes(folder, tmp_path):
    repo = _repo(tmp_path)
    plan = plan_dataset(folder, "ds", repo)
    assert len(plan.images) == 4 and len(plan.todo) == 4   # .txt bị bỏ qua

    results = _run(plan, folder, repo, FakeEngine())
    statuses = {r.source_key: r.status for r in results}
    assert statuses == {"a.jpg": STATUS_AUTO, "b.jpg": STATUS_AUTO, "c.jpg": STATUS_AUTO, "hong.jpg": STATUS_ERROR}
    assert all(r.code.startswith("DS-") for r in results)
    assert repo.list_records(dataset="ds")[0]["dataset"] == "ds"

    # chạy lại: không còn gì để làm; retry_errors: chỉ ảnh hỏng
    assert plan_dataset(folder, "ds", repo).todo == []
    assert [p.name for p in plan_dataset(folder, "ds", repo, retry_errors=True).todo] == ["hong.jpg"]


def test_stops_when_service_unavailable(folder, tmp_path):
    repo = _repo(tmp_path)
    plan = plan_dataset(folder, "ds", repo)
    with pytest.raises(OcrUnavailableError):
        _run(plan, folder, repo, FakeEngine(unavailable_on=2))
    # ảnh đầu đã được lưu, các ảnh còn lại sẽ chạy tiếp lần sau
    assert len(repo.list_records(dataset="ds")) == 1
    assert len(plan_dataset(folder, "ds", repo).todo) == 3


def test_records_store_prompt_and_dataset_prompt_changes(folder, tmp_path):
    repo = _repo(tmp_path)
    base = repo.list_prompts(profile="food")[0]
    plan = plan_dataset(folder, "ds", repo)
    plan.todo = plan.todo[:2]
    _run(plan, folder, repo, FakeEngine(), base)
    assert repo.get_dataset("ds")["prompt_id"] == base.id

    # chạy tiếp với prompt mới: bản ghi cũ giữ prompt cũ, bản ghi mới dùng prompt mới
    new = repo.save_prompt_version("Prompt thử", "food", base.content + "\n10. Thêm quy ước.")
    _run(plan_dataset(folder, "ds", repo), folder, repo, FakeEngine(), new)
    assert repo.get_dataset("ds")["prompt_id"] == new.id
    by_image = {r["source_key"]: r for r in repo.list_records(dataset="ds")}
    assert by_image["a.jpg"]["prompt_id"] == base.id
    assert by_image["c.jpg"]["prompt_id"] == new.id
    assert by_image["c.jpg"]["prompt_name"] == "Prompt thử" and by_image["c.jpg"]["llm_model"] == "fake"
    assert by_image["hong.jpg"]["prompt_id"] is None   # lỗi OCR -> chưa chạy LLM
    assert repo.dataset_prompt_ids("ds") == {base.id, new.id}
