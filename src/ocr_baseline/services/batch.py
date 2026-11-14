"""Chế độ 2: chạy cả thư mục ảnh (mỗi ảnh = 1 sản phẩm), lưu từng kết quả vào SQLite.

- Chạy tiếp được: ảnh đã có bản ghi trong dataset sẽ được bỏ qua.
- Lỗi của riêng một ảnh (ảnh hỏng, LLM trả JSON lỗi): ghi bản ghi trạng thái "error", chạy tiếp.
- Dịch vụ không dùng được (mất mạng, sai key...): dừng ngay và báo lên giao diện
  (OcrUnavailableError / LlmUnavailableError) để người dùng chọn cách khác.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..config import Settings
from ..constants import MODE_DATASET, STATUS_AUTO, STATUS_ERROR
from ..extraction import LlmError, LlmUnavailableError
from ..ocr import OcrError, OcrUnavailableError
from ..pipeline import extract, ocr_image
from ..profiles import Profile
from ..prompts import Prompt
from ..storage import Repository
from .images import list_images, read_image


@dataclass
class BatchPlan:
    images: list[Path]   # tất cả ảnh trong thư mục
    todo: list[Path]     # ảnh cần xử lý lần này

    @property
    def skipped(self) -> int:
        return len(self.images) - len(self.todo)


@dataclass
class ItemResult:
    index: int          # thứ tự trong danh sách cần xử lý (bắt đầu từ 1)
    total: int
    source_key: str     # tên file ảnh
    code: str           # mã bản ghi đã lưu
    status: str         # auto | error
    message: str        # tiêu đề sản phẩm hoặc nội dung lỗi
    seconds: float


def source_key_of(image: Path, folder: Path) -> str:
    return image.relative_to(folder).as_posix()


def plan_dataset(folder: Path, dataset: str, repo: Repository, retry_errors: bool = False,
                 recursive: bool = False) -> BatchPlan:
    folder = Path(folder)
    images = list_images(folder, recursive)
    # retry_errors=True: ảnh bị lỗi lần trước sẽ được xử lý lại
    done = repo.processed_sources(dataset, include_errors=not retry_errors)
    todo = [img for img in images if source_key_of(img, folder) not in done]
    return BatchPlan(images=images, todo=todo)


def run_dataset(plan: BatchPlan, folder: Path, dataset: str, profile: Profile, prompt: Prompt, method: str,
                engine, repo: Repository, settings: Settings) -> Iterator[ItemResult]:
    """Xử lý lần lượt từng ảnh, trả về kết quả từng ảnh ngay khi xong (để giao diện cập nhật tiến độ).

    `prompt` được gắn với dataset và lưu vào từng bản ghi (records.prompt_id).
    """
    folder = Path(folder)
    repo.ensure_dataset(dataset, str(folder), profile.key, prompt.id)
    total = len(plan.todo)

    for index, image_path in enumerate(plan.todo, start=1):
        key = source_key_of(image_path, folder)
        record = {
            "mode": MODE_DATASET, "dataset": dataset, "source_key": key, "image_paths": [str(image_path)],
            "profile": profile.key, "ocr_method": method, "ocr_text": "", "fields": profile.empty_fields(),
            "warnings": {}, "status": STATUS_AUTO, "title": "", "error_message": None,
            "ocr_seconds": None, "llm_seconds": None, "prompt_id": prompt.id, "llm_model": settings.deepseek_model,
        }

        # Bước 1: OCR
        try:
            page = ocr_image(engine, read_image(image_path), key)
            record["ocr_text"] = page.text
            record["ocr_seconds"] = page.seconds
        except OcrUnavailableError:
            raise  # dừng cả dataset, giao diện sẽ báo để chuyển sang OCR local
        except (OcrError, ValueError) as e:
            record["status"], record["error_message"] = STATUS_ERROR, f"Lỗi OCR: {e}"
            record["prompt_id"], record["llm_model"] = None, None  # chưa chạy LLM -> chưa dùng prompt nào

        # Bước 2: LLM
        if record["status"] != STATUS_ERROR:
            try:
                result = extract(record["ocr_text"], profile, prompt, settings)
                record["fields"] = result.fields
                record["warnings"] = result.warnings
                record["llm_seconds"] = result.seconds
                record["title"] = profile.title_of(result.fields)
                record["prompt_id"] = result.prompt_id
                record["llm_model"] = result.llm_model
            except LlmUnavailableError:
                raise
            except LlmError as e:
                record["status"], record["error_message"] = STATUS_ERROR, f"Lỗi LLM: {e}"

        code = save_dataset_record(repo, record)
        seconds = (record["ocr_seconds"] or 0) + (record["llm_seconds"] or 0)
        message = record["error_message"] or record["title"]
        yield ItemResult(index, total, key, code, record["status"], message, round(seconds, 2))


def save_dataset_record(repo: Repository, record: dict) -> str:
    """Thêm mới, hoặc ghi đè bản ghi cũ của cùng ảnh (khi chạy lại ảnh lỗi). Trả về mã bản ghi."""
    existing_id = repo.find_dataset_record(record["dataset"], record["source_key"])
    if existing_id is None:
        _, code = repo.add_record(**record)
        return code

    repo.update_record(
        existing_id, ocr_method=record["ocr_method"], ocr_text=record["ocr_text"], fields=record["fields"],
        warnings=record["warnings"], status=record["status"], title=record["title"],
        error_message=record["error_message"], ocr_seconds=record["ocr_seconds"],
        llm_seconds=record["llm_seconds"], prompt_id=record.get("prompt_id"), llm_model=record.get("llm_model"),
    )
    return repo.get_record(existing_id)["code"]
