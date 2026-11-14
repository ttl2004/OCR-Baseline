"""Các bước xử lý dùng chung cho cả 2 chế độ: ảnh -> OCR -> text -> LLM -> các cột + cảnh báo."""
import time
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .extraction import extract_fields, unverified_numbers
from .ocr import OcrLine
from .profiles import Profile
from .prompts import Prompt


@dataclass
class PageOcr:
    """Kết quả OCR của một ảnh."""
    name: str
    text: str
    n_lines: int
    mean_score: float
    seconds: float
    low_confidence: list[OcrLine] = field(default_factory=list)  # dòng có độ tin cậy thấp


@dataclass
class Extraction:
    """Kết quả LLM của một sản phẩm."""
    fields: dict
    warnings: dict      # {cột: [con số không có trong text OCR]}
    seconds: float
    prompt_id: int | None = None   # prompt đã dùng -> lưu vào records.prompt_id
    llm_model: str = ""


def ocr_image(engine, img: np.ndarray, name: str = "", low_threshold: float = 0.8) -> PageOcr:
    start = time.perf_counter()
    result = engine.run(img)
    seconds = round(time.perf_counter() - start, 2)
    return PageOcr(
        name=name,
        text=result.text,
        n_lines=len(result.lines),
        mean_score=round(result.mean_score, 3),
        seconds=seconds,
        low_confidence=result.low_confidence(low_threshold),
    )


def join_pages(pages: list[PageOcr]) -> str:
    """Ghép text nhiều ảnh (nhiều mặt) của cùng một sản phẩm."""
    return "\n\n".join(p.text for p in pages if p.text)


def extract(ocr_text: str, profile: Profile, prompt: Prompt, settings: Settings) -> Extraction:
    if prompt.profile != profile.key:
        raise ValueError(f"Prompt '{prompt.label}' dành cho loại tài liệu '{prompt.profile}', "
                         f"không dùng được cho '{profile.key}'")
    fields, info = extract_fields(ocr_text, profile, prompt.content, settings)
    return Extraction(fields=fields, warnings=unverified_numbers(fields, ocr_text), seconds=info["seconds"],
                      prompt_id=prompt.id, llm_model=info["model"])
