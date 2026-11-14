"""Kiểu dữ liệu kết quả OCR, dùng chung cho mọi engine."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OcrLine:
    text: str
    score: float
    box: list[list[float]]      # polygon 4 điểm [x, y]
    line_id: int | None = None  # các box cùng line_id nằm trên cùng một dòng hiển thị


@dataclass
class OcrResult:
    lines: list[OcrLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Ghép text theo thứ tự đọc; box cùng dòng hiển thị nối bằng dấu cách."""
        out: list[str] = []
        prev_id = object()
        for line in self.lines:
            if line.line_id is not None and line.line_id == prev_id:
                out[-1] += " " + line.text
            else:
                out.append(line.text)
            prev_id = line.line_id
        return "\n".join(out)

    @property
    def mean_score(self) -> float:
        if not self.lines:
            return 0.0
        return float(np.mean([line.score for line in self.lines]))

    def low_confidence(self, threshold: float = 0.8) -> list[OcrLine]:
        return [line for line in self.lines if line.score < threshold]
