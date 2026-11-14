from unittest.mock import patch

import numpy as np
import pytest
import requests

from ocr_baseline.ocr import OcrImageError, OcrUnavailableError
from ocr_baseline.ocr.vision import VisionEngine, parse_response


def _word(text, x, brk=None):
    symbols = [{"text": c} for c in text]
    if brk:
        symbols[-1]["property"] = {"detectedBreak": {"type": brk}}
    box = {"vertices": [{"x": x, "y": 0}, {"x": x + 10, "y": 0}, {"x": x + 10, "y": 8}, {"x": x, "y": 8}]}
    return {"symbols": symbols, "boundingBox": box, "confidence": 0.9}


def test_parse_words_and_lines():
    raw = {"fullTextAnnotation": {"pages": [{"blocks": [{"paragraphs": [{"words": [
        _word("Paracetamol", 0, "SPACE"), _word("500", 20), _word("mg", 30, "EOL_SURE_SPACE"),
        _word("Hộp", 0, "SPACE"), _word("10", 20, "SPACE"), _word("vỉ", 30, "LINE_BREAK"),
    ]}]}]}]}}
    result = parse_response(raw)
    assert result.text == "Paracetamol 500mg\nHộp 10 vỉ"
    assert parse_response({}).text == ""


def test_missing_key_is_unavailable():
    with pytest.raises(OcrUnavailableError):
        VisionEngine("")


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


IMG = np.full((20, 20, 3), 255, np.uint8)


def test_network_error_is_unavailable():
    engine = VisionEngine("AIza-test")
    with patch.object(engine.session, "post", side_effect=requests.ConnectionError("no network")):
        with pytest.raises(OcrUnavailableError):
            engine.run(IMG)


def test_http_403_is_unavailable():
    engine = VisionEngine("AIza-test")
    with patch.object(engine.session, "post", return_value=_Resp(403, text="billing")):
        with pytest.raises(OcrUnavailableError, match="403"):
            engine.run(IMG)


def test_image_error_is_per_image():
    engine = VisionEngine("AIza-test")
    payload = {"responses": [{"error": {"message": "Bad image data"}}]}
    with patch.object(engine.session, "post", return_value=_Resp(200, payload)):
        with pytest.raises(OcrImageError):
            engine.run(IMG)
