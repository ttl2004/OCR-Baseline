"""Nhận dạng dòng chữ tiếng Việt bằng VietOCR, tối ưu cho CPU.

Hàm `translate()` gốc của VietOCR, ở mỗi bước sinh ký tự lại chạy lại decoder trên
TOÀN BỘ chuỗi đã sinh (độ phức tạp O(L^2)). `_greedy_decode_cached` lưu đầu vào của
từng layer decoder nên mỗi bước chỉ tính một vị trí mới. Kết quả giống hệt bản gốc
(đã kiểm tra trong project OCR: tests/test_vietocr_decode.py), nhanh hơn ~1.7-1.8 lần.
"""
import math
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

MAX_SEQ_LEN = 128
SOS, EOS = 1, 2


class VietOcrRecognizer:
    def __init__(self, model_name: str, models_dir: Path, batch_size: int = 32,
                 use_cached_decode: bool = True):
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name(model_name)
        cfg["weights"] = ensure_weights(model_name, cfg["weights"], models_dir)
        cfg["device"] = "cpu"
        cfg["cnn"]["pretrained"] = False
        predictor = Predictor(cfg)

        self.model = predictor.model.eval()
        self.vocab = predictor.vocab
        self.image_cfg = cfg["dataset"]
        self.batch_size = batch_size
        # decode có cache chỉ áp dụng cho model transformer; seq2seq vốn đã sinh tăng dần
        self.use_cached_decode = use_cached_decode and cfg["seq_modeling"] == "transformer"

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Nhận dạng các ảnh dòng chữ (BGR). Trả về [(text, độ tin cậy)] đúng thứ tự đầu vào."""
        from vietocr.tool.translate import process_input

        # gom các ảnh cùng độ rộng sau resize vào một batch (giống VietOCR gốc, không cần padding)
        tensors = []
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, crop in enumerate(crops):
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            t = process_input(pil, self.image_cfg["image_height"],
                              self.image_cfg["image_min_width"], self.image_cfg["image_max_width"])
            tensors.append(t)
            buckets[t.shape[-1]].append(i)

        results: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        for indices in buckets.values():
            for k in range(0, len(indices), self.batch_size):
                chunk = indices[k:k + self.batch_size]
                batch = torch.cat([tensors[i] for i in chunk])
                seqs, probs = self._decode(batch)
                texts = self.vocab.batch_decode(seqs.tolist())
                for i, text, prob in zip(chunk, texts, probs):
                    results[i] = (text, float(prob))
        return results

    def _decode(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if self.use_cached_decode:
            return _greedy_decode_cached(self.model, batch)
        from vietocr.tool.translate import translate

        return translate(batch, self.model)


@torch.inference_mode()
def _greedy_decode_cached(model, img: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Greedy decode tương đương vietocr.tool.translate.translate nhưng có cache."""
    tf = model.transformer                      # LanguageTransformer
    decoder = tf.transformer.decoder            # nn.TransformerDecoder (post-norm)
    memory = tf.forward_encoder(model.cnn(img))
    batch_size = img.shape[0]
    max_len = min(MAX_SEQ_LEN, tf.pos_enc.pe.shape[0] - 1)

    # caches[i] = đầu vào của layer i tại mọi vị trí đã sinh (T, B, D)
    caches: list[torch.Tensor | None] = [None] * len(decoder.layers)
    token = torch.full((batch_size,), SOS, dtype=torch.long)
    tokens, probs = [token], [torch.ones(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool)

    for step in range(max_len + 1):
        x = tf.embed_tgt(token)[None] * math.sqrt(tf.d_model) + tf.pos_enc.pe[step:step + 1]
        for i, layer in enumerate(decoder.layers):
            caches[i] = x if caches[i] is None else torch.cat([caches[i], x])
            x = layer.norm1(x + layer.self_attn(x, caches[i], caches[i], need_weights=False)[0])
            x = layer.norm2(x + layer.multihead_attn(x, memory, memory, need_weights=False)[0])
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        if decoder.norm is not None:
            x = decoder.norm(x)

        prob, token = torch.softmax(tf.fc(x[0]), dim=-1).max(dim=-1)
        tokens.append(token)
        probs.append(prob)
        finished |= token == EOS
        if finished.all():
            break

    seqs = torch.stack(tokens, dim=1).numpy()
    char_probs = torch.stack(probs, dim=1).numpy() * (seqs > 3)  # bỏ token đặc biệt
    mean_probs = char_probs.sum(-1) / np.maximum((char_probs > 0).sum(-1), 1)
    return seqs, mean_probs


def ensure_weights(name: str, url: str, models_dir: Path) -> str:
    """Tải weights VietOCR một lần vào models_dir."""
    path = Path(models_dir) / f"vietocr_{name}.pth"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return str(path)
