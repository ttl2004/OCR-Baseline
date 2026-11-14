"""Gửi text OCR cho LLM (DeepSeek, API tương thích OpenAI) để tách thành các cột theo profile."""
import json
import time
from functools import lru_cache

from ..config import Settings
from ..profiles import Profile


class LlmError(RuntimeError):
    """Lỗi của riêng lần gọi này (JSON hỏng, output bị cắt...) -> ghi lỗi, xử lý tiếp ảnh khác."""


class LlmUnavailableError(LlmError):
    """LLM không dùng được (thiếu/sai key, mất mạng, hết tiền) -> dừng toàn bộ."""


@lru_cache(maxsize=4)
def _client(api_key: str, base_url: str, timeout: float):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)


def extract_fields(ocr_text: str, profile: Profile, system_prompt: str, settings: Settings) -> tuple[dict, dict]:
    """Gửi text OCR với prompt đã chọn. Trả về (các cột đã chuẩn hoá theo profile, thông tin lần gọi)."""
    import openai

    if not settings.deepseek_api_key:
        raise LlmUnavailableError("Chưa có DEEPSEEK_API_KEY trong file .env")
    if not ocr_text.strip():
        raise LlmError("Text OCR rỗng, không có gì để trích xuất")
    if not system_prompt.strip():
        raise LlmError("Prompt rỗng")

    client = _client(settings.deepseek_api_key, settings.deepseek_base_url, settings.llm_timeout)
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Văn bản OCR:\n"""\n{ocr_text}\n"""\n\nTrả về JSON.'},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=8000,
        )
    except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
        raise LlmUnavailableError(f"DeepSeek API key không hợp lệ hoặc không có quyền: {e}") from e
    except openai.APIConnectionError as e:
        raise LlmUnavailableError(f"Không kết nối được DeepSeek API: {e}") from e
    except openai.RateLimitError as e:
        raise LlmUnavailableError(f"DeepSeek API: vượt giới hạn hoặc hết số dư: {e}") from e
    except openai.APIStatusError as e:
        if e.status_code == 402:  # hết số dư
            raise LlmUnavailableError(f"DeepSeek API: tài khoản hết số dư: {e}") from e
        raise LlmError(f"DeepSeek API lỗi HTTP {e.status_code}: {e}") from e

    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise LlmError("Output của LLM bị cắt do quá dài (max_tokens)")

    content = choice.message.content or ""
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        raise LlmError(f"LLM trả về JSON không hợp lệ: {e}\n{content[:300]}") from e

    info = {
        "seconds": round(time.perf_counter() - start, 2),
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "model": settings.deepseek_model,
    }
    return profile.normalize(raw), info
