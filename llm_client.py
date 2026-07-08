"""
Unified LLM client for clinical pipeline agents.

Supports:
  - ollama  — local Ollama server
  - api     — OpenAI-compatible chat API (OpenAI, Groq, Together, Azure, etc.)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import requests

Provider = Literal["ollama", "api"]


@dataclass
class LLMConfig:
    provider: Provider = "ollama"
    model: str = "qwen2.5:7b"
    # Ollama
    base_url: str = "http://localhost:11434"
    # API (OpenAI-compatible)
    api_base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    api_extra_headers: Dict[str, str] = field(default_factory=dict)
    api_label: str = "api"  # used in logs and extraction_method tags
    # Shared
    max_note_chars: int = 4000
    timeout_seconds: int = 600
    max_retries: int = 2
    num_predict: int = 4096  # Ollama
    max_tokens: int = 4096  # API
    temperature: float = 0.1
    json_mode: bool = True  # request JSON object from API when supported

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env) or os.environ.get("LLM_API_KEY")

    def method_prefix(self) -> str:
        if self.provider == "ollama":
            return "ollama"
        return self.api_label or "api"


# Backward-compatible alias used by notebooks / older code
OllamaConfig = LLMConfig


class LLMNotAvailableError(RuntimeError):
    pass


class OllamaNotAvailableError(LLMNotAvailableError):
    pass


def _strip_json_wrappers(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(
        r"<(?:think|redacted_thinking)>.*?</(?:think|redacted_thinking)>",
        "",
        cleaned,
        flags=re.I | re.DOTALL,
    )
    return cleaned.strip()


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_json_wrappers(text)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def truncate_note(note: str, max_chars: int) -> str:
    note = note or ""
    if len(note) <= max_chars:
        return note
    return note[:max_chars] + "\n\n[... note truncated for LLM input ...]"


def check_llm(config: LLMConfig, strict: bool = True) -> Tuple[bool, str]:
    if config.provider == "ollama":
        return _check_ollama(config, strict=strict)
    return _check_api(config)


def require_llm(config: LLMConfig) -> str:
    ok, message = check_llm(config)
    if not ok:
        raise LLMNotAvailableError(message)
    return message


def check_ollama(config: LLMConfig, strict: bool = True) -> Tuple[bool, str]:
    return _check_ollama(config, strict=strict)


def _check_ollama(config: LLMConfig, strict: bool = True) -> Tuple[bool, str]:
    try:
        response = requests.get(f"{config.base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        if config.model in models:
            return True, config.model
        if not strict and models:
            return True, models[0]
        if not models:
            return False, "No models installed. Run: ollama pull qwen2.5:7b"
        return False, (
            f"Model '{config.model}' not found. Installed: {', '.join(models[:5])}. "
            f"Run: ollama pull {config.model}"
        )
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Start with: ollama serve"
    except Exception as exc:
        return False, str(exc)


def _check_api(config: LLMConfig) -> Tuple[bool, str]:
    key = config.resolved_api_key()
    if not key:
        return False, (
            f"API key not set. Export {config.api_key_env} or LLM_API_KEY, "
            "or set API_KEY in pipeline_config.py"
        )
    if not config.model:
        return False, "API_MODEL is not set in pipeline_config.py"
    return True, config.model


def warn_if_slow_model(model: str, provider: Provider) -> None:
    if provider != "ollama":
        return
    lower = model.lower()
    if "vl" in lower or "vision" in lower:
        print(
            f"WARNING: '{model}' is a vision model and is slow for text-only NLP. "
            "Prefer a text model, e.g. ollama pull qwen2.5:7b"
        )


def call_llm_chat(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: Optional[str] = None,
    json_mode: Optional[bool] = None,
) -> str:
    use_model = model or config.model
    use_json = config.json_mode if json_mode is None else json_mode

    if config.provider == "ollama":
        return _call_ollama_chat(system_prompt, user_prompt, config, use_model)
    return _call_api_chat(system_prompt, user_prompt, config, use_model, json_mode=use_json)


def _call_ollama_chat(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: str,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
        },
    }
    url = f"{config.base_url.rstrip('/')}/api/chat"
    label = "Ollama"

    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=config.timeout_seconds)
        except requests.exceptions.ConnectionError as exc:
            raise LLMNotAvailableError(
                "Cannot connect to Ollama. Is 'ollama serve' running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            last_error = _timeout_error(config, label, attempt)
            if attempt < config.max_retries:
                _sleep_retry(attempt)
                continue
            raise last_error from exc

        if response.status_code == 404:
            raise LLMNotAvailableError(f"Model '{model}' not found. Run: ollama pull {model}")
        response.raise_for_status()
        return (response.json().get("message") or {}).get("content", "")

    raise last_error or TimeoutError(f"{label} request failed")


def _call_api_chat(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: str,
    json_mode: bool,
) -> str:
    api_key = config.resolved_api_key()
    if not api_key:
        raise LLMNotAvailableError(
            f"API key not set. Export {config.api_key_env} or LLM_API_KEY."
        )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    url = f"{config.api_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **config.api_extra_headers,
    }
    label = config.api_label or "API LLM"

    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=config.timeout_seconds
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMNotAvailableError(
                f"Cannot connect to API at {config.api_base_url}: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            last_error = _timeout_error(config, label, attempt)
            if attempt < config.max_retries:
                _sleep_retry(attempt)
                continue
            raise last_error from exc

        if response.status_code == 401:
            raise LLMNotAvailableError("API authentication failed (401). Check your API key.")
        if response.status_code == 404:
            raise LLMNotAvailableError(
                f"Model '{model}' not found at {config.api_base_url}"
            )
        if not response.ok:
            raise LLMNotAvailableError(
                f"API error {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Empty API response: {json.dumps(data)[:300]}")
        return (choices[0].get("message") or {}).get("content", "")

    raise last_error or TimeoutError(f"{label} request failed")


def _timeout_error(config: LLMConfig, label: str, attempt: int) -> TimeoutError:
    return TimeoutError(
        f"{label} request timed out after {config.timeout_seconds}s "
        f"(attempt {attempt + 1}/{config.max_retries + 1})"
    )


def _sleep_retry(attempt: int) -> None:
    wait = 5 * (attempt + 1)
    print(f"  Timeout — retrying in {wait}s...")
    time.sleep(wait)


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    raw = call_llm_chat(system_prompt, user_prompt, config, model=model, json_mode=True)
    parsed = parse_json_object(raw)
    if parsed is None:
        raise ValueError(
            f"Could not parse JSON from {config.provider} LLM. Preview: {raw[:300]}"
        )
    return parsed
