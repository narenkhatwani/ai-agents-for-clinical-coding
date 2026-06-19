from __future__ import annotations

import json
import logging
import re

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3-vl:8b"
MAX_NOTE_CHARS = 12_000

_VALID_TYPES = {"diagnosis", "medication", "symptom", "procedure", "lab"}

_TYPE_ALIASES = {
    "disease": "diagnosis",
    "condition": "diagnosis",
    "disorder": "diagnosis",
    "dx": "diagnosis",
    "drug": "medication",
    "medicine": "medication",
    "med": "medication",
    "sign": "symptom",
    "finding": "symptom",
    "complaint": "symptom",
    "lab_test": "lab",
    "laboratory": "lab",
    "vital": "lab",
    "test": "procedure",
    "imaging": "procedure",
}

_SYSTEM_PROMPT = """\
You are a clinical NLP system. The user will give you a clinical note. \
Extract every medical entity mentioned in it.

Return ONLY a JSON object — no explanation, no markdown, no extra text:
{
  "entities": [
    {
      "text": "<exact substring copied from the note>",
      "type": "<diagnosis | medication | symptom | procedure | lab>",
      "start": <0-indexed char offset, or 0 if unknown>,
      "end": <0-indexed end offset, or 0 if unknown>,
      "normalized": "<standard medical term, or empty string>"
    }
  ]
}

Type definitions:
- diagnosis: diseases, conditions, disorders (e.g. "sepsis", "COPD", "atrial fibrillation")
- medication: drugs, doses, routes (e.g. "metoprolol 25mg PO", "heparin drip")
- symptom: patient complaints or clinical signs (e.g. "shortness of breath", "diaphoresis")
- procedure: tests, imaging, surgeries (e.g. "CT chest", "intubation", "CABG")
- lab: lab results with values/units (e.g. "WBC 14.2", "creatinine 2.1 mg/dL")

Extract as many entities as are present. Copy the exact text from the note for the "text" field.\
"""

_COMPACT_SYSTEM_PROMPT = """\
You are a clinical entity extractor. Extract all diagnoses, medications, symptoms, procedures, \
and lab values from the clinical note. Output ONLY valid JSON with this structure — \
no other text whatsoever:
{"entities":[{"text":"<exact phrase from the note>","type":"diagnosis|medication|symptom|procedure|lab","start":0,"end":0,"normalized":""}]}
List every entity you find. The "text" value must be copied verbatim from the note.\
"""

# Placeholder strings that indicate the model echoed the prompt example rather than extracting.
_PROMPT_PLACEHOLDERS = {
    "exact note substring",
    "exact phrase from note",
    "<exact substring copied from the note>",
    "<exact phrase from the note>",
}


class OllamaConnectionError(RuntimeError):
    pass


class OllamaModelError(RuntimeError):
    pass


class OllamaTimeoutError(RuntimeError):
    pass


def _preferred_qwen_model(models: list[str]) -> str | None:
    """Pick the best available Qwen model for clinical text extraction."""
    preferred_prefixes = (
        "qwen3-vl:8b",
        "qwen2.5:",
        "qwen3:",
        "qwen:",
    )
    lowered = {m.lower(): m for m in models}
    for prefix in preferred_prefixes:
        match = next((lowered[m] for m in lowered if m.startswith(prefix)), None)
        if match:
            return match
    return None


def check_ollama_health() -> tuple[bool, str]:
    """Returns (is_healthy, message). Checks if Ollama is running and model is available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, f"Ollama returned HTTP {resp.status_code}"
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        match = _preferred_qwen_model(models)
        if match:
            return True, match
        if models:
            return False, (
                "No supported Qwen model found. "
                f"Available: {', '.join(models[:5])}. "
                "Run: ollama pull qwen3-vl:8b"
            )
        return False, "No models found. Run: ollama pull qwen3-vl:8b"
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return False, str(e)


def _detect_model() -> str:
    """Return the best available qwen model name."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            match = _preferred_qwen_model(models)
            if match:
                return match
    except Exception:
        pass
    return OLLAMA_MODEL


def _normalize_entity_type(raw_type: str) -> str:
    entity_type = (raw_type or "diagnosis").strip().lower()
    entity_type = _TYPE_ALIASES.get(entity_type, entity_type)
    if entity_type not in _VALID_TYPES:
        entity_type = "diagnosis"
    return entity_type


def _strip_response_wrappers(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    # Remove entire think blocks including their content, not just the tags
    cleaned = re.sub(
        r"<(?:think|redacted_thinking)>.*?</(?:think|redacted_thinking)>",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Strip any remaining orphaned tags
    cleaned = re.sub(r"</?(?:think|redacted_thinking)>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _try_parse_entities_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("entities"), list):
        return data
    return None


def _extract_json_payload(text: str) -> dict | None:
    cleaned = _strip_response_wrappers(text)
    if not cleaned:
        return None

    direct = _try_parse_entities_json(cleaned)
    if direct is not None:
        return direct

    # Qwen3 may bury the JSON at the end of a long thinking trace.
    best: dict | None = None
    for match in re.finditer(r"\{", cleaned):
        start = match.start()
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    if '"entities"' in candidate:
                        parsed = _try_parse_entities_json(candidate)
                        if parsed is not None:
                            best = parsed
                    break
    return best


def _truncate_note(text: str, max_chars: int = MAX_NOTE_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cutoff = int(max_chars * 0.9)
    last_period = text.rfind(".", cutoff, max_chars)
    cut = last_period + 1 if last_period != -1 else max_chars
    return text[:cut].strip(), True


_JSON_PREFILL = '{"entities":['


def _call_ollama(note_text: str, model: str, system_prompt: str = _SYSTEM_PROMPT) -> str:
    # format:"json" causes empty content on qwen3-vl regardless of think setting.
    # Assistant-prefill forces the model to continue directly from '{"entities":['
    # so it can never output reasoning prose before the JSON.
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Clinical note:\n\n{note_text}"},
            {"role": "assistant", "content": _JSON_PREFILL},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 4096,
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=300,
        )
    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError("Cannot connect to Ollama. Is 'ollama serve' running?") from e
    except requests.exceptions.Timeout as e:
        raise OllamaTimeoutError("Ollama request timed out after 300s. Try a shorter note or re-extract.") from e

    if resp.status_code == 404:
        raise OllamaModelError(f"Model '{model}' not found. Run: ollama pull {model}")
    resp.raise_for_status()

    message = (resp.json().get("message") or {})
    # Reconstruct full JSON by prepending the prefill we injected
    continuation = (message.get("content") or "").strip()
    content = _JSON_PREFILL + continuation

    logger.debug("Ollama content preview: %s", content[:300].replace("\n", " "))

    return content


def _repair_offsets(entities: list[dict], note_text: str) -> list[dict]:
    repaired = []
    for e in entities:
        text_val = e.get("text", "")
        start = e.get("start", 0)
        end = e.get("end", 0)
        recovered = False

        if text_val and 0 <= start < end <= len(note_text):
            if note_text[start:end].lower() == text_val.lower():
                # Offsets are correct
                pass
            else:
                # Try to find the actual position
                m = re.search(re.escape(text_val), note_text, re.IGNORECASE)
                if m:
                    start, end = m.start(), m.end()
                    recovered = True
                else:
                    start = end = 0
                    recovered = True
        elif text_val:
            m = re.search(re.escape(text_val), note_text, re.IGNORECASE)
            if m:
                start, end = m.start(), m.end()
                recovered = True
            else:
                start = end = 0
                recovered = True

        entity_type = _normalize_entity_type(e.get("type", "diagnosis"))

        repaired.append({
            "text": text_val,
            "type": entity_type,
            "start": start,
            "end": end,
            "normalized": e.get("normalized", ""),
            "offset_recovered": recovered,
        })
    return repaired


def extract_entities(note_text: str) -> tuple[list[dict], str]:
    """Extract clinical entities from a note using Ollama/Qwen.

    Returns (entities, debug_info) where debug_info is a short diagnostic string.
    """
    if not note_text or not note_text.strip():
        return [], "Note was empty."

    truncated, was_truncated = _truncate_note(note_text)
    if was_truncated:
        logger.info("Note truncated to %d chars for LLM processing.", len(truncated))

    model = _detect_model()
    raw_response = ""
    data: dict | None = None
    try:
        raw_response = _call_ollama(truncated, model)
        data = _extract_json_payload(raw_response)
        if data is None:
            logger.info("Retrying entity extraction with compact prompt for %s.", model)
            raw_response = _call_ollama(truncated, model, system_prompt=_COMPACT_SYSTEM_PROMPT)
            data = _extract_json_payload(raw_response)
    except OllamaConnectionError:
        raise
    except OllamaModelError:
        raise
    except OllamaTimeoutError:
        raise
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return [], f"LLM call failed: {e}"

    if data is None:
        preview = raw_response[:200].replace("\n", " ") if raw_response else "(empty response)"
        logger.error("Failed to parse LLM JSON response. Preview: %s", preview)
        return [], f"Could not parse JSON from model response. Raw preview: {preview}"

    raw_entities = data.get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = []

    repaired = _repair_offsets(raw_entities, truncated)
    entities = [
        e for e in repaired
        if e.get("text", "").strip() and e["text"].strip() not in _PROMPT_PLACEHOLDERS
    ]
    debug = f"Model {model} returned {len(raw_entities)} raw entities, {len(entities)} valid."
    return entities, debug
