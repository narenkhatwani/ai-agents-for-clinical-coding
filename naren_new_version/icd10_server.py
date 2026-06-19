from __future__ import annotations

from collections import defaultdict
from fastmcp import FastMCP

try:
    import icd10
    _ICD10_AVAILABLE = True
except ImportError:
    _ICD10_AVAILABLE = False

mcp = FastMCP("icd10-coding-server", version="1.0.0")

# Inverted word index built at import time for fast search
_word_index: dict[str, list[str]] = defaultdict(list)
_code_data: dict[str, dict] = {}


def _build_index() -> None:
    if not _ICD10_AVAILABLE:
        return
    for raw_code, (billable, description) in icd10.codes.items():
        formatted = raw_code[:3] + ("." + raw_code[3:] if len(raw_code) > 3 else "")
        entry = {
            "code": formatted,
            "raw_code": raw_code,
            "description": description,
            "billable": billable,
        }
        _code_data[raw_code] = entry
        for word in description.lower().split():
            cleaned = word.strip("(),.-/")
            if len(cleaned) >= 3:
                _word_index[cleaned].append(raw_code)


_build_index()


@mcp.tool()
async def search_icd10_codes(query: str, max_results: int = 10) -> list[dict]:
    """Search ICD-10-CM codes by clinical term. Returns ranked matches."""
    if not _ICD10_AVAILABLE:
        return [{"error": "icd10-cm library not installed"}]

    query_words = [w.strip("(),.-/").lower() for w in query.split() if len(w.strip("(),.-/")) >= 3]
    if not query_words:
        return []

    score_map: dict[str, int] = defaultdict(int)
    for word in query_words:
        for raw_code in _word_index.get(word, []):
            score_map[raw_code] += 1

    # Secondary: partial match boost for longer query phrases
    query_lower = query.lower()
    for raw_code, entry in _code_data.items():
        if query_lower in entry["description"].lower():
            score_map[raw_code] += 3

    if not score_map:
        return []

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:max_results]
    return [
        {**_code_data[raw_code], "score": score}
        for raw_code, score in ranked
        if raw_code in _code_data
    ]


@mcp.tool()
async def get_icd10_description(code: str) -> dict:
    """Get full details for a specific ICD-10-CM code (e.g. 'J18.9' or 'J189')."""
    if not _ICD10_AVAILABLE:
        return {"error": "icd10-cm library not installed"}

    code_obj = icd10.find(code)
    if code_obj is None:
        return {"error": f"Code '{code}' not found", "code": code}

    result = {
        "code": str(code_obj),
        "description": code_obj.description,
        "billable": code_obj.billable,
    }
    # icd10-cm exposes optional hierarchy fields depending on version
    for attr in ("chapter", "block", "block_description"):
        val = getattr(code_obj, attr, None)
        if val is not None:
            result[attr] = str(val)
    return result
