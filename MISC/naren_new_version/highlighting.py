import html
import re

ENTITY_STYLES = {
    "diagnosis":  {"bg": "#FFD06066", "border": "#D4650A", "label": "Diagnosis",  "text_color": "#B85408"},
    "medication": {"bg": "#FDECD480", "border": "#FF8C42", "label": "Medication", "text_color": "#C45A06"},
    "symptom":    {"bg": "#FFE8CC80", "border": "#E07020", "label": "Symptom",    "text_color": "#A84800"},
    "procedure":  {"bg": "#FFF3CC80", "border": "#D4A017", "label": "Procedure",  "text_color": "#8A6800"},
    "lab":        {"bg": "#FDE8D880", "border": "#C86428", "label": "Lab Value",  "text_color": "#963C10"},
}

_DEFAULT_STYLE = {"bg": "#88888833", "border": "#888888", "label": "Entity", "text_color": "#888888"}


def render_legend() -> str:
    items = []
    for style in ENTITY_STYLES.values():
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:16px;gap:6px;">'
            f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
            f'background:{style["bg"]};border:2px solid {style["border"]};"></span>'
            f'<span style="font-size:13px;color:{style["text_color"]};font-weight:600;">{style["label"]}</span>'
            f'</span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;padding:8px 0;margin-bottom:8px;">'
        + "".join(items)
        + "</div>"
    )


def render_highlighted_note(text: str, entities: list[dict]) -> str:
    if not text:
        return "<div>No note text available.</div>"

    # Filter to entities with valid offsets
    valid = [e for e in entities if e.get("start", 0) != e.get("end", 0)]
    # Sort by start asc; on tie, longer span first
    valid.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

    # Remove overlaps
    non_overlapping = []
    last_end = 0
    for e in valid:
        if e["start"] >= last_end:
            non_overlapping.append(e)
            last_end = e["end"]

    parts = []
    cursor = 0
    for e in non_overlapping:
        start, end = e["start"], e["end"]
        if start > len(text) or end > len(text):
            continue
        # Plain text before this entity
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        # Highlighted entity span
        style = ENTITY_STYLES.get(e.get("type", ""), _DEFAULT_STYLE)
        entity_text = html.escape(text[start:end])
        tooltip = f'{style["label"]}: {html.escape(e.get("normalized") or e.get("text", ""))}'
        parts.append(
            f'<mark title="{tooltip}" style="'
            f'background:{style["bg"]};'
            f'border-bottom:2px solid {style["border"]};'
            f'border-radius:3px;'
            f'padding:1px 2px;'
            f'cursor:help;'
            f'">{entity_text}</mark>'
        )
        cursor = end

    # Remaining plain text
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    inner = "".join(parts)
    return (
        '<div style="'
        "white-space:pre-wrap;"
        "font-family:'Courier New',Courier,monospace;"
        "font-size:13px;"
        "line-height:1.8;"
        "padding:20px 24px;"
        "background:#FFFDF7;"
        "color:#2C2416;"
        "border-radius:10px;"
        "border:1px solid #F5DEB3;"
        "overflow-y:auto;"
        '">'
        + inner
        + "</div>"
    )


def render_entity_summary(entities: list[dict]) -> str:
    from collections import Counter
    counts = Counter(e.get("type", "unknown") for e in entities)
    if not counts:
        return ""
    items = []
    for entity_type, count in sorted(counts.items()):
        style = ENTITY_STYLES.get(entity_type, _DEFAULT_STYLE)
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'background:{style["bg"]};border:1px solid {style["border"]};'
            f'border-radius:12px;padding:3px 10px;margin:2px;">'
            f'<span style="font-weight:600;color:{style["text_color"]}">{style["label"]}</span>'
            f'<span style="color:#ccc">×{count}</span>'
            f'</span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:8px 0;">'
        + "".join(items)
        + "</div>"
    )
