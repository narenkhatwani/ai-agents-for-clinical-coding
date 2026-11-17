"""
LLM-only DiffDx + ICD baseline (no ontology / SNOMED / symptom tree).

Reads redacted clinical notes + history from patient_records/,
writes ALL outputs under llm_only_baseline/ only — never merges into
patient_records/accuracy or data/stage_10_evaluation.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pipeline import (
    DIFF_DX_CONSTRAINTS,
    DIFF_DX_OUTPUT_SCHEMA,
    DIFF_DX_SYSTEM_PROMPT,
    DIFF_DX_TEMPERATURE,
    EXPORT_DIR,
    LLMConfig,
    LLM_REQUEST_DELAY_SECONDS,
    REPO_ROOT,
    _clip_clinical_context,
    _diagnosis_match_from_llm,
    _dot_icd_code,
    _format_diffdx_list_for_export,
    _heuristic_match_diffdx_rank,
    _item_at_rank,
    _load_gt,
    _score_linked_icd,
    _write_json,
    _write_text,
    _yn_export,
    call_llm_json,
    format_accuracy_comparison_txt,
    format_admission_history_text,
    format_cohort_metrics_txt,
    get_llm_config,
    load_discharge_diagnosis_gt,
    load_llm_qe_cache,
    llm_clinical_query_expansion,
    require_llm,
    save_llm_qe_cache,
    summarize_stage10,
    truncate_note,
    warn_if_slow_model,
)

# ---------------------------------------------------------------------------
# Isolated output root (never patient_records or stage_10_evaluation)
# ---------------------------------------------------------------------------
LLM_ONLY_DIR = REPO_ROOT / "llm_only_baseline"
LLM_ONLY_CHECKPOINT = LLM_ONLY_DIR / "checkpoint.json"
LLM_ONLY_SUMMARY = LLM_ONLY_DIR / "accuracy_summary.json"
LLM_ONLY_COHORT_TXT = LLM_ONLY_DIR / "cohort_accuracy.txt"
LLM_ONLY_DIFFDX_CSV = LLM_ONLY_DIR / "diffdx_match.csv"
LLM_ONLY_ICD_CSV = LLM_ONLY_DIR / "icd_match.csv"
LLM_ONLY_QE_CACHE = LLM_ONLY_DIR / "llm_qe_cache.json"
LLM_ONLY_README = LLM_ONLY_DIR / "README.txt"

LLM_ONLY_DIFF_DX_ROLE = """\
You are an experienced hospital clinician. You formulate working differential \
diagnoses for the CURRENT admission from the clinical note and prior-admission \
history only. You do NOT use SNOMED ontology graphs or pre-built symptom trees.

You support clinical documentation: rank principal disease for THIS stay, plus \
competing differentials and relevant comorbidities when evidence supports them."""

LLM_ONLY_DIFF_DX_TASK = """\
Using only the CONTEXT below (redacted current note + structured context + prior \
admission history), produce a SCORED differential diagnosis for the CURRENT admission.

For each candidate:
1. Likelihood score 0–100 (relative ranking; need not sum to 100).
2. Confidence: high | medium | low.
3. Category: primary | secondary | complication | risk_factor | differential.
4. Short supporting / opposing evidence from the note or prior history.
5. Brief reasoning.

Diagnosis naming:
- Use FULL clinical condition phrases (typically 3–12 words).
- Rank-1 must be a billable principal DISEASE/CONDITION, not a symptom or device event.
- Prefer diagnoses that explain the CURRENT presentation; prior history is PMH context only.

Also provide summary, most_likely, rule_outs, uncertain_areas.
Aim for 5–8 differentials when evidence allows.

Return COMPLETE valid JSON only matching the OUTPUT schema."""

LLM_ONLY_ICD_SYSTEM = """\
You are a clinical coding specialist. Assign ICD-10-CM codes to ranked differential \
diagnoses using only the clinical context provided. Return valid JSON only."""

LLM_ONLY_ICD_SCHEMA = """\
{
  "differential": [
    {
      "rank": 1,
      "diagnosis": "same diagnosis string as input",
      "icd10_primary": "A00.0",
      "icd10_primary_title": "official or short title",
      "icd10_supporting": [{"code": "E11.9", "title": "..."}],
      "coding_reason": "≤25 words"
    }
  ],
  "primary_icd_code": "A00.0",
  "primary_icd_title": "...",
  "primary_condition": "rank-1 diagnosis or refined name"
}"""


def _adm_out_dir(patient_id: str, hadm_id: str) -> Path:
    return LLM_ONLY_DIR / f"patient_{patient_id}" / f"hadm_{hadm_id}"


def _iter_source_admissions(export_dir: Path) -> List[Path]:
    """Source admissions from the ontology pipeline export (inputs only)."""
    root = Path(export_dir)
    out: List[Path] = []
    for patient_dir in sorted(root.glob("patient_*")):
        adm_root = patient_dir / "admissions"
        if not adm_root.is_dir():
            continue
        for adm_dir in sorted(adm_root.glob("hadm_*")):
            if (adm_dir / "clinical_note.txt").exists() and (adm_dir / "ground_truth.json").exists():
                out.append(adm_dir)
    return out


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_llm_only_diff_dx_context(
    patient_id: str,
    hadm_id: str,
    clinical_note: str,
    clinical_context_text: Optional[str],
    admission_history: Optional[List[Dict[str, Any]]],
) -> str:
    parts: List[str] = [
        f"Patient ID: {patient_id}",
        f"Current admission ID (HADM): {hadm_id}",
        "",
        "LLM-ONLY baseline: redacted current note + optional vitals/labs/radiology "
        "+ full prior-admission history. No SNOMED / symptom-tree / ontology routing.",
        "CURRENT discharge diagnosis / ICD labels are redacted from the note — do not invent them.",
        "",
        "--- PRIOR ADMISSION HISTORY ---",
        format_admission_history_text(admission_history or []),
    ]
    if clinical_context_text:
        parts.extend(
            [
                "",
                "--- STRUCTURED CLINICAL CONTEXT (current stay vitals/labs/radiology) ---",
                _clip_clinical_context(clinical_context_text),
            ]
        )
    parts.extend(
        [
            "",
            "--- CURRENT ADMISSION CLINICAL NOTE (diagnosis sections redacted) ---",
            clinical_note.strip() or "(empty note)",
        ]
    )
    return "\n".join(parts)


def _normalize_diffdx_result(
    result: Dict[str, Any],
    patient_id: str,
    hadm_id: str,
    config: LLMConfig,
) -> Dict[str, Any]:
    differentials = result.get("differential") or result.get("differentials") or []
    if not isinstance(differentials, list):
        differentials = []
    cleaned: List[Dict[str, Any]] = []
    for i, item in enumerate(differentials):
        if not isinstance(item, dict):
            continue
        score = item.get("score", item.get("confidence_score", 0))
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        if 0.0 < score_f <= 1.0:
            score_f *= 100.0
        score_f = max(0.0, min(100.0, score_f))
        dx = str(item.get("diagnosis") or item.get("name") or "").strip()
        if not dx:
            continue
        cleaned.append(
            {
                "rank": int(item.get("rank") or i + 1),
                "diagnosis": dx,
                "score": round(score_f, 1),
                "confidence": str(item.get("confidence") or "medium").lower(),
                "category": str(item.get("category") or "differential"),
                "supporting_evidence": item.get("supporting_evidence") or [],
                "opposing_evidence": item.get("opposing_evidence") or [],
                "reasoning": item.get("reasoning") or "",
            }
        )
    cleaned.sort(key=lambda x: (-x["score"], x["rank"]))
    for i, c in enumerate(cleaned, start=1):
        c["rank"] = i

    result["differential"] = cleaned
    result["n_candidates"] = len(cleaned)
    if cleaned and not result.get("most_likely"):
        result["most_likely"] = cleaned[0]["diagnosis"]
    result["type"] = "llm_only_differential_diagnosis"
    result["_method"] = f"{config.method_prefix()}_llm:{config.model}"
    result["_agent"] = "llm_only_diff_dx"
    result["_temperature"] = config.temperature
    result["patient_id"] = str(patient_id)
    result["hadm_id"] = str(hadm_id)
    result["generated_at"] = datetime.now().isoformat()
    result["pipeline"] = "llm_only_baseline"
    result["inputs"] = {
        "clinical_note_redacted": True,
        "admission_history": True,
        "clinical_context": True,
        "snomed": False,
        "symptom_tree": False,
        "ontology_routing": False,
        "temperature": config.temperature,
    }
    return result


def format_llm_only_diffdx_txt(result: Dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "LLM-ONLY DIFFERENTIAL DIAGNOSIS (no ontology)",
        "=" * 60,
        f"Patient ID       : {result.get('patient_id', 'N/A')}",
        f"HADM ID          : {result.get('hadm_id', 'N/A')}",
        f"Method           : {result.get('_method', 'unknown')}",
        f"Temperature      : {result.get('_temperature', 'N/A')}",
        f"Generated        : {result.get('generated_at', 'N/A')}",
        f"Candidates       : {result.get('n_candidates', 0)}",
        f"Most likely      : {result.get('most_likely', 'N/A')}",
        "",
    ]
    if result.get("summary"):
        lines.extend(["Summary:", f"  {result['summary']}", ""])
    lines.append("-" * 40)
    lines.append("RANKED DIFFERENTIAL")
    lines.append("-" * 40)
    for item in result.get("differential") or []:
        lines.append(
            f"  #{item.get('rank')}  [{item.get('score')}/100 | {item.get('confidence')}]  "
            f"{item.get('diagnosis')}  ({item.get('category')})"
        )
        for e in item.get("supporting_evidence") or []:
            lines.append(f"      + {e}")
        for e in item.get("opposing_evidence") or []:
            lines.append(f"      - {e}")
        if item.get("reasoning"):
            lines.append(f"      why: {item['reasoning']}")
    lines.append("")
    return "\n".join(lines)


def run_llm_only_diff_dx(
    patient_id: str,
    hadm_id: str,
    clinical_note: str,
    clinical_context_text: Optional[str] = None,
    admission_history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[LLMConfig] = None,
    temperature: float = DIFF_DX_TEMPERATURE,
) -> Dict[str, Any]:
    from dataclasses import replace

    base = config or get_llm_config()
    config = replace(base, temperature=float(temperature))
    require_llm(config)
    warn_if_slow_model(config.model, config.provider)

    note = truncate_note(clinical_note, config.max_note_chars)
    context = build_llm_only_diff_dx_context(
        patient_id=str(patient_id),
        hadm_id=str(hadm_id),
        clinical_note=note,
        clinical_context_text=clinical_context_text,
        admission_history=admission_history,
    )
    user_prompt = (
        f"ROLE:\n{LLM_ONLY_DIFF_DX_ROLE}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"TASK:\n{LLM_ONLY_DIFF_DX_TASK}\n\n"
        f"CONSTRAINTS:\n{DIFF_DX_CONSTRAINTS}\n\n"
        f"OUTPUT SCHEMA (return only this JSON object):\n{DIFF_DX_OUTPUT_SCHEMA}"
    )
    raw = call_llm_json(DIFF_DX_SYSTEM_PROMPT, user_prompt, config, model=config.model)
    return _normalize_diffdx_result(raw, patient_id, hadm_id, config)


def run_llm_only_icd_coding(
    diffdx: Dict[str, Any],
    clinical_note: str,
    clinical_context_text: Optional[str] = None,
    admission_history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """Map each DiffDx rank to ICD-10-CM via LLM (no SNOMED ExtendedMap)."""
    from dataclasses import replace

    base = config or get_llm_config()
    config = replace(base, temperature=float(DIFF_DX_TEMPERATURE))
    require_llm(config)

    diffs = diffdx.get("differential") or []
    slim = [
        {
            "rank": d.get("rank"),
            "diagnosis": d.get("diagnosis"),
            "score": d.get("score"),
            "confidence": d.get("confidence"),
            "category": d.get("category"),
        }
        for d in diffs
    ]
    note = truncate_note(clinical_note, min(config.max_note_chars, 12000))
    hist = format_admission_history_text(admission_history or [])
    ctx = _clip_clinical_context(clinical_context_text) if clinical_context_text else ""

    user_prompt = f"""\
Assign ICD-10-CM codes for each ranked differential below for the CURRENT admission.

RULES:
- Provide icd10_primary (dotted preferred) + title for every rank.
- Optionally add 0–3 supporting comorbidity codes.
- Prefer specific codes when the note supports them; otherwise use appropriate unspecified codes.
- Do NOT invent clinical facts. Prior admissions are PMH only.
- Keep coding_reason ≤25 words.
- primary_icd_code should match rank-1's principal code.

DIFFERENTIAL:
{json.dumps(slim, indent=2, ensure_ascii=False)}

PRIOR HISTORY (PMH context):
{hist[:8000]}

STRUCTURED CONTEXT:
{ctx[:6000] if ctx else "(none)"}

CURRENT NOTE (redacted; excerpt):
{note[:8000]}

Return JSON matching this schema (no markdown):
{LLM_ONLY_ICD_SCHEMA}
"""
    raw = call_llm_json(LLM_ONLY_ICD_SYSTEM, user_prompt, config, model=config.model)

    by_rank: Dict[int, Dict[str, Any]] = {}
    for item in raw.get("differential") or []:
        if not isinstance(item, dict):
            continue
        try:
            rank = int(item.get("rank") or 0)
        except (TypeError, ValueError):
            continue
        if rank:
            by_rank[rank] = item

    merged: List[Dict[str, Any]] = []
    for d in diffs:
        rank = int(d.get("rank") or 0)
        mapped = by_rank.get(rank) or {}
        code = _dot_icd_code(str(mapped.get("icd10_primary") or ""))
        title = str(mapped.get("icd10_primary_title") or "")
        supporting = mapped.get("icd10_supporting") or []
        if not isinstance(supporting, list):
            supporting = []
        package = []
        if code:
            package.append({"code": code, "title": title, "role": "principal"})
        for s in supporting:
            if not isinstance(s, dict):
                continue
            sc = _dot_icd_code(str(s.get("code") or ""))
            if sc:
                package.append(
                    {
                        "code": sc,
                        "title": str(s.get("title") or ""),
                        "role": "supporting",
                    }
                )
        row = {
            **d,
            "icd10_primary": code or None,
            "icd10_primary_title": title,
            "icd10_package": package,
            "icd10_supporting": [p for p in package if p.get("role") == "supporting"],
            "coding_reason": mapped.get("coding_reason") or "",
            "coding_method": "llm_only",
        }
        merged.append(row)

    primary_code = _dot_icd_code(str(raw.get("primary_icd_code") or "")) or (
        merged[0].get("icd10_primary") if merged else None
    )
    primary_title = str(raw.get("primary_icd_title") or "") or (
        merged[0].get("icd10_primary_title") if merged else ""
    )

    return {
        "type": "llm_only_icd_coding",
        "pipeline": "llm_only_baseline",
        "patient_id": diffdx.get("patient_id"),
        "hadm_id": diffdx.get("hadm_id"),
        "most_likely": diffdx.get("most_likely"),
        "summary": diffdx.get("summary"),
        "differential": merged,
        "primary_icd_code": primary_code,
        "primary_icd_title": primary_title,
        "primary_condition": raw.get("primary_condition") or diffdx.get("most_likely"),
        "_method": f"{config.method_prefix()}_llm:{config.model}",
        "_temperature": config.temperature,
        "generated_at": datetime.now().isoformat(),
        "ontology_routing": False,
    }


def format_llm_only_icd_txt(result: Dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "LLM-ONLY ICD CODING (no SNOMED map)",
        "=" * 60,
        f"Patient ID       : {result.get('patient_id', 'N/A')}",
        f"HADM ID          : {result.get('hadm_id', 'N/A')}",
        f"Method           : {result.get('_method', 'unknown')}",
        f"Primary ICD      : {result.get('primary_icd_code') or '-'} — {result.get('primary_icd_title') or ''}",
        f"Primary condition: {result.get('primary_condition') or '-'}",
        "",
    ]
    for d in result.get("differential") or []:
        lines.append(
            f"  #{d.get('rank')}  {d.get('diagnosis')}  →  "
            f"{d.get('icd10_primary') or '-'} — {d.get('icd10_primary_title') or ''}"
        )
        if d.get("coding_reason"):
            lines.append(f"      reason: {d['coding_reason']}")
    lines.append("")
    return "\n".join(lines)


def _load_checkpoint() -> Dict[str, Any]:
    if not LLM_ONLY_CHECKPOINT.exists():
        return {"completed": {}}
    try:
        data = json.loads(LLM_ONLY_CHECKPOINT.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"completed": {}}
        data.setdefault("completed", {})
        return data
    except (json.JSONDecodeError, OSError):
        return {"completed": {}}


def _save_checkpoint(data: Dict[str, Any]) -> None:
    LLM_ONLY_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(LLM_ONLY_CHECKPOINT, data)


def ensure_llm_only_readme() -> None:
    LLM_ONLY_DIR.mkdir(parents=True, exist_ok=True)
    if LLM_ONLY_README.exists():
        return
    _write_text(
        LLM_ONLY_README,
        """LLM-ONLY BASELINE (isolated)
============================

This folder is completely separate from patient_records/ and data/stage_*.

Pipeline:
  1. DiffDx from redacted clinical note + clinical_context + full prior history
  2. ICD-10 codes assigned by LLM (no SNOMED / ontology routing)
  3. Accuracy: DiffDx vs physician Discharge Diagnosis; ICD vs billed primary ICD

Nothing here is written back into the ontology pipeline exports.
""",
    )


def run_llm_only_admission(
    source_adm_dir: Path,
    config: Optional[LLMConfig] = None,
    temperature: float = DIFF_DX_TEMPERATURE,
    skip_if_done: bool = True,
) -> Dict[str, Any]:
    """Run DiffDx + ICD for one admission; write only under llm_only_baseline/."""
    source_adm_dir = Path(source_adm_dir)
    gt = _load_gt(source_adm_dir)
    pid = str(gt.get("patient_id") or source_adm_dir.parent.parent.name.replace("patient_", ""))
    hid = str(gt.get("hadm_id") or source_adm_dir.name.replace("hadm_", ""))
    key = f"{pid}/{hid}"

    ckpt = _load_checkpoint()
    out_dir = _adm_out_dir(pid, hid)
    if skip_if_done and key in (ckpt.get("completed") or {}) and (out_dir / "icd_coding.json").exists():
        return {
            "patient_id": pid,
            "hadm_id": hid,
            "skipped": True,
            "out_dir": str(out_dir),
        }

    note = (source_adm_dir / "clinical_note.txt").read_text(encoding="utf-8", errors="replace")
    ctx_path = source_adm_dir / "clinical_context.txt"
    ctx = ctx_path.read_text(encoding="utf-8", errors="replace") if ctx_path.exists() else ""
    hist_path = source_adm_dir.parent.parent / "admission_history.json"
    history = _load_json(hist_path) if hist_path.exists() else []
    if not isinstance(history, list):
        history = []

    cfg = config or get_llm_config()
    diffdx = run_llm_only_diff_dx(
        patient_id=pid,
        hadm_id=hid,
        clinical_note=note,
        clinical_context_text=ctx,
        admission_history=history,
        config=cfg,
        temperature=temperature,
    )
    time.sleep(float(LLM_REQUEST_DELAY_SECONDS or 0))
    icd = run_llm_only_icd_coding(
        diffdx=diffdx,
        clinical_note=note,
        clinical_context_text=ctx,
        admission_history=history,
        config=cfg,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "differential_diagnosis.json", diffdx)
    _write_text(out_dir / "differential_diagnosis.txt", format_llm_only_diffdx_txt(diffdx))
    _write_json(out_dir / "icd_coding.json", icd)
    _write_text(out_dir / "icd_coding.txt", format_llm_only_icd_txt(icd))
    # Pointers only — do not copy ontology artifacts
    _write_json(
        out_dir / "source_paths.json",
        {
            "source_admission_dir": str(source_adm_dir),
            "clinical_note": str(source_adm_dir / "clinical_note.txt"),
            "ground_truth": str(source_adm_dir / "ground_truth.json"),
            "note": "Outputs in this folder are LLM-only; ontology pipeline untouched.",
        },
    )

    ckpt.setdefault("completed", {})[key] = {
        "finished_at": datetime.now().isoformat(),
        "out_dir": str(out_dir),
        "model": cfg.model,
    }
    _save_checkpoint(ckpt)
    return {"patient_id": pid, "hadm_id": hid, "skipped": False, "out_dir": str(out_dir), "icd": icd}


def _diagnosis_items_from_llm_only_icd(icd: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for d in icd.get("differential") or []:
        items.append(
            {
                "rank": d.get("rank"),
                "diagnosis": d.get("diagnosis"),
                "score": d.get("score"),
                "confidence": d.get("confidence"),
                "icd10_primary": d.get("icd10_primary"),
                "icd10_primary_title": d.get("icd10_primary_title") or "",
            }
        )
    return items


def evaluate_llm_only_admission(
    out_adm_dir: Path,
    source_adm_dir: Path,
    llm_config: Optional[LLMConfig] = None,
    llm_qe_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score LLM-only DiffDx/ICD; write accuracy only under out_adm_dir/accuracy/."""
    out_adm_dir = Path(out_adm_dir)
    source_adm_dir = Path(source_adm_dir)
    gt = _load_gt(source_adm_dir)
    icd = _load_json(out_adm_dir / "icd_coding.json") or {}
    pid = str(gt.get("patient_id") or icd.get("patient_id") or "")
    hid = str(gt.get("hadm_id") or icd.get("hadm_id") or "")

    gt_primary_code = gt.get("primary_icd_code")
    gt_icd_title = ""
    if gt_primary_code and gt.get("dx_titles"):
        gt_icd_title = (gt.get("dx_titles") or [""])[0] or ""
    if not gt_icd_title:
        gt_icd_title = str(gt.get("primary_dx_title") or "")

    gt_discharge_dx = load_discharge_diagnosis_gt(source_adm_dir)
    gt_diffdx_title = gt_discharge_dx or str(gt.get("primary_dx_title") or "")
    gt_source = "discharge_diagnosis" if gt_discharge_dx else "icd_title_fallback"

    items = _diagnosis_items_from_llm_only_icd(icd)
    rank1 = _item_at_rank(items, 1) or (items[0] if items else {})

    qe = llm_clinical_query_expansion(
        ground_truth_primary=gt_diffdx_title,
        ground_truth_icd_code=str(gt_primary_code or ""),
        ground_truth_icd_title=gt_icd_title or gt_diffdx_title,
        diffdx_items=items,
        config=llm_config or get_llm_config(),
        cache=llm_qe_cache,
        patient_id=pid,
        hadm_id=hid,
    )
    dx_match = _diagnosis_match_from_llm(qe)
    matched_rank = qe.get("matched_rank") if dx_match else None
    matched_item = _item_at_rank(items, matched_rank) if matched_rank else {}
    heuristic_applied = False
    if not dx_match:
        hit = _heuristic_match_diffdx_rank(gt_diffdx_title, items)
        if hit:
            dx_match = True
            matched_rank = int(hit.get("rank") or 0) or None
            matched_item = hit
            heuristic_applied = True
            qe = {
                **qe,
                "matched_rank": matched_rank,
                "diagnosis_equivalent": True,
                "diagnosis_relationship": "same_principal",
                "diagnosis_confidence": "medium",
                "diagnosis_reason": (
                    "Heuristic same-principal (specificity/compound containment): "
                    f"DiffDx '{hit.get('diagnosis')}' ↔ GT '{gt_diffdx_title}'"
                ),
                "_heuristic_specificity_match": True,
            }

    is_rank1 = bool(dx_match and matched_rank == 1)
    diagnosis = {
        "ground_truth_primary": gt_diffdx_title,
        "rank1_diagnosis": rank1.get("diagnosis") or "",
        "rank1_score": rank1.get("score"),
        "rank1_confidence": rank1.get("confidence") or "",
        "match": dx_match,
        "is_rank1_match": is_rank1 if dx_match else None,
        "matched_diffdx_rank": matched_rank,
        "matched_diagnosis": matched_item.get("diagnosis") or "" if dx_match else "",
        "matched_score": matched_item.get("score") if dx_match else None,
        "matched_confidence": matched_item.get("confidence") or "" if dx_match else "",
        "llm_relationship": qe.get("diagnosis_relationship") or "",
        "llm_confidence": qe.get("diagnosis_confidence") or "",
        "llm_reason": qe.get("diagnosis_reason") or "",
        "llm_expanded_ground_truth": " | ".join(qe.get("ground_truth_expansions") or []),
        "llm_expanded_matched": " | ".join(qe.get("matched_diagnosis_expansions") or []),
        "heuristic_specificity_match": heuristic_applied,
        "diagnosis_items": items,
        "n_diffdx": len(items),
    }
    icd_linked = _score_linked_icd(
        matched_item,
        rank1,
        gt_primary_code,
        gt_icd_title,
        qe,
        dx_match,
        stage9_primary_icd=icd.get("primary_icd_code"),
    )
    icd_linked["diagnosis_match"] = dx_match
    icd_linked["is_rank1_diagnosis_match"] = is_rank1 if dx_match else None
    icd_linked["matched_diffdx_rank"] = matched_rank

    result = {
        "stage": "llm_only_baseline",
        "type": "accuracy_vs_ground_truth",
        "method": "llm_query_expansion",
        "pipeline": "llm_only_baseline",
        "patient_id": pid,
        "hadm_id": hid,
        "generated_at": datetime.now().isoformat(),
        "ground_truth": {
            "primary_icd_code": gt_primary_code,
            "primary_dx_title": gt.get("primary_dx_title") or "",
            "primary_icd_title": gt_icd_title or gt.get("primary_dx_title") or "",
            "discharge_diagnosis": gt_discharge_dx,
            "diffdx_gt": gt_diffdx_title,
            "diffdx_gt_source": gt_source,
            "n_diagnoses": gt.get("n_diagnoses") or len(gt.get("icd10_codes") or []),
        },
        "evaluation": {"diagnosis": diagnosis, "icd_linked": icd_linked, "llm_qe_raw": qe},
        "has_stage8": True,
        "has_stage9": True,
        "has_ground_truth": bool(gt_diffdx_title or gt_primary_code),
    }

    acc_dir = out_adm_dir / "accuracy"
    acc_dir.mkdir(parents=True, exist_ok=True)
    _write_json(acc_dir / "accuracy.json", result)
    _write_text(acc_dir / "comparison.txt", format_accuracy_comparison_txt(result))
    return result


def run_llm_only_cohort(
    export_dir: Union[str, Path] = None,
    config: Optional[LLMConfig] = None,
    temperature: float = DIFF_DX_TEMPERATURE,
    skip_if_done: bool = True,
    evaluate: bool = True,
    force_llm_qe_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Full LLM-only cohort: DiffDx + ICD + optional eval.

    All artifacts stay under llm_only_baseline/. Does not modify patient_records/
    or data/stage_10_evaluation/.
    """
    ensure_llm_only_readme()
    export_dir = Path(export_dir or EXPORT_DIR)
    cfg = config or get_llm_config()
    admissions = _iter_source_admissions(export_dir)
    print(f"LLM-only baseline → {LLM_ONLY_DIR}")
    print(f"Source admissions : {len(admissions)} under {export_dir}")
    print(f"Model             : {cfg.model}  temp={temperature}")

    run_results = []
    for i, adm in enumerate(admissions, start=1):
        print(f"[{i}/{len(admissions)}] {adm.parent.parent.name}/{adm.name} …")
        try:
            r = run_llm_only_admission(
                adm, config=cfg, temperature=temperature, skip_if_done=skip_if_done
            )
            status = "skip" if r.get("skipped") else "ok"
            print(f"  → {status}  {r.get('out_dir')}")
            run_results.append(r)
            if not r.get("skipped"):
                time.sleep(float(LLM_REQUEST_DELAY_SECONDS or 0))
        except Exception as exc:  # noqa: BLE001 — continue cohort
            print(f"  → ERROR: {exc}")
            run_results.append({"source": str(adm), "error": str(exc)})

    eval_results: List[Dict[str, Any]] = []
    if evaluate:
        cache = {} if force_llm_qe_refresh else load_llm_qe_cache(LLM_ONLY_QE_CACHE)
        for adm in admissions:
            gt = _load_gt(adm)
            pid = str(gt.get("patient_id") or adm.parent.parent.name.replace("patient_", ""))
            hid = str(gt.get("hadm_id") or adm.name.replace("hadm_", ""))
            out_dir = _adm_out_dir(pid, hid)
            if not (out_dir / "icd_coding.json").exists():
                continue
            try:
                ev = evaluate_llm_only_admission(
                    out_dir, adm, llm_config=cfg, llm_qe_cache=cache
                )
                eval_results.append(ev)
                dx = (ev.get("evaluation") or {}).get("diagnosis") or {}
                icd = (ev.get("evaluation") or {}).get("icd_linked") or {}
                print(
                    f"  eval {pid}/{hid}: dx={'YES' if dx.get('match') else 'NO'} "
                    f"icd={'YES' if icd.get('icd_match') else 'NO'}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  eval ERROR {pid}/{hid}: {exc}")
        save_llm_qe_cache(cache, LLM_ONLY_QE_CACHE)

        # Write cohort metrics ONLY into llm_only_baseline/
        summary = summarize_stage10(eval_results)
        summary["pipeline"] = "llm_only_baseline"
        summary["output_dir"] = str(LLM_ONLY_DIR)
        _write_json(LLM_ONLY_SUMMARY, summary)
        _write_text(LLM_ONLY_COHORT_TXT, format_cohort_metrics_txt(summary, eval_results))
        # Reuse CSV writer but redirect paths by temporarily writing via custom call
        _write_llm_only_csvs(eval_results)
        print(f"Saved cohort accuracy → {LLM_ONLY_COHORT_TXT}")

    return {
        "n_source": len(admissions),
        "n_run": len(run_results),
        "n_evaluated": len(eval_results),
        "output_dir": str(LLM_ONLY_DIR),
        "run_results": run_results,
        "eval_results": eval_results,
    }


def _write_llm_only_csvs(results: List[Dict[str, Any]]) -> None:
    """Write DiffDx/ICD CSVs under llm_only_baseline only."""
    import csv

    LLM_ONLY_DIR.mkdir(parents=True, exist_ok=True)
    dx_header = [
        ("patient_id", "Patient ID"),
        ("hadm_id", "Admission ID"),
        ("ground_truth_primary", "Ground truth primary diagnosis"),
        ("rank1_diagnosis", "Rank-1 predicted diagnosis"),
        ("rank1_score", "Rank-1 score"),
        ("rank1_confidence", "Rank-1 confidence"),
        ("match", "Diagnosis match"),
        ("is_rank1_match", "Is rank-1 match"),
        ("matched_diffdx_rank", "Matched DiffDx rank"),
        ("matched_diagnosis", "Matched predicted diagnosis"),
        ("matched_score", "Matched score"),
        ("matched_confidence", "Matched confidence"),
        ("llm_relationship", "LLM relationship"),
        ("llm_confidence", "LLM confidence"),
        ("llm_reason", "LLM match reason"),
        ("full_diffdx_list", "Full DiffDx list"),
    ]
    icd_header = [
        ("patient_id", "Patient ID"),
        ("hadm_id", "Admission ID"),
        ("diagnosis_match", "Diagnosis match"),
        ("matched_diffdx_rank", "Matched DiffDx rank"),
        ("ground_truth_icd", "Ground truth ICD"),
        ("ground_truth_icd_title", "Ground truth ICD title"),
        ("rank1_predicted_icd", "Rank-1 predicted ICD"),
        ("matched_predicted_icd", "Matched predicted ICD"),
        ("icd_match", "ICD match"),
        ("icd_exact_match", "ICD exact"),
        ("icd_family_match", "ICD family"),
        ("llm_icd_reason", "LLM ICD reason"),
    ]
    dx_rows = []
    icd_rows = []
    for r in results:
        ev = r.get("evaluation") or {}
        dx = ev.get("diagnosis") or {}
        icd = ev.get("icd_linked") or {}
        items = dx.get("diagnosis_items") or []
        dx_rows.append(
            {
                "patient_id": r.get("patient_id") or "",
                "hadm_id": r.get("hadm_id") or "",
                "ground_truth_primary": dx.get("ground_truth_primary") or "",
                "rank1_diagnosis": dx.get("rank1_diagnosis") or "",
                "rank1_score": dx.get("rank1_score") if dx.get("rank1_score") is not None else "",
                "rank1_confidence": dx.get("rank1_confidence") or "",
                "match": _yn_export(dx.get("match")),
                "is_rank1_match": _yn_export(dx.get("is_rank1_match")) if dx.get("match") else "",
                "matched_diffdx_rank": dx.get("matched_diffdx_rank") if dx.get("match") else "",
                "matched_diagnosis": dx.get("matched_diagnosis") or "",
                "matched_score": dx.get("matched_score") if dx.get("matched_score") is not None else "",
                "matched_confidence": dx.get("matched_confidence") or "",
                "llm_relationship": dx.get("llm_relationship") or "",
                "llm_confidence": dx.get("llm_confidence") or "",
                "llm_reason": dx.get("llm_reason") or "",
                "full_diffdx_list": _format_diffdx_list_for_export(items),
            }
        )
        icd_rows.append(
            {
                "patient_id": r.get("patient_id") or "",
                "hadm_id": r.get("hadm_id") or "",
                "diagnosis_match": _yn_export(icd.get("diagnosis_match")),
                "matched_diffdx_rank": icd.get("matched_diffdx_rank") if icd.get("diagnosis_match") else "",
                "ground_truth_icd": icd.get("ground_truth_icd") or "",
                "ground_truth_icd_title": icd.get("ground_truth_icd_title") or "",
                "rank1_predicted_icd": icd.get("rank1_predicted_icd") or "",
                "matched_predicted_icd": icd.get("matched_predicted_icd") or "",
                "icd_match": _yn_export(icd.get("icd_match")),
                "icd_exact_match": _yn_export(icd.get("icd_exact_match")),
                "icd_family_match": _yn_export(icd.get("icd_family_match")),
                "llm_icd_reason": icd.get("llm_icd_reason") or "",
            }
        )

    def _write(path: Path, header: List, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([lab for _, lab in header])
            for row in rows:
                w.writerow([row.get(k, "") for k, _ in header])

    _write(LLM_ONLY_DIFFDX_CSV, dx_header, dx_rows)
    _write(LLM_ONLY_ICD_CSV, icd_header, icd_rows)
