"""
Export cohort data into a per-patient folder structure.

Layout:
    patient_records/
        cohort_index.json
        README.txt
        patient_<subject_id>/
            patient_summary.json
            patient_summary.txt
            symptom_tree.json          # patient-level aggregate
            symptom_tree.txt
            admissions/
                hadm_<hadm_id>/
                    clinical_note.txt
                    metadata.json
                    ground_truth.txt
                    information_extraction.json
                    information_extraction.txt
                    symptom_tree.json
                    symptom_tree.txt
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _format_section(title: str, char: str = "=") -> str:
    line = char * len(title)
    return f"{title}\n{line}\n"


def format_ground_truth_txt(row: pd.Series) -> str:
    lines = [
        _format_section("GROUND TRUTH DIAGNOSES"),
        f"Admission ID     : {row.get('admission_id', row.get('hadm_id'))}",
        f"Primary ICD-10   : {row.get('primary_icd_code', row.get('icd_code', 'N/A'))}",
        f"Primary Title    : {row.get('primary_dx_title', row.get('long_title', 'N/A'))}",
        f"Total diagnoses  : {row.get('n_diagnoses', 'N/A')}",
        "",
        "All ICD-10 codes (ordered):",
        "-" * 40,
    ]
    codes = row.get("ground_truth_icd10") or []
    titles = row.get("ground_truth_dx_titles") or []
    for i, code in enumerate(codes):
        title = titles[i] if i < len(titles) else ""
        lines.append(f"  {i+1:2d}. {code} — {title}")
    return "\n".join(lines) + "\n"


def format_extraction_txt(extracted: Dict[str, Any], meta: Dict[str, Any]) -> str:
    lines = [
        _format_section("INFORMATION EXTRACTION"),
        f"Patient ID       : {meta.get('patient_id')}",
        f"Admission ID     : {meta.get('admission_id')}",
        f"Method           : {meta.get('extraction_method', 'unknown')}",
        "",
    ]

    def block(name: str, items: List[Any], formatter) -> None:
        lines.append(_format_section(name, "-"))
        if not items:
            lines.append("  (none)\n")
            return
        for item in items:
            lines.append(formatter(item))
        lines.append("")

    block("SYMPTOMS", extracted.get("symptoms", []),
          lambda s: f"  • [{s.get('status')}] {s.get('term')} — \"{s.get('evidence', '')}\"")
    block("VITALS", extracted.get("vitals", []),
          lambda v: f"  • {v.get('name')}: {v.get('value')} {v.get('unit', '')}".strip())
    block("LABS", extracted.get("labs", []),
          lambda l: f"  • {l.get('name')}: {l.get('value')} {l.get('unit', '')} [{l.get('flag', '')}]".strip())
    block("DIAGNOSES MENTIONED", extracted.get("diagnoses_mentioned", []),
          lambda d: f"  • [{d.get('certainty')}] {d.get('term')}")
    block("MEDICATIONS", extracted.get("medications", []),
          lambda m: f"  • {m.get('name')} ({m.get('status')})")
    block("PROCEDURES", extracted.get("procedures", []),
          lambda p: f"  • {p.get('name')}: {p.get('result', '')}")
    block("NEGATIONS", extracted.get("negations", []),
          lambda n: f"  • {n}")
    block("TEMPORAL", extracted.get("temporal", []),
          lambda t: f"  • {t.get('finding')}: {t.get('onset')}")

    return "\n".join(lines)


def _format_symptom_node_lines(node: Dict[str, Any], indent: int = 2) -> List[str]:
    lines: List[str] = []
    prefix = " " * indent
    term = node.get("term", "")
    status = node.get("status", "")
    severity = node.get("severity", "")
    evidence = node.get("evidence", "")
    related = node.get("related_findings") or []

    status_label = status
    if severity and severity != "unknown":
        status_label = f"{status}, {severity}"
    lines.append(f"{prefix}• [{status_label}] {term}")
    if evidence:
        lines.append(f"{prefix}    evidence: \"{evidence}\"")
    if related:
        lines.append(f"{prefix}    related: {', '.join(str(r) for r in related)}")
    for child in node.get("children") or []:
        lines.extend(_format_symptom_node_lines(child, indent + 4))
    return lines


def format_symptom_tree_txt(tree: Dict[str, Any]) -> str:
    lines = [
        _format_section("SYMPTOM TREE"),
        f"Type             : {tree.get('type')}",
        f"Method           : {tree.get('_method', 'unknown')}",
        f"Root             : {tree.get('root', 'N/A')}",
        f"Patient ID       : {tree.get('patient_id', 'N/A')}",
        f"Admission ID     : {tree.get('admission_id', 'N/A (aggregate)')}",
    ]
    if tree.get("n_admissions"):
        lines.append(f"Admissions       : {tree.get('n_admissions')}")
    if tree.get("reasoning"):
        lines.extend(["", "Reasoning:", f"  {tree['reasoning']}", ""])

    key_symptoms = tree.get("key_symptoms") or []
    if key_symptoms:
        lines.append(f"Key symptoms     : {', '.join(key_symptoms)}")
    red_flags = tree.get("red_flags") or []
    if red_flags:
        lines.append(f"Red flags        : {', '.join(red_flags)}")
    recurrent = tree.get("recurrent_symptoms") or []
    if recurrent:
        lines.append("")
        lines.append(_format_section("RECURRENT SYMPTOMS", "-"))
        for item in recurrent:
            adms = ", ".join(str(a) for a in item.get("admissions", []))
            lines.append(f"  • {item.get('term')} [admissions: {adms}]")

    branches = tree.get("branches") or []
    if branches:
        lines.append("")
        for branch in branches:
            cat = branch.get("category", "other")
            hint = branch.get("ontology_hint", "")
            lines.append(_format_section(f"{cat.upper()} → {hint}", "-"))
            for symptom in branch.get("symptoms") or []:
                lines.extend(_format_symptom_node_lines(symptom, indent=2))
            lines.append("")

    return "\n".join(lines)


def format_patient_summary_txt(patient_id: str, cohort_rows: pd.DataFrame, n_admissions: int) -> str:
    first = cohort_rows.iloc[0]
    lines = [
        _format_section("PATIENT SUMMARY"),
        f"Patient ID       : {patient_id}",
        f"Subject ID       : {first.get('subject_id', patient_id)}",
        f"Gender           : {first.get('gender', 'N/A')}",
        f"Age (anchor)     : {first.get('anchor_age', 'N/A')}",
        f"Admissions       : {n_admissions}",
        "",
        "Admission history:",
        "-" * 40,
    ]
    for _, row in cohort_rows.iterrows():
        lines.append(
            f"  • hadm {row['hadm_id']} | {row.get('admittime', '')} | "
            f"{row.get('admission_type', '')} | primary: {row.get('primary_icd_code', row.get('icd_code', 'N/A'))}"
        )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_cohort_to_folders(
    cohort_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_dir: Path | str = "patient_records",
    patient_symptom_trees: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Export all patients to organized folders with txt + json artifacts.
    Returns the output root path.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    cohort_by_patient = {pid: grp for pid, grp in cohort_df.groupby("patient_id")}
    results_by_patient = {pid: grp for pid, grp in results_df.groupby("patient_id")}

    index: List[Dict[str, Any]] = []

    for patient_id, patient_cohort in cohort_by_patient.items():
        patient_dir = root / f"patient_{patient_id}"
        admissions_dir = patient_dir / "admissions"
        admissions_dir.mkdir(parents=True, exist_ok=True)

        patient_results = results_by_patient.get(patient_id)
        if patient_results is None:
            continue

        admission_trees: List[Dict[str, Any]] = []
        admission_index: List[Dict[str, Any]] = []

        for _, cohort_row in patient_cohort.iterrows():
            hadm_id = str(cohort_row["hadm_id"])
            adm_dir = admissions_dir / f"hadm_{hadm_id}"
            adm_dir.mkdir(parents=True, exist_ok=True)

            result_row = patient_results[patient_results["hadm_id"] == cohort_row["hadm_id"]]
            if result_row.empty:
                continue
            result_row = result_row.iloc[0]
            extracted = result_row["extracted"]
            tree = result_row.get("symptom_tree")
            if tree is None or (isinstance(tree, float) and pd.isna(tree)):
                raise ValueError(
                    f"Missing LLM symptom_tree for hadm_id={hadm_id}. "
                    "Run the Ollama pipeline before export."
                )

            # Clinical note
            _write_text(adm_dir / "clinical_note.txt", cohort_row.get("clinical_note", cohort_row.get("text", "")))

            # Metadata + ground truth
            metadata = {
                "patient_id": patient_id,
                "subject_id": int(cohort_row["subject_id"]),
                "hadm_id": int(cohort_row["hadm_id"]),
                "admission_id": str(cohort_row["admission_id"]),
                "admittime": str(cohort_row.get("admittime", "")),
                "dischtime": str(cohort_row.get("dischtime", "")),
                "admission_type": cohort_row.get("admission_type"),
                "gender": cohort_row.get("gender"),
                "anchor_age": int(cohort_row["anchor_age"]) if pd.notna(cohort_row.get("anchor_age")) else None,
                "note_type": cohort_row.get("note_type", "discharge"),
                "text_len": int(cohort_row.get("text_len", 0)),
                "ground_truth": {
                    "primary_icd_code": cohort_row.get("primary_icd_code", cohort_row.get("icd_code")),
                    "primary_dx_title": cohort_row.get("primary_dx_title", cohort_row.get("long_title")),
                    "icd10_codes": cohort_row.get("ground_truth_icd10", []),
                    "dx_titles": cohort_row.get("ground_truth_dx_titles", []),
                    "n_diagnoses": int(cohort_row.get("n_diagnoses", 0)) if pd.notna(cohort_row.get("n_diagnoses")) else None,
                },
            }
            _write_json(adm_dir / "metadata.json", metadata)
            _write_text(adm_dir / "ground_truth.txt", format_ground_truth_txt(cohort_row))

            # Information extraction
            ie_meta = {
                "patient_id": patient_id,
                "admission_id": hadm_id,
                "extraction_method": result_row.get("extraction_method"),
            }
            _write_json(adm_dir / "information_extraction.json", extracted)
            _write_text(adm_dir / "information_extraction.txt", format_extraction_txt(extracted, ie_meta))

            # Symptom tree (admission-level, from Ollama LLM)
            admission_trees.append(tree)
            _write_json(adm_dir / "symptom_tree.json", tree)
            _write_text(adm_dir / "symptom_tree.txt", format_symptom_tree_txt(tree))

            branch_symptoms = sum(
                len(b.get("symptoms") or []) for b in (tree.get("branches") or [])
            )
            admission_index.append({
                "hadm_id": hadm_id,
                "admittime": str(cohort_row.get("admittime", "")),
                "primary_icd_code": metadata["ground_truth"]["primary_icd_code"],
                "primary_dx_title": metadata["ground_truth"]["primary_dx_title"],
                "symptom_count": branch_symptoms,
                "symptom_tree_method": tree.get("_method"),
            })

        # Patient-level summary + aggregate symptom tree
        n_adm = len(patient_cohort)
        patient_summary = {
            "patient_id": patient_id,
            "subject_id": int(patient_cohort.iloc[0]["subject_id"]),
            "gender": patient_cohort.iloc[0].get("gender"),
            "anchor_age": int(patient_cohort.iloc[0]["anchor_age"]) if pd.notna(patient_cohort.iloc[0].get("anchor_age")) else None,
            "n_admissions": n_adm,
            "admissions": admission_index,
            "generated_at": datetime.now().isoformat(),
        }
        _write_json(patient_dir / "patient_summary.json", patient_summary)
        _write_text(patient_dir / "patient_summary.txt", format_patient_summary_txt(patient_id, patient_cohort, n_adm))

        patient_tree = (patient_symptom_trees or {}).get(patient_id)
        if patient_tree is None:
            raise ValueError(
                f"Missing patient-level symptom tree for patient_id={patient_id}. "
                "Pass patient_symptom_trees from the notebook pipeline."
            )
        _write_json(patient_dir / "symptom_tree.json", patient_tree)
        _write_text(patient_dir / "symptom_tree.txt", format_symptom_tree_txt(patient_tree))

        index.append({
            "patient_id": patient_id,
            "folder": str(patient_dir.relative_to(root)),
            "n_admissions": n_adm,
            "admissions": admission_index,
        })

    cohort_index = {
        "generated_at": datetime.now().isoformat(),
        "n_patients": len(index),
        "n_admissions": int(len(results_df)),
        "patients": index,
    }
    _write_json(root / "cohort_index.json", cohort_index)

    readme = f"""Clinical Coding Pipeline — Patient Records Export
Generated: {cohort_index['generated_at']}

{len(index)} patients | {cohort_index['n_admissions']} admissions
Agents: Ollama NLP information extraction + LLM symptom tree

Folder layout per patient:
  patient_<subject_id>/
    patient_summary.txt / .json
    symptom_tree.txt / .json          (aggregated across admissions)
    admissions/
      hadm_<id>/
        clinical_note.txt
        metadata.json
        ground_truth.txt
        information_extraction.txt / .json
        symptom_tree.txt / .json
"""
    _write_text(root / "README.txt", readme)

    return root
