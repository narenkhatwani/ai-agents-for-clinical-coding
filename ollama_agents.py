"""
Clinical NLP agents — information extraction and symptom trees.

Uses the configured LLM backend (Ollama or API) via llm_client.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from llm_client import (
    LLMConfig,
    LLMNotAvailableError,
    OllamaConfig,
    OllamaNotAvailableError,
    call_llm_json,
    check_llm,
    check_ollama,
    parse_json_object,
    require_llm,
    truncate_note,
    warn_if_slow_model,
)

IE_SYSTEM_PROMPT = """You are a clinical Information Extraction Agent (NLP).
Read the clinical note and extract structured medical findings.
Return ONLY valid JSON (no markdown) with this schema:
{
  "symptoms": [{"term": "", "status": "present|absent|history", "evidence": "verbatim phrase from note"}],
  "vitals": [{"name": "", "value": "", "unit": ""}],
  "labs": [{"name": "", "value": "", "unit": "", "flag": "high|low|normal|unknown"}],
  "diagnoses_mentioned": [{"term": "", "certainty": "confirmed|suspected|rule_out"}],
  "medications": [{"name": "", "dose": "", "route": "", "status": "started|continued|stopped"}],
  "procedures": [{"name": "", "result": ""}],
  "negations": [""],
  "temporal": [{"finding": "", "onset": ""}]
}
Use standard clinical terminology. Copy evidence verbatim from the note."""

SYMPTOM_TREE_SYSTEM_PROMPT = """You are a clinical Symptom Tree Agent.
Given a clinical note and structured information extraction, build a hierarchical symptom tree
for ontology routing (Infectious, Cardiovascular, Respiratory, etc.).

Return ONLY valid JSON (no markdown):
{
  "root": "ClinicalPresentation",
  "reasoning": "1-2 sentence summary of dominant clinical picture",
  "branches": [
    {
      "category": "constitutional|respiratory|cardiovascular|infectious|neurologic|gi|renal|other",
      "ontology_hint": "Infectious Diseases|Cardiovascular|Respiratory|Other",
      "symptoms": [
        {
          "term": "standardized symptom name",
          "status": "present|absent|history",
          "severity": "mild|moderate|severe|unknown",
          "evidence": "verbatim phrase",
          "related_findings": ["supporting lab/vital/diagnosis"],
          "children": [
            {"term": "more specific sub-symptom", "status": "present", "evidence": ""}
          ]
        }
      ]
    }
  ],
  "key_symptoms": ["fever", "cough"],
  "red_flags": ["hypotension", "AMS"]
}
Group related symptoms under the correct branch. Use the extraction JSON as a guide but reason over the full note."""


def require_ollama(config: LLMConfig) -> str:
    """Backward-compatible alias."""
    return require_llm(config)


def call_ollama_json(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Backward-compatible alias."""
    return call_llm_json(system_prompt, user_prompt, config, model=model)


def information_extraction_agent(
    clinical_note: str,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """NLP information extraction via configured LLM."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)
    note = truncate_note(clinical_note, config.max_note_chars)

    extracted = call_llm_json(
        IE_SYSTEM_PROMPT,
        f"Clinical note:\n\n{note}",
        config,
        model=model,
    )
    extracted["_method"] = f"{config.method_prefix()}_nlp:{model}"
    extracted["_agent"] = "information_extraction"
    return extracted


def symptom_tree_agent(
    clinical_note: str,
    extracted: Dict[str, Any],
    admission_id: str,
    patient_id: Optional[str] = None,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """LLM-built hierarchical symptom tree from note + information extraction."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)
    note = truncate_note(clinical_note, config.max_note_chars)

    extraction_for_prompt = {
        k: v for k, v in extracted.items() if not str(k).startswith("_")
    }
    user_prompt = (
        f"Admission ID: {admission_id}\n"
        f"Patient ID: {patient_id or 'unknown'}\n\n"
        f"Clinical note:\n{note}\n\n"
        f"Information extraction JSON:\n{json.dumps(extraction_for_prompt, indent=2)}"
    )

    tree = call_llm_json(SYMPTOM_TREE_SYSTEM_PROMPT, user_prompt, config, model=model)
    tree["type"] = "symptom_tree"
    tree["_method"] = f"{config.method_prefix()}_llm:{model}"
    tree["_agent"] = "symptom_tree"
    tree["patient_id"] = patient_id
    tree["admission_id"] = admission_id
    tree["generated_at"] = datetime.now().isoformat()
    return tree


def aggregate_patient_symptom_tree_llm(
    admission_trees: List[Dict[str, Any]],
    patient_id: str,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """LLM merges admission-level symptom trees into a patient-level view."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)

    system = """You are a clinical Symptom Tree Agent.
Merge multiple admission-level symptom trees for ONE patient into a single patient-level tree.
Track which symptoms recurred across admissions. Return ONLY valid JSON:
{
  "root": "PatientClinicalHistory",
  "reasoning": "summary across admissions",
  "n_admissions": 0,
  "branches": [...same branch schema as admission trees...],
  "recurrent_symptoms": [{"term": "", "admissions": ["hadm_id1", "hadm_id2"]}],
  "key_symptoms": []
}"""

    user_prompt = json.dumps(
        {
            "patient_id": patient_id,
            "admission_trees": [
                {k: v for k, v in t.items() if not str(k).startswith("_")}
                for t in admission_trees
            ],
        },
        indent=2,
    )
    tree = call_llm_json(system, user_prompt, config, model=model)
    tree["type"] = "symptom_tree_aggregate"
    tree["_method"] = f"{config.method_prefix()}_llm:{model}"
    tree["_agent"] = "symptom_tree_aggregate"
    tree["patient_id"] = patient_id
    tree["generated_at"] = datetime.now().isoformat()
    tree["n_admissions"] = len(admission_trees)
    return tree
