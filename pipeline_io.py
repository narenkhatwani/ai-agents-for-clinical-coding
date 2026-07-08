"""Load/save artifacts between stage notebooks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from pipeline_config import (
    IE_CHECKPOINT_JSON,
    IE_RESULTS_JSON,
    STAGE_02_DIR,
    STAGE_03_DIR,
    SYMPTOM_TREE_RESULTS_JSON,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_ie_results(
    results_df: pd.DataFrame,
    path: Union[str, Path] = IE_RESULTS_JSON,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    _ensure_parent(path)
    records = results_df.to_dict(orient="records")
    payload = {
        "stage": 2,
        "description": "Ollama information extraction",
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "results": records,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_ie_results(path: Union[str, Path] = IE_RESULTS_JSON) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"IE results not found at {path}. Run notebooks/stage_02_information_extraction.ipynb first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["results"])


def load_ie_checkpoint(path: Union[str, Path] = IE_CHECKPOINT_JSON) -> Dict[int, Dict[str, Any]]:
    """Return completed extractions keyed by hadm_id (for resume)."""
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["hadm_id"]): r for r in payload.get("results", [])}


def save_ie_checkpoint(records: List[Dict[str, Any]], path: Union[str, Path] = IE_CHECKPOINT_JSON) -> Path:
    """Save progress after each admission so a timeout does not lose work."""
    path = Path(path)
    _ensure_parent(path)
    payload = {
        "stage": 2,
        "checkpoint": True,
        "updated_at": datetime.now().isoformat(),
        "n_completed": len(records),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_symptom_tree_results(
    results_df: pd.DataFrame,
    patient_symptom_trees: Dict[str, Dict[str, Any]],
    path: Union[str, Path] = SYMPTOM_TREE_RESULTS_JSON,
) -> Path:
    path = Path(path)
    _ensure_parent(path)
    records = results_df.to_dict(orient="records")
    payload = {
        "stage": 3,
        "description": "Ollama symptom tree (admission + patient aggregate)",
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "n_patients": len(patient_symptom_trees),
        "results": records,
        "patient_symptom_trees": patient_symptom_trees,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_symptom_tree_results(
    path: Union[str, Path] = SYMPTOM_TREE_RESULTS_JSON,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Symptom tree results not found at {path}. Run notebooks/stage_03_symptom_tree.ipynb first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["results"]), payload.get("patient_symptom_trees", {})
