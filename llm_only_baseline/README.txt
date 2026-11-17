LLM-ONLY BASELINE (isolated)
============================

This folder is completely separate from patient_records/ and data/stage_*.

Pipeline:
  1. DiffDx from redacted clinical note + clinical_context + full prior history
  2. ICD-10 codes assigned by LLM (no SNOMED / ontology routing)
  3. Accuracy: DiffDx vs physician Discharge Diagnosis; ICD vs billed primary ICD

Nothing here is written back into the ontology pipeline exports.
