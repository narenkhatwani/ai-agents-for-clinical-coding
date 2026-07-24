Clinical Coding Pipeline — Patient Records Export
Generated: 2026-07-20T01:20:07.588909

15 patients | latest admission note + prior history
Agents: LLM information extraction + symptom tree

Folder layout per patient:
  patient_<subject_id>/
    patient_summary.txt / .json
    admission_history.txt / .json   (prior admissions)
    symptom_tree.txt / .json
    admissions/
      hadm_<latest_id>/
        clinical_note.txt              (redacted — LLM / coding input)
        clinical_note_full.txt         (original discharge note)
        redacted_diagnosis_sections.txt (removed dx text — eval only)
        ground_truth.json / .txt       (ICD-10 labels — eval only)
        clinical_context.txt
        structured_vitals.json
        structured_labs.json
        radiology_reports.json
        metadata.json
        ground_truth.txt
        information_extraction.txt / .json
        symptom_tree.txt / .json
