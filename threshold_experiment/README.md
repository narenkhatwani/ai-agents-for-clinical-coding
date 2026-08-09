# Threshold experiment (MiniLM × SNOMED CT)

Standalone calibration for Stage 6 retain thresholds. Lives at the **repo root** so it stays separate from the pipeline notebooks.

## What it does

1. Loads local **SNOMED CT RF2** (same concepts/terms as the SNOMED Terminology API)
2. Draws **20 random** clinical concepts (findings / disorders / procedures / …)
3. For each concept, collects **Stage-6-style neighbors**:
   - Is-a parent + grandparent
   - Attribute targets (Due to, causative agent, finding site, morphology, pathological process)
4. Embeds labels with **`all-MiniLM-L6-v2`**
5. For each seed term, ranks neighbors by cosine similarity and takes the **most relevant** ones
6. Averages across the 20 terms and **recommends** a min-similarity threshold (never below **0.70**)

## Run

From the repository root:

```bash
pip install sentence-transformers pandas
python threshold_experiment/run_threshold_experiment.py
# optional
python threshold_experiment/run_threshold_experiment.py --n 20 --seed 42
```

Requires an existing SNOMED index (or the RF2 package under `data/SnomedCT_*` so the index can build).

## Outputs

| File | Content |
|------|---------|
| `results/threshold_experiment_summary.json` | Distributions, pass rates, recommendation |
| `results/threshold_experiment_per_term.json` | Full per-term neighbor rankings |
| `results/threshold_experiment_report.md` | Human-readable report |

## How to read the recommendation

- **top-1 mean** — average sim of the single best ontology neighbor per term  
- **top-2 mean** — average of the two best  
- **is-a parent mean** — average sim of true direct parents  

Recommended min similarity is snapped to `{0.70, 0.75, 0.80}` based on that band, with a **policy floor of 0.70**.
