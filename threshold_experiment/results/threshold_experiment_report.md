# SNOMED MiniLM threshold experiment

Generated: `2026-08-08T17:28:57.273581`

## Setup

- **Source:** Random sample from local US SNOMED CT Snapshot (RF2). Same concept/description model as SNOMED Terminology API.
- **n terms:** 20 (seed=42)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Neighbors:** is_a depth≤2 + Stage-6 attribute targets (no inverse)

## Average similarities (across sampled terms)

| Quantity | mean | median | p25 | p75 | n |
|----------|-----:|-------:|----:|----:|--:|
| Most relevant neighbor (top-1) | 0.7254 | 0.7773 | 0.6165 | 0.8841 | 20 |
| Top-2 mean | 0.6679 | 0.707 | 0.56 | 0.8527 | 20 |
| Is-a parent (depth 1) | 0.6565 | 0.7148 | 0.5247 | 0.8572 | 32 |
| All is-a (depth ≤2) | 0.5152 | 0.5243 | 0.32 | 0.742 | 99 |
| All neighbors (is-a + attributes) | 0.4677 | 0.4698 | 0.2335 | 0.6943 | 159 |

## Pass rates

| Threshold | top-1 passes | is-a parent passes | any-neighbor passes |
|----------:|-------------:|-------------------:|--------------------:|
| 0.7 | 0.65 | 0.5625 | 0.2453 |
| 0.75 | 0.55 | 0.4375 | 0.1824 |
| 0.8 | 0.4 | 0.3438 | 0.1195 |

## Recommendation

- **Recommended min similarity:** **0.7**
- **Recommended high-confidence:** **0.8**
- Policy floor (never below): `0.7`
- Current pipeline: min=`0.7`, high-conf=`0.8`

Across n=20 sampled SNOMED terms, the average MiniLM cosine of the single most-relevant ontology neighbor (top-1) is 0.725; mean over top-2 is 0.668; mean direct is-a parent is 0.657. Band center ≈ 0.683. Recommended retain threshold is 0.70 (≥ 0.70 policy floor); high-confidence tier ≥ 0.80.

## Sampled concepts (top-1 neighbor)

- `31942007` **Transection of greater occipital nerve** → *Transection of nerve* (sim=0.7253)
- `253600008` **Quadricuspid pulmonary valve** → *Congenital abnormality of pulmonary valve cusp* (sim=0.5921)
- `127350007` **Motor vehicle accident, passenger** → *Motor vehicle accident victim* (sim=0.8843)
- `55131000087105` **At increased risk of disorientation** → *Finding of increased risk level* (sim=0.328)
- `117660006` **Norketamine measurement** → *Measurement* (sim=0.4374)
- `14386001` **Indeterminate leprosy** → *Leprosy* (sim=0.7955)
- `208636004` **Closed fracture distal tibia, intra-articular** → *Closed fracture distal tibia* (sim=0.9347)
- `1003549007` **Agenesis of radius** → *Radius* (sim=0.6852)
- `388722004` **Rf271 specific IgE antibody measurement** → *Allergen specific IgE antibody measurement* (sim=0.7295)
- `400170001` **Hypocalcaemia of puerperium** → *Puerperal hypocalcaemia* (sim=0.8517)
- `15724121000119102` **Enthesopathy of bilateral knees** → *Enthesopathy of knee* (sim=0.9067)
- `679101000119104` **Xerosis of cornea of right eye** → *Xerosis of cornea* (sim=0.8999)
- `27572006` **Intention myoclonus** → *Myoclonus* (sim=0.7708)
- `424728002` **Prepapillary vascular loop** → *Ocular blood vessel* (sim=0.4698)
- `1110121000000102` **Ocular hypertension due to aphakia** → *Ocular hypertension* (sim=0.7838)
- `30043008` **Left lateral anal sphincterotomy** → *Lateral sphincterotomy* (sim=0.8916)
- `276978009` **Percutaneous embolectomy of intracranial artery** → *Removal of embolus from intracranial artery* (sim=0.8734)
- `874930000` **Nonvenomous insect bite of scrotum** → *Insect bite, nonvenomous, of scrotum and testis* (sim=0.884)
- `364377005` **Appearance of nipple** → *Nipple observable* (sim=0.6246)
- `408551003` **Exercise tolerance test refused** → *Procedure refused* (sim=0.4391)
