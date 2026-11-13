# Final output of the RAG with the billed codes added

Billed codes come from each admission's ground_truth.json (evaluation labels, never shown to the model). 'Naren's code' is the code from his pipeline that the RAG checked. 'Naren's code billed?' is whether it appears in the billing record. 'RAG code in billed list' is whether the RAG's final code does (exact, or related = same code but billed more/less specific).


## Patient 10361982, admission 24286431

**Billed primary:** N99820  |  **All billed codes:** N99820, Y836, Y92009, E039, F909

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Post-hysterectomy bleeding | N93.9 | not billed | flag | N99.820 | yes | yes |
| 2 | Menorrhagia | N92.0 | not billed | keep | N92.0 | no | no |
| 3 | Hypothyroidism | E03.9 | billed (secondary) | keep | E03.9 | yes | no |
| 4 | Anxiety disorder | F41.9 | not billed | keep | F41.9 | no | no |

## Patient 10426859, admission 29908281

**Billed primary:** I471  |  **All billed codes:** I471, I214, S32511A, E872, I959, C689, F329, I10, E785, Z66, I2510, F419, K219, I440, G4700, J310, Z9181, Z96641, Z87891, Z8673, W19XXXA, Y929

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Supraventricular tachycardia | I47.10 | same family billed | keep | I47.10 | related (more/less specific) | no |
| 2 | Right hip fracture | S72.009? | not billed | flag | S32.511 | related (more/less specific) | yes |
| 3 | Anxiety disorder | F41.9 | billed (secondary) | keep | F41.9 | yes | no |
| 4 | Hypertension | I10 | billed (secondary) | keep | I10 | yes | no |
| 5 | Hyperlipidemia | E78.5 | billed (secondary) | keep | E78.5 | yes | no |
| 6 | Atherosclerotic heart disease | I25.10 | billed (secondary) | keep | I25.10 | yes | no |

## Patient 10458324, admission 21744342

**Billed primary:** S270XXA  |  **All billed codes:** S270XXA, J189, S2241XA, E46, F1120, Z681, I480, W1830XA, Y92002, R0902, R600, F4310

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Traumatic pneumothorax | J93.9 | not billed | flag | S27.0 | related (more/less specific) | no |
| 2 | Acute respiratory failure | J96.00 | not billed | replace | J96.01 | no | no |
| 3 | Influenza A | J10.1 | not billed | keep | J10.1 | no | no |
| 4 | Opioid dependence | F11.20 | billed (secondary) | keep | F11.20 | yes | no |
| 5 | Bipolar disorder | F31.9 | not billed | keep | F31.9 | no | no |
| 6 | Subcutaneous emphysema | J98.2 | not billed | flag | T81.82, T79.7 | T81.82: no; T79.7: no | yes |

## Patient 11251337, admission 29568708

**Billed primary:** N132  |  **All billed codes:** N132, Z936, E785, I69954, I10, N19, Z87440

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Acute kidney injury | N17.9 | not billed | keep | N17.9 | no | no |
| 2 | Anemia | D64.9 | not billed | flag | D62 | no | yes |
| 3 | Hematuria | R31.9 | not billed | keep | R31.9 | no | no |
| 4 | Pain around nephrostomy tubes | R52 | not billed | flag | R10.85 | no | yes |
| 5 | Fever | R50.9 | not billed | keep | R50.9 | no | no |

## Patient 11474876, admission 29672491

**Billed primary:** K8030  |  **All billed codes:** K8030, I130, I5032, I480, I110, I739, I129, N183, I4510, E785, F4310, N400, Z7902, Z87891, Z85828, Z8249, Z8546

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Heart failure with preserved ejection fraction | I50.30 | same family billed | flag | no code proposed | — | yes |
| 2 | Peripheral vascular disease | I73.9 | billed (secondary) | flag | I70.799, I70.609 | I70.799: no; I70.609: no | yes |
| 3 | Chronic kidney disease | N18.30 | same family billed | flag | no code proposed | — | yes |
| 4 | Atrial fibrillation | I48.91 | same family billed | keep | I48.91 | no | no |
| 5 | Leg swelling | M79.89 | not billed | flag | R60.0 | no | yes |
| 6 | Nocturnal hypoxia | R09.02 | not billed | keep | R09.02 | no | no |

## Patient 11607177, admission 23293838

**Billed primary:** I472  |  **All billed codes:** I472, Z95811, N179, I429, Z8673, E119, E785, I10, M109, Z96652, I272, Z6837, E669

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Refractory ventricular tachycardia | I47.20 | same family billed | keep | I47.20 | related (more/less specific) | no |
| 2 | Dilated non-ischemic cardiomyopathy | I42.0 | same family billed | keep | I42.0 | no | no |
| 3 | Heart failure | I50.9 | not billed | flag | no code proposed | — | yes |
| 4 | Hypertension | I10 | billed (secondary) | keep | I10 | yes | no |
| 5 | Obesity | E66.9 | billed (secondary) | keep | E66.9 | yes | no |

## Patient 12007928, admission 23749816

**Billed primary:** I5033  |  **All billed codes:** I5033, N179, G7000, Q211, F0390, I480, G629, F329, I10, E041, G40909, E039, E785, Z7901, Z950, K219, G4730, N3941, Z66

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Acute decompensated heart failure | I50.9 | same family billed | replace | I50.3 | related (more/less specific) | no |
| 2 | Chronic kidney disease | N18.9 | not billed | flag | no code proposed | — | yes |
| 3 | Paroxysmal atrial fibrillation | I48.0 | billed (secondary) | keep | I48.0 | yes | no |
| 4 | Congestive heart failure | I50.9 | same family billed | replace | I50.3 | related (more/less specific) | no |
| 5 | Myasthenia gravis | G70.00 | billed (secondary) | keep | G70.00 | yes | no |

## Patient 13196707, admission 21475988

**Billed primary:** A419  |  **All billed codes:** A419, G936, I314, J690, R6521, E870, K521, G92, E872, J189, L89153, N179, C787, C7931, C7951, I313, I319, C3490, Z515, Z66, I10, T360X5A, Y92239, R740, D638, K259, Z87891, R569, K5900, Z7902, R0902, Z86718

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Severe sepsis secondary to RUL PNA | A41.9 | billed (primary) | keep | A41.9 | yes | no |
| 2 | Metastatic adenocarcinoma of unknown primary | C79.9 | same family billed | keep | C79.9 | no | no |
| 3 | Altered mental status due to metastatic brain lesions | R41.82 | not billed | flag | C79.31 | yes | no |
| 4 | Hypotension | I95.9 | not billed | keep | I95.9 | no | no |
| 5 | Pneumonia | J18.9 | billed (secondary) | flag | no code proposed | — | yes |
| 6 | Acute kidney failure | I50.9 | not billed | flag | N17.9 | yes | yes |

## Patient 13508515, admission 21834271

**Billed primary:** C642  |  **All billed codes:** C642, Z6841, I429, I272, I509, I5020, J449, E119, I2510, Z951, G4733, E669, Z87891, I340

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Acute on chronic systolic heart failure | I50.23 | same family billed | keep | I50.23 | no | no |
| 3 | Type 2 diabetes mellitus | E11.9 | billed (secondary) | flag | E11.22 | no | yes |
| 4 | Obstructive sleep apnea | G47.33 | billed (secondary) | keep | G47.33 | yes | no |
| 5 | Anemia | D64.9 | not billed | flag | D62, D63.0 | D62: no; D63.0: no | yes |
| 6 | Hypokalemia | E87.6 | not billed | flag | no code proposed | — | yes |

## Patient 13952483, admission 23852410

**Billed primary:** I9589  |  **All billed codes:** I9589, N179, R64, D6959, B3781, I272, D701, C3432, Z940, Z681, J449, E860, R627, D649, T451X5A, T66XXXA, I10, E785, R1319, Y929, R630, B349, F329, F17210, Z79899

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Squamous Cell Lung Carcinoma | C34.90 | same family billed | flag | no code proposed | — | yes |
| 2 | Pulmonary Hypertension | I27.20 | same family billed | flag | I27.81 | no | yes |
| 3 | Anemia | D64.9 | billed (secondary) | flag | D63.0 | no | yes |
| 4 | Malnutrition | E46 | not billed | keep | E46 | no | no |
| 5 | Thrombocytopenia | D69.6 | same family billed | keep | D69.6 | no | no |
| 6 | Neutropenia | D70.9 | same family billed | keep | D70.9 | no | no |

## Patient 16014068, admission 29042843

**Billed primary:** C7931  |  **All billed codes:** C7931, G936, C7801, D619, C7802, C775, D696, M7981, Z86711, Z86718, E538, L8992, R791, G629, K219, I129, N183, Z85830, Z87891

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Metastatic sarcoma with anemia and thrombocytopenia | C79.89 | same family billed | keep | C79.89 | no | no |
| 2 | Chronic anemia and thrombocytopenia | D61.810 | same family billed | flag | D69.6 | yes | yes |
| 4 | Acute kidney failure | I50.9 | not billed | flag | N17.9 | no | yes |
| 5 | Hypotension | I95.9 | not billed | flag | no code proposed | — | yes |

## Patient 17774110, admission 27339772

**Billed primary:** A4189  |  **All billed codes:** A4189, R6521, N170, J9601, G92, D65, I82220, C7972, J910, D689, I4892, C220, K766, E872, K920, Z515, B1920, F319, F909, K219, J449, K2270, F1010, B961, I4891, K7290, F209, G40909, K7030

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Septic shock with multi-organ failure | R65.21 | billed (secondary) | keep | R65.21 | yes | no |
| 2 | Hepatic encephalopathy | K76.82 | same family billed | keep | K76.82 | no | no |
| 3 | Acute liver failure | K72.00 | same family billed | keep | K72.00 | no | no |
| 4 | GI bleeding | K92.2 | same family billed | flag | I85.01 | no | yes |
| 5 | Coagulopathy | D61.818 | not billed | flag | D68.9 | yes | yes |
| 6 | Atrial fibrillation/flutter | I48.91 | billed (secondary) | keep | I48.91 | yes | no |

## Patient 18412100, admission 26093939

**Billed primary:** D481  |  **All billed codes:** D481, B1920, K219, G8918, R109, R110, E669, Z6832

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Post-procedural pain | R10.9 | billed (secondary) | flag | G89.18 | yes | yes |
| 2 | Hepatitis C | B19.20 | billed (secondary) | flag | B18.2 | no | no |
| 3 | Hypokalemia | E87.6 | not billed | keep | E87.6 | no | no |
| 4 | Hypocalcemia | E83.51 | not billed | keep | E83.51 | no | no |
| 5 | Anemia | D64.9 | not billed | keep | D64.9 | no | no |

## Patient 19104262, admission 24271247

**Billed primary:** O2403  |  **All billed codes:** O2403, O904, G9340, E1010, E875, E860, O99355, O99285, Z794, Z9114

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Diabetic ketoacidosis | E13.10 | not billed | flag | Z79.4, Z79.84, Z79.85 | Z79.4: yes; Z79.84: no; Z79.85: no | yes |
| 2 | Type 1 diabetes mellitus with ketoacidosis | E10.9 | same family billed | flag | E10.10 | yes | no |
| 3 | Lactic acidosis | E87.20 | same family billed | flag | no code proposed | — | yes |
| 4 | Missed abortion | O02.1 | not billed | keep | O02.1 | no | no |
| 6 | Type 1 diabetes mellitus | E10.9 | same family billed | keep | E10.9 | no | no |
| 7 | Decreased fetal movement | O36.8190 | not billed | flag | Z3A.00 | no | yes |

## Patient 19632936, admission 26696232

**Billed primary:** D509  |  **All billed codes:** D509, I272, I495, I2510, I4891, I10, E785, I350, Z955, Z7901, Z950, Z87891, N400, K2270, I878

| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |
|---|---|---|---|---|---|---|---|
| 1 | Acute macrocytic anemia with respiratory compromise | D53.9 | not billed | flag | no code proposed | — | yes |
| 2 | Chronic diastolic heart failure | I50.32 | not billed | keep | I50.32 | no | no |
| 3 | Atrial fibrillation | I48.91 | billed (secondary) | keep | I48.91 | yes | no |
| 4 | Anemia of chronic disease | D63.8 | not billed | flag | no code proposed | — | yes |
| 6 | Coronary artery disease | I25.10 | billed (secondary) | keep | I25.10 | yes | no |