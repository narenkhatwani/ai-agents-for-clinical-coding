# Guidelines-grounded ICD-10-CM code checker: summary of the RAG

## 1. What it is

A verification layer for the coding pipeline. Naren's Stage 8/9 proposes an ICD-10-CM code for each diagnosis. This system
checks each proposed code against two official sources and returns a structured verdict that names the rules it relied on:

- **keep**: the code is acceptable (possibly with a companion code to add);
- **flag**: the code is wrong, too unspecific, or incomplete, and a provider query may be needed;
- **replace**: a different code is proposed.

It does not discover diagnoses and it does not read the clinical notes. Its input is Naren's diagnosis row plus the
retrieved reference material. The LLM is Naren's configured model (Claude Opus 5 via OpenRouter, ZDR on, temperature 0.3).

## 2. Knowledge sources and why these two

| Source | What it gives | Why it is used this way |
|---|---|---|
| **Tabular List** (FY2026 XML) | Each code's exact title, Excludes1/Excludes2, code-first and use-additional notes, 7th-character rules. Notes are **inherited** from parent categories. | It is structured data, so it is looked up exactly, not retrieved by similarity. A wrong Excludes1 would be a real coding error, and similarity search cannot guarantee exactness. |
| **Official Guidelines** (FY2026 PDF) | The coding rules: complications, unspecified codes, sequencing, symptom codes, and so on. | It is prose, so it needs retrieval. It is organised as an outline, so it is chunked along that outline. |

Not used: the Alphabetic Index, the Code Descriptions files, and the PDF versions of the Tabular List. The Tabular XML already
holds every code and note, and the checker does not need to look terms up from free text.

## 3. The steps

**One-time build (no LLM, free)**
1. `tabular_list.py` parses the XML into one record per code, with inherited notes.
2. `guidelines_chunker.py` splits the Guidelines PDF along its own headings. Parent chunks are whole rule sections (up to 900
   tokens, what the LLM reads). Child chunks are small pieces (up to 220 tokens, what gets searched), each prefixed with its
   heading path.
3. `guideline_rag.py` builds the flat search index, and `guideline_hier.py` builds the outline (hierarchical) index.

**Per candidate code (what happens at run time)**
1. **Build the case.** From Naren's saved Stage 8 files: diagnosis, candidate code and title, up to 4 supporting-evidence
   phrases, and up to 5 procedures from the admission.
2. **Retrieve rules.** Run the fused search for rules related to the case. Cues in the case (complication wording, an
   "unspecified" code, code-first notes, 7th characters) add targeted queries. Whole-Section introductions are dropped.
3. **Look up the Tabular facts** for the candidate code, and build the allowed-code list (the candidate and its siblings).
4. **Prompt the LLM** with the case, the Tabular facts, the retrieved rules and the allowed codes. It must answer in a fixed JSON schema.
5. **Validate.** Check the schema and consistency, such as cited rules actually shown and no invented codes. One retry if invalid.
6. **Find a code for any suggested condition.** If the LLM names another condition in plain English, search the Tabular titles
   for candidate codes, then a small LLM call picks from that candidate pool.
7. **Write the output:** verdict, cited rules, reason, provider question, final code.

**Evaluation (after the run)**
`rag_compare_report.py` compares the checker with Naren's saved codes and the billing record and writes a report. Each admission
is also run once without the Guidelines text (the "control arm"), so that the effect of the Guidelines can be isolated.

## 4. Components and why each was chosen

| Component | What it does | Why |
|---|---|---|
| Heading-aware chunker | Parents = rule sections, children = small searchable pieces with the heading path in front | A rule makes sense only within its heading path. Small children make search precise, and the full parent gives the LLM complete context. A bug found on 2026-10-27 (bold body text absorbed into headings, which dropped the Type 2 diabetes in remission rule) was fixed, and the v2 chunks are used. |
| Flat retriever (MiniLM vectors + BM25, reciprocal rank fusion, in Chroma) | Finds paragraphs by meaning and by exact words | Vectors catch paraphrases, and BM25 catches exact terms such as "Excludes" or code words. Rank fusion combines them without tuning weights. |
| Hierarchical retriever | Walks the Guidelines outline top-down (Section, letter, chapter, topic, rule). Keeps the best 5 at each level and blends scores down the path | Your idea: narrow to the right area first, then to the specific rule, the way a coder would. On held-out scenarios it showed the governing rule in 20/25 cases against 15/25 for flat search. |
| Fused retriever (used in the final run) | Merges the flat and hierarchical result lists | The outline walk can prune the right branch, and the flat search can miss a rule that sits in a weak section. Together they showed the governing rule in 21/25. |
| Section-introduction filter | Removes whole-Section introductions (I, II, III, IV) from results | They are long and generic and crowded out specific rules. They were 16% of citations before the filter and 0% after. |
| Tabular exact lookup | Returns notes, Excludes1, 7th-character rules and the allowed-code list | Exactness and no hallucination risk. |
| Structured verdict + citation requirement | LLM must return keep/flag/replace and cite rule ids that were actually shown | Makes every decision auditable. It also lets the validator reject answers that cite rules that were not retrieved. |
| Validator and one retry | Rejects malformed or inconsistent answers | Keeps output usable without letting a single bad generation through. |
| Code lookup + LLM pick from a pool | Turns a plain-English suggestion into code(s) | The LLM can never invent a code. Title search alone was the weakest step, so the pick was added. |
| Procedures in the case | Shows up to 5 procedures from the stay | Context for complication questions (for example bleeding after a hysterectomy). |
| Naren's LLM layer | Same model, temperature and calls | Keeps the comparison with his baseline fair, and nothing of his is modified. |

## 5. How to read the output

- **Final code** is the accepted code for *keep*, or the code the checker selected for the condition it named for *flag/replace*.
- **Provider query = yes** means the documentation cannot settle the question, so the checker asks a clinician/coder rather than guessing.
- A flag on a code that was not billed is a prompt for review, not proof of an error.
- The code selection step is the least reliable part. A few selected codes in the final output look wrong
  (for example the diabetic ketoacidosis row, which was given long-term drug use Z codes) and need a coder's check.
  (The hip-fracture row, S32.511, matches a billed pelvic fracture code, S32.511A, so it is plausible.)

The full per-code table is in `rag_final_run/report/final_output.md` (and `final_output.csv`).

## 6. Limits to state

- 15 admissions and a single run, so no statistical claims.
- Only the first principal code of each diagnosis row is checked, and at most 6 rows per admission.
- The checker sees Naren's extracted evidence phrases, not the note text.
- The billing record is the only reference standard and it is incomplete (and some billed codes are older, less specific versions).
- Fixes made after seeing results are post-hoc.
- On these patients, adding Guidelines text did not improve accuracy. What it adds is rule citations for every decision.

## 7. Files

Scripts: `notebooks/` in the repo (see `README_guideline_rag.md` there). Generated indexes and result folders stay outside the repo.
