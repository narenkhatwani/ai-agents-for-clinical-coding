"""Guideline-grounded review of ICD-10-CM codes, using Naren's LLM layer (pipeline.get_llm_config / call_llm_json).

For each candidate code: retrieve Guidelines rules (guideline_rag), gather exact Tabular List facts (tabular_list),
ask the LLM for a structured verdict that must cite retrieved rule ids, validate it, and look up codes for any
condition the LLM names in plain English (the LLM never invents codes).

    $env:OPENROUTER_API_KEY = "<your key>"      # PowerShell; keeps notebooks/settings.json untouched
    python guideline_check.py --admission patient_records/patient_x/admissions/hadm_y \
        --index ../../icd10cm_sources/guideline_index --models ../../icd10cm_sources/models \
        --tabular ../../icd10cm_sources/tabular_FY2026.json --max-cases 2

The LLM is whatever Naren's pipeline.py is configured for (get_llm_config); nothing in pipeline.py is modified.
--show-prompt prints the first case's prompt without calling any LLM.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from guideline_rag import Embedder, GuidelineIndex, cases_from_stage8, make_bm25, retrieve_for_case, tokenize
from tabular_list import TabularList, dotted, nodot

SYSTEM_PROMPT = (
    "You are a clinical coding reviewer. You check one ICD-10-CM code against the official Coding Guidelines "
    "excerpts and Tabular List facts you are given. Use only the provided material. Never invent ICD-10-CM codes: "
    "you may only propose a code from ALLOWED_CODES, or describe a different condition in plain English. "
    "Return complete valid JSON only."
)

SCHEMA = """{
  "rules_considered": [{"rule": "rule id as shown in RULES", "applies": "yes | no | unclear", "why": "at most 25 words, quote the case evidence"}],
  "verdict": "keep | flag | replace",
  "proposed_code": "a code from ALLOWED_CODES, only when verdict is replace, otherwise empty",
  "suggested_condition": "a different condition or complication that may need its own code, or a companion code that a code-first / use-additional-code note or sequencing rule requires, in formal ICD-10-CM wording (hemorrhage not bleeding, postprocedural not post-surgery), otherwise empty",
  "cited_rules": ["rule ids exactly as shown in brackets, e.g. I.B.16"],
  "reason": "at most 40 words",
  "needs_provider_query": false,
  "query_to_provider": "the question to ask the provider, otherwise empty"
}"""

TASK = """First, go through EVERY rule shown in RULES and fill rules_considered: does that rule apply to this case, based on the case evidence? For a rule about complications of care, ask: was a procedure performed, and does the evidence document that the condition resulted from it?
Then decide whether the candidate code itself is appropriate for this case.
- keep: nothing in the rules or Tabular facts contradicts the code. If the code is correct but a rule or a code-first / use-additional-code note also requires ANOTHER code (a companion or sequencing code), still choose keep and name that other condition in suggested_condition.
- flag: a rule suggests the candidate code itself may be wrong, too nonspecific or unsupported, but the documentation does not settle it (set needs_provider_query when the rules say to query the provider).
- replace: the rules and facts clearly support a different code from ALLOWED_CODES.
Every flag or replace must cite at least one rule id that appears in RULES. Do not cite rules that are not shown.
If a rule says coding depends on provider documentation that the case evidence does not show, prefer flag over replace."""


SYSTEM_PROMPT_CONTROL = (
    "You are a clinical coding reviewer. You check one ICD-10-CM code using the Tabular List facts you are given "
    "and your own knowledge of ICD-10-CM coding conventions. Never invent ICD-10-CM codes: "
    "you may only propose a code from ALLOWED_CODES, or describe a different condition in plain English. "
    "Return complete valid JSON only."
)

TASK_CONTROL = """No Coding Guidelines excerpts are provided for this review. Leave rules_considered and cited_rules as empty lists.
Decide whether the candidate code itself is appropriate for this case, using the case evidence, the Tabular List facts and your own knowledge of coding conventions.
- keep: nothing in the facts or your knowledge contradicts the code. If the code is correct but a code-first / use-additional-code note or sequencing convention also requires ANOTHER code (a companion or sequencing code), still choose keep and name that other condition in suggested_condition.
- flag: the candidate code itself may be wrong, too nonspecific or unsupported, but the documentation does not settle it (set needs_provider_query when a provider query is appropriate).
- replace: the facts clearly support a different code from ALLOWED_CODES.
If coding depends on provider documentation that the case evidence does not show, prefer flag over replace."""


class TitleSearch:
    """Hybrid search (BM25 + MiniLM in Chroma) over Tabular List code titles for a plain-English condition."""

    def __init__(self, tabular, embedder, chroma_dir, batch=256):
        import chromadb
        self.embedder = embedder
        self.entries = [(n["code"], n["title"]) for n in tabular.nodes.values()
                        if n["level"] == "diag" and not n["children"]]
        self.pos = {code: i for i, (code, _) in enumerate(self.entries)}
        self.bm25 = make_bm25([tokenize(t) for _, t in self.entries])
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            self.col = client.get_collection("icd_titles")
            if self.col.count() != len(self.entries):
                raise ValueError("stale")
        except Exception:
            try:
                client.delete_collection("icd_titles")
            except Exception:
                pass
            self.col = client.create_collection("icd_titles", configuration={"hnsw": {"space": "cosine"}})
            print(f"Embedding {len(self.entries):,} ICD code titles once (cached in Chroma)...")
            for i in range(0, len(self.entries), 2000):
                part = self.entries[i:i + 2000]
                vecs = np.vstack([embedder.encode([t for _, t in part[j:j + batch]]) for j in range(0, len(part), batch)])
                self.col.add(ids=[c for c, _ in part], embeddings=vecs.tolist(), documents=[t for _, t in part])

    def search(self, text, k=5, pool=30, rrf_k=60):
        bm = self.bm25.scores(tokenize(text))
        lists = [[i for i in np.argsort(-bm)[:pool] if bm[i] > 0]]
        res = self.col.query(query_embeddings=[self.embedder.encode([text])[0].tolist()], n_results=pool)
        lists.append([self.pos[c] for c in res["ids"][0]])
        fused = {}
        for lst in lists:
            for rank, i in enumerate(lst):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{"code": self.entries[i][0], "title": self.entries[i][1], "score": round(fused[i], 4)} for i in top]


_COMPLICATION_TITLE = re.compile(
    r"postprocedural|post-procedural|intraoperative|following (a |an )?(procedure|surgery|operation)|complications? of|"
    r"(due to|from) (an? )?(implant|device|graft|catheter|prosthe|transplant)|surgical site", re.I)


def is_complication_code(tabular, code):
    """True when the candidate already IS a complication-of-care code (T80-T88, 'postprocedural ...', 'following a
    procedure'), so the 'a complication rule applies' consistency check must not push it towards flag."""
    rec = tabular.record(code)
    if rec is None:
        return False
    stem = nodot(code)[:3].upper()
    return ("T80" <= stem <= "T88") or bool(_COMPLICATION_TITLE.search(rec["title"]))


_COND_NOISE = re.compile(r"\b(initial|subsequent) encounter(?: for [a-z ]*?fracture)?\b|\bsequela\b|\bsequenced\b.*$", re.I)


def condition_queries(text):
    """Plain-English suggested condition -> up to two clean title-search queries. Drops parentheticals, encounter
    wording and 'with additional code ...' tails that dilute the search (e.g. 'Adverse effect of anticoagulants
    (warfarin), initial encounter, with gross hematuria sequenced first' -> ['Adverse effect of anticoagulants',
    'gross hematuria']). A bare ' with ' is kept: 'Type 2 diabetes mellitus with diabetic chronic kidney disease'.
    (Searching the dropped parentheticals as extra queries was tried and made results worse: '(metformin)' pulled in
    W86.8, '(Z85 category)' a blindness code. Use parenthetical_terms() for a wider candidate pool instead.)"""
    t = re.sub(r"\([^)]*\)", " ", text)
    t = _COND_NOISE.sub(" ", t)
    t = re.sub(r"(\s*,)+", ",", re.sub(r"\s+", " ", t)).strip(" ,;")
    parts = [p.strip(" ,;") for p in re.split(r";|,\s*with\s+|,\s*and\s+|,?\s+plus\s+", t, flags=re.I)]
    return [p for p in parts if len(p) > 2][:2] or [text]


def parenthetical_terms(text):
    """Text inside parentheses that is not a list of examples ('e.g. ...'): it often holds the specific term, e.g.
    'Underlying systemic infection (sepsis due to unspecified organism) sequenced first' -> ['sepsis due to unspecified organism']."""
    return [a.strip() for a in re.findall(r"\(([^)]*)\)", text)
            if len(a.strip()) > 2 and not re.match(r"(e\.g\.|for example|such as)", a.strip(), re.I)]


def map_condition(title_search, text, anchor, k=5):
    """Codes for a suggested condition: each cleaned query is searched with and without the candidate's body-system
    anchor; the ranked lists are interleaved and de-duplicated."""
    lists = []
    for q in condition_queries(text):
        lists.append(title_search.search(f"{q} {anchor}".strip()))
        if anchor:
            lists.append(title_search.search(q))
    merged, seen = [], set()
    for rank in range(max(len(x) for x in lists)):
        for lst in lists:
            if rank < len(lst) and lst[rank]["code"] not in seen:
                seen.add(lst[rank]["code"])
                merged.append(lst[rank])
    return merged[:k]


_NO_BODY_SYSTEM = re.compile(r"^(symptoms, signs|factors influencing|external causes|codes for special purposes)", re.I)


def body_system(tabular, code):
    """'genitourinary system' for N93.9: the Tabular List chapter's body system, used to anchor title searches."""
    node, _ = tabular.resolve(code)
    ch = next((n for n in tabular.ancestry(node) if n["level"] == "chapter"), None) if node else None
    if ch is None or _NO_BODY_SYSTEM.search(ch["title"]):
        return ""   # symptom / Z / external-cause / special-purpose chapters name no body system; a companion code lies elsewhere
    title = re.sub(r"\s*\(.*\)$", "", ch["title"]).lower()
    return re.sub(r"^(diseases|disorders) of (the )?", "", title)


def tabular_findings(tabular, code, admission_codes):
    rec = tabular.record(code)
    if rec is None:
        return {"found": False}
    conflicts = [c for c in tabular.excludes_conflicts(admission_codes)
                 if nodot(code) in (nodot(c["code"]), nodot(c["conflicts_with"]))]
    notes = {k: [e["text"][:160] for e in v][:4] for k, v in rec["notes"].items()
             if k in ("excludes1", "codeFirst", "useAdditionalCode", "codeAlso")}
    seventh = ("complete (" + rec["seventh_char"] + ")" if rec["seventh_char"] else "MISSING, the code is incomplete") if rec["needs_7th"] else ""
    return {"found": True, "title": rec["title"], "needs_7th": rec["needs_7th"], "seventh": seventh, "notes": notes,
            "excludes1_conflicts": conflicts, "more_specific": tabular.more_specific(code)[:8]}


def est_tokens(text):
    return int(len(text) / 3.6)


def ensure_ca_bundle(path):
    """Windows fix for CERTIFICATE_VERIFY_FAILED: on some networks (antivirus/proxy) HTTPS is re-signed with a
    certificate Windows trusts but Python's certifi list lacks. Builds certifi + the Windows trusted certificates
    into one file and points requests at it. Verification stays ON; nothing in pipeline.py is touched."""
    import os
    if os.environ.get("REQUESTS_CA_BUNDLE") or sys.platform != "win32":
        return
    p = Path(path)
    if not p.exists():
        import ssl
        import certifi
        pems = [ssl.DER_cert_to_PEM_cert(der) for store in ("ROOT", "CA") for der, enc, trust in ssl.enum_certificates(store)
                if enc == "x509_asn" and (trust is True or "1.3.6.1.5.5.7.3.1" in trust)]
        p.write_text(Path(certifi.where()).read_text() + "\n" + "\n".join(pems), encoding="utf-8")
    os.environ["REQUESTS_CA_BUNDLE"] = str(p)


def build_prompt(case, rules, facts, allowed, budget_tokens, previous_problems=None, control=False):
    system = SYSTEM_PROMPT_CONTROL if control else SYSTEM_PROMPT

    def render(rule_subset):
        lines = [f"CASE\nDiagnosis: {case['diagnosis']}\nCandidate code: {case['code']} - {case['code_title']}"]
        if case.get("evidence"):
            lines.append("Evidence from the note: " + "; ".join(case["evidence"][:4]))
        if case.get("procedures"):
            lines.append("Procedures this stay: " + "; ".join(case["procedures"][:5]))
        t = ["\nTABULAR LIST FACTS (exact, from the code book)"]
        if facts.get("found"):
            seventh = f"7th character required, {facts['seventh']}" if facts["needs_7th"] else f"needs 7th character: {facts['needs_7th']}"
            t.append(f"- Title: {facts['title']}; {seventh}")
            for k, v in facts["notes"].items():
                t.append(f"- {k}: " + " | ".join(v))
            for c in facts["excludes1_conflicts"]:
                t.append(f"- EXCLUDES1 CONFLICT with {c['conflicts_with'] if nodot(c['code']) == nodot(case['code']) else c['code']}: {c['note']}")
            if facts["more_specific"]:
                t.append("- More specific options: " + "; ".join(f"{m['code']} {m['title'][:60]}" for m in facts["more_specific"]))
        else:
            t.append("- code not found in the Tabular List")
        if control:
            r = ["\nRULES: none provided for this review"]
        else:
            r = ["\nRULES (official Coding Guidelines excerpts; cite by the bracketed id)"]
            r += [x["text"][:2800] for x in rule_subset] or ["(no relevant rule was retrieved)"]
        tail = [f"\nALLOWED_CODES: {', '.join(allowed)}", f"\nTASK\n{TASK_CONTROL if control else TASK}",
                f"\nOUTPUT JSON SCHEMA\n{SCHEMA}"]
        if previous_problems:
            tail.append("\nYour previous answer had these problems, fix them:\n- " + "\n- ".join(previous_problems))
        return "\n".join(lines + t + r + tail)

    used = list(rules)
    prompt = render(used)
    while est_tokens(system + prompt) > budget_tokens and len(used) > 1:
        used.pop()
        prompt = render(used)
    return prompt, used


def validate(out, shown_ids, allowed, code, groups=None, control=False, candidate_is_complication=False):
    """groups: trigger label -> rule ids retrieved for it; at least one rule per trigger must be considered.
    control=True: no rules were shown, so no rule citations are required or allowed.
    candidate_is_complication: the candidate already is a complication-of-care code, so 'keep' is allowed even when the
    complication rule applies (the rule is what justifies the code)."""
    problems = []
    if control:
        v = str(out.get("verdict", "")).strip().lower()
        if v not in ("keep", "flag", "replace"):
            problems.append(f"verdict must be keep, flag or replace (got {out.get('verdict')!r})")
        prop = str(out.get("proposed_code") or "").strip()
        if v == "replace" and (not prop or nodot(prop) not in {nodot(a) for a in allowed}):
            problems.append(f"proposed_code must be one of ALLOWED_CODES (got {prop!r})")
        elif v != "replace" and prop and nodot(prop) != nodot(code):
            problems.append("proposed_code must be empty unless verdict is replace")
        if not str(out.get("reason") or "").strip():
            problems.append("reason is empty")
        return problems
    groups = groups or {}
    complication_ids = groups.get("complication", set())
    considered = {str(r.get("rule", "")).strip().strip("[]"): str(r.get("applies", "")).strip().lower()
                  for r in (out.get("rules_considered") or []) if isinstance(r, dict)}
    for label, ids in groups.items():
        if ids and not any(i in considered for i in ids):
            problems.append(f"rules_considered must address the {label} rule(s) {sorted(ids)}")
    v = str(out.get("verdict", "")).strip().lower()
    if v == "keep" and not candidate_is_complication and any(considered.get(i) in ("yes", "unclear") for i in complication_ids):
        problems.append("a complication-of-care rule applies or is unclear, so the verdict cannot be keep: "
                        "use flag (query the provider) or replace")
    if v not in ("keep", "flag", "replace"):
        problems.append(f"verdict must be keep, flag or replace (got {out.get('verdict')!r})")
    cited = [str(c).strip().strip("[]") for c in (out.get("cited_rules") or [])]
    bad = [c for c in cited if c not in shown_ids]
    if bad:
        problems.append(f"cited rule ids not shown in RULES: {bad}")
    if v in ("flag", "replace") and not [c for c in cited if c in shown_ids]:
        problems.append("flag/replace must cite at least one rule id shown in RULES")
    prop = str(out.get("proposed_code") or "").strip()
    if v == "replace":
        if not prop or nodot(prop) not in {nodot(a) for a in allowed}:
            problems.append(f"proposed_code must be one of ALLOWED_CODES (got {prop!r})")
    elif prop and nodot(prop) != nodot(code):
        problems.append("proposed_code must be empty unless verdict is replace")
    if not str(out.get("reason") or "").strip():
        problems.append("reason is empty")
    return problems


def prepare(case, index, tabular, admission_codes, budget_tokens, max_rules=4, control=False):
    """control=True skips retrieval entirely: the LLM gets the same case and Tabular facts but no Guidelines text."""
    if control:
        results = []
    else:
        results = retrieve_for_case(index, case, tabular)
        results = sorted(results, key=lambda r: set(r["triggers"]) == {"chapter+case"})[:max_rules]
    facts = tabular_findings(tabular, case["code"], admission_codes)
    allowed = [dotted(case["code"])] + [m["code"] for m in facts.get("more_specific", [])]
    prompt, used = build_prompt(case, results, facts, allowed, budget_tokens, control=control)
    return results, facts, allowed, prompt, used


def check_case(case, index, tabular, title_search, admission_codes, config, call_llm_json, budget_tokens, max_rules=4,
               control=False):
    t0 = time.time()
    system = SYSTEM_PROMPT_CONTROL if control else SYSTEM_PROMPT
    results, facts, allowed, prompt, used = prepare(case, index, tabular, admission_codes, budget_tokens, max_rules, control)
    def groups_of(rules):
        g = {}
        for r in rules:
            for t in r["triggers"]:
                if t != "chapter+case":
                    g.setdefault(t, set()).add(r["parent_id"])
        return g

    shown = {r["parent_id"] for r in used}
    is_complication = is_complication_code(tabular, case["code"])
    out, problems, attempts = {}, [], 0
    for attempts in (1, 2):
        out = call_llm_json(system, prompt, config)
        problems = validate(out, shown, allowed, case["code"], groups_of(used), control, is_complication)
        if not problems:
            break
        prompt, used = build_prompt(case, results, facts, allowed, budget_tokens, previous_problems=problems, control=control)
        shown = {r["parent_id"] for r in used}
    suggestion = str(out.get("suggested_condition") or "").strip()
    return {"case": case, "retrieved": [{"id": r["parent_id"], "triggers": r["triggers"]} for r in results],
            "rules_shown": sorted(shown), "tabular": {k: v for k, v in facts.items() if k != "more_specific"},
            "allowed_codes": allowed, "llm": out, "valid": not problems, "problems": problems, "attempts": attempts,
            "codes_for_suggested_condition": (
                map_condition(title_search, suggestion, body_system(tabular, case["code"])) if suggestion else []),
            "prompt_tokens_est": est_tokens(system + prompt), "seconds": round(time.time() - t0, 1),
            "model": getattr(config, "model", None), "arm": "control" if control else "rules"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--admission", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--tabular", required=True)
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: pipeline.ICD_CONFIRM_TEMPERATURE, the value Naren's Stage 9 uses")
    ap.add_argument("--max-cases", type=int, default=3)
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt for the first case and exit (no LLM call)")
    ap.add_argument("--out")
    a = ap.parse_args()
    ensure_ca_bundle(Path(a.index).parent / "ca_bundle.pem")

    from dataclasses import replace
    from pipeline import ICD_CONFIRM_TEMPERATURE, call_llm_json, get_llm_config
    config = replace(get_llm_config(),
                     temperature=float(ICD_CONFIRM_TEMPERATURE if a.temperature is None else a.temperature))
    budget = 12000
    print(f"LLM (Naren's settings): {config.api_label}/{config.model} temperature={config.temperature} zdr={config.openrouter_zdr}")
    if not a.show_prompt and not config.resolved_api_key():
        sys.exit(f"No API key. In this terminal run:  $env:{config.api_key_env} = '<your OpenRouter key>'   "
                 "(an environment variable keeps notebooks/settings.json untouched)")

    tabular = TabularList.load(a.tabular)
    embedder = Embedder(Path(a.models) / "minilm")
    title_search = TitleSearch(tabular, embedder, Path(a.index) / "chroma")
    index = GuidelineIndex.load(a.index, embedder, None)
    cases = cases_from_stage8(a.admission)
    admission_codes = [c["code"] for c in cases]
    if a.show_prompt:
        _, _, _, prompt, used = prepare(cases[0], index, tabular, admission_codes, budget)
        print(f"\n--- SYSTEM ---\n{SYSTEM_PROMPT}\n\n--- USER ---\n{prompt}\n\n(~{est_tokens(SYSTEM_PROMPT + prompt)} tokens, rules shown: {[r['parent_id'] for r in used]})")
        return
    records = []
    for case in cases[:a.max_cases]:
        print(f"\n=== rank {case['rank']} {case['diagnosis']} -> {case['code']}")
        rec = check_case(case, index, tabular, title_search, admission_codes, config, call_llm_json, budget)
        records.append(rec)
        llm = rec["llm"]
        print(f"  retrieved: {[(r['id'], '+'.join(r['triggers'])) for r in rec['retrieved']]}")
        for rc in llm.get("rules_considered") or []:
            print(f"    rule {rc.get('rule')}: applies={rc.get('applies')} - {str(rc.get('why'))[:90]}")
        print(f"  verdict: {llm.get('verdict')} | cited: {llm.get('cited_rules')} | provider query: {llm.get('needs_provider_query')}")
        print(f"  reason: {llm.get('reason')}")
        if llm.get("suggested_condition"):
            print(f"  suggested condition: {llm['suggested_condition']}")
            print(f"  -> candidate codes: {[(c['code'], c['title'][:50]) for c in rec['codes_for_suggested_condition'][:3]]}")
        print(f"  valid: {rec['valid']} {rec['problems'] or ''} | attempts: {rec['attempts']} | ~{rec['prompt_tokens_est']} prompt tokens | {rec['seconds']}s")
    if a.out:
        Path(a.out).write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
