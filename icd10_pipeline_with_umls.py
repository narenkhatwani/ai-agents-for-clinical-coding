"""
ICD-10 Multi-Agent Pipeline — with UMLS Normalization + MCP ICD-10 Lookup
==========================================================================
Pipeline stages:
  Agent 1  - Evidence Extractor       : pulls findings from clinical note
  [NEW] UMLS Normalizer               : maps clinical shorthand to canonical terms,
                                        synonyms, CUI codes, and cross-terminology
  Agent 2  - Code Candidate Generator : MCP-grounded ICD-10 code lookup
  Agent 3  - Validator                : audits codes against guidelines
  Agent 4  - Reconciler               : produces final claim-ready code list

Setup (one time):
  1. Get free UMLS API key at: https://uts.nlm.nih.gov
  2. pip install requests umls-python-client
  3. Make sure Node.js + Ollama + qwen2.5:7b are installed
  4. Set your UMLS API key below

Run:
  python icd10_pipeline_with_umls.py
"""

import requests
import json
import subprocess
import time
from typing import Any

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "qwen2.5:7b"
UMLS_API_KEY = "YOUR_UMLS_API_KEY_HERE"   # ← paste your key here


# ── Core Ollama caller ────────────────────────────────────────────────────────
def call_ollama(prompt: str, agent_name: str) -> str:
    print(f"\n  [Agent: {agent_name}] thinking...", end="", flush=True)
    start = time.time()
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False
    })
    response.raise_for_status()
    raw = response.json()["response"]
    print(f" done ({time.time() - start:.1f}s)")
    return raw


def parse_json(raw: str) -> Any:
    """Extract and parse the first JSON object found in a string."""
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response:\n{raw}")
    return json.loads(raw[start:end])


# ═══════════════════════════════════════════════════════════════════════════
# UMLS NORMALIZER
# Sits between Agent 1 and Agent 2.
# Takes raw clinical terms and returns canonical names, synonyms,
# CUI codes, and cross-terminology mappings (ICD-10, SNOMED, LOINC).
# ═══════════════════════════════════════════════════════════════════════════

class UMLSNormalizer:
    """
    Calls the UMLS REST API to normalize clinical terms.

    Why this matters:
      - "AFib" → canonical: "Atrial fibrillation" → better MCP search results
      - "T2DM" → canonical: "Type 2 diabetes mellitus" → exact ICD-10 match
      - "DKD stage 3" → maps to SNOMED C0403447 → maps to ICD-10 N18.3

    UMLS API docs: https://documentation.uts.nlm.nih.gov/rest/home.html
    Free API key:  https://uts.nlm.nih.gov
    """

    BASE_URL  = "https://uts-ws.nlm.nih.gov/rest"
    AUTH_URL  = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    TGT_URL   = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
    SERVICE   = "http://umlsks.nlm.nih.gov"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._tgt    = None   # Ticket-Granting Ticket (session token)

    # ── Authentication ────────────────────────────────────────────────────

    def _get_tgt(self) -> str:
        """
        Get a Ticket-Granting Ticket (TGT) from UMLS.
        The TGT is a session token that lasts 8 hours.
        We cache it in self._tgt so we don't re-authenticate every call.
        """
        if self._tgt:
            return self._tgt

        r = requests.post(
            self.AUTH_URL,
            data={"apikey": self.api_key},
            headers={"Content-type": "application/x-www-form-urlencoded"}
        )
        # UMLS returns the TGT URL inside an HTML form action attribute
        # We extract it by finding the action="..." in the response text
        tgt_url = r.text.split('action="')[1].split('"')[0]
        self._tgt = tgt_url
        return self._tgt

    def _get_ticket(self) -> str:
        """
        Get a single-use Service Ticket (ST) from the TGT.
        UMLS requires a fresh ST for every API call.
        Each ST is valid for only one request.
        """
        tgt = self._get_tgt()
        r   = requests.post(
            tgt,
            data={"service": self.SERVICE},
            headers={"Content-type": "application/x-www-form-urlencoded"}
        )
        return r.text.strip()

    # ── Core search ───────────────────────────────────────────────────────

    def search_concept(self, term: str) -> dict:
        """
        Search UMLS for a clinical term.
        Returns the top matching concept with CUI, canonical name,
        semantic type, and source vocabulary mappings.

        CUI = Concept Unique Identifier — UMLS's internal ID for a concept.
        For example: "Atrial fibrillation" = CUI C0004238 across all terminologies.
        """
        try:
            ticket = self._get_ticket()
            r = requests.get(
                f"{self.BASE_URL}/search/current",
                params={
                    "string":      term,
                    "ticket":      ticket,
                    "pageSize":    5,
                    "returnIdType": "concept"
                }
            )
            r.raise_for_status()
            data    = r.json()
            results = data.get("result", {}).get("results", [])

            if not results or results[0].get("ui") == "NONE":
                return {"term": term, "normalized": False, "canonical": term}

            # Take the top result
            top = results[0]
            return {
                "term":          term,
                "normalized":    True,
                "canonical":     top.get("name", term),
                "cui":           top.get("ui"),
                "source":        top.get("rootSource", ""),
                "uri":           top.get("uri", "")
            }

        except Exception as e:
            # If UMLS is unreachable or key is wrong, fail gracefully
            # and return the original term so the pipeline can still run
            print(f"\n    [UMLS] Warning: could not normalize '{term}': {e}")
            return {"term": term, "normalized": False, "canonical": term}

    def get_synonyms(self, cui: str) -> list:
        """
        Get all synonym names (atoms) for a CUI.
        This is what gives us "AFib" = "atrial fibrillation" = "auricular fibrillation".
        Atoms are individual name strings from different source vocabularies.
        """
        try:
            ticket = self._get_ticket()
            r = requests.get(
                f"{self.BASE_URL}/content/current/CUI/{cui}/atoms",
                params={
                    "ticket":   ticket,
                    "pageSize": 20
                }
            )
            r.raise_for_status()
            atoms    = r.json().get("result", [])
            # Extract unique names, filter out very long ones
            synonyms = list({
                a.get("name", "") for a in atoms
                if a.get("name") and len(a.get("name", "")) < 100
            })
            return synonyms[:10]   # return top 10 synonyms

        except Exception:
            return []

    def get_icd10_mapping(self, cui: str) -> list:
        """
        Get ICD-10-CM codes mapped to this concept via UMLS crosswalk.
        This lets us go directly from a UMLS CUI to ICD-10 codes,
        bypassing the need for a text search in Agent 2.
        """
        try:
            ticket = self._get_ticket()
            r = requests.get(
                f"{self.BASE_URL}/content/current/CUI/{cui}/atoms",
                params={
                    "ticket":    ticket,
                    "sabs":      "ICD10CM",   # filter to ICD-10-CM source only
                    "pageSize":  10
                }
            )
            r.raise_for_status()
            atoms = r.json().get("result", [])
            codes = []
            for atom in atoms:
                code = atom.get("code", "")
                name = atom.get("name", "")
                # ICD-10 codes have a specific format: letter + digits + optional dot
                if code and len(code) >= 3:
                    codes.append({
                        "code":        code.replace("ICD10CM/", ""),
                        "description": name,
                        "source":      "UMLS crosswalk"
                    })
            return codes

        except Exception:
            return []

    def get_snomed_mapping(self, cui: str) -> list:
        """
        Get SNOMED CT codes for this concept.
        Useful for MIMIC data which uses SNOMED codes in the EHR,
        and for Dr. Wang's knowledge graph phase (SNOMED has richer
        clinical relationships than ICD-10).
        """
        try:
            ticket = self._get_ticket()
            r = requests.get(
                f"{self.BASE_URL}/content/current/CUI/{cui}/atoms",
                params={
                    "ticket":   ticket,
                    "sabs":     "SNOMEDCT_US",
                    "pageSize": 5
                }
            )
            r.raise_for_status()
            atoms = r.json().get("result", [])
            codes = []
            for atom in atoms:
                code = atom.get("code", "")
                name = atom.get("name", "")
                if code:
                    codes.append({
                        "snomed_code": code.replace("SNOMEDCT_US/", ""),
                        "description": name
                    })
            return codes

        except Exception:
            return []

    # ── Main normalize function ───────────────────────────────────────────

    def normalize(self, term: str) -> dict:
        """
        Full normalization pipeline for one clinical term.
        Returns everything Agent 2 needs: canonical name, synonyms,
        ICD-10 hints, SNOMED codes.
        """
        print(f"    [UMLS] Normalizing '{term}'...", end="", flush=True)

        # Step 1: Find the canonical concept and CUI
        concept = self.search_concept(term)

        if not concept.get("normalized"):
            print(f" not found, using original term")
            return {
                "original_term": term,
                "canonical":     term,
                "normalized":    False,
                "synonyms":      [],
                "icd10_hints":   [],
                "snomed_codes":  [],
                "cui":           None
            }

        cui       = concept.get("cui")
        canonical = concept.get("canonical", term)

        # Step 2: Get synonyms (for better MCP search)
        synonyms = self.get_synonyms(cui) if cui else []

        # Step 3: Get direct ICD-10 mappings (may bypass need for MCP search)
        icd10_hints = self.get_icd10_mapping(cui) if cui else []

        # Step 4: Get SNOMED codes (for KG phase later)
        snomed_codes = self.get_snomed_mapping(cui) if cui else []

        print(f" → '{canonical}' (CUI: {cui}, {len(icd10_hints)} ICD-10 hints)")

        return {
            "original_term": term,
            "canonical":     canonical,
            "normalized":    True,
            "cui":           cui,
            "synonyms":      synonyms,
            "icd10_hints":   icd10_hints,    # direct ICD-10 codes from UMLS
            "snomed_codes":  snomed_codes    # for Phase 2 KG integration
        }

    def normalize_all_findings(self, findings: list) -> list:
        """
        Normalize all findings from Agent 1.
        Adds normalization data to each finding dict.
        Returns enriched findings list.
        """
        enriched = []
        for finding in findings:
            term      = finding.get("term", "")
            certainty = finding.get("certainty", "confirmed")

            # Don't waste API calls on ruled-out findings
            if certainty == "ruled_out":
                enriched.append({**finding, "normalized": False, "canonical": term})
                continue

            norm_data = self.normalize(term)

            # Merge normalization data into the finding
            enriched.append({
                **finding,                              # original Agent 1 fields
                "canonical":    norm_data["canonical"], # normalized search term
                "cui":          norm_data.get("cui"),
                "synonyms":     norm_data.get("synonyms", []),
                "icd10_hints":  norm_data.get("icd10_hints", []),   # direct codes
                "snomed_codes": norm_data.get("snomed_codes", [])   # for KG phase
            })

        return enriched


# ═══════════════════════════════════════════════════════════════════════════
# MCP ICD-10 CLIENT (unchanged from previous version)
# ═══════════════════════════════════════════════════════════════════════════

class ICD10MCPClient:
    def __init__(self):
        self.process = None
        self._msg_id = 0
        self._start()

    def _start(self):
        print("\n  [MCP] Starting ICD-10 server...", end="", flush=True)
        self.process = subprocess.Popen(
            ["cmd", "/c", "npx", "-y", "@findicd10/mcp"],  # Windows fix
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self._send({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {},
                "clientInfo":      {"name": "icd10-pipeline", "version": "1.0"}
            }
        })
        self._read()
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        print(" ready ✓")

    def _send(self, msg: dict):
        self.process.stdin.write(json.dumps(msg) + "\n")
        self.process.stdin.flush()

    def _read(self) -> dict:
        while True:
            line = self.process.stdout.readline()
            if line.strip():
                return json.loads(line)

    def _call(self, tool: str, args: dict) -> str:
        self._msg_id += 1
        self._send({
            "jsonrpc": "2.0", "id": self._msg_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args}
        })
        response = self._read()
        content  = response.get("result", {}).get("content", [])
        return content[0].get("text", "") if content else ""

    def search(self, query: str, limit: int = 8) -> list:
        raw = self._call("search_icd10_codes", {"query": query, "limit": limit})
        try:    return json.loads(raw)
        except: return [{"raw": raw}]

    def get_code(self, code: str) -> dict:
        raw = self._call("get_icd10_code", {"code": code})
        try:    return json.loads(raw)
        except: return {"raw": raw}

    def close(self):
        if self.process:
            self.process.terminate()


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1: Evidence Extractor (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def agent_evidence_extractor(clinical_note: str) -> dict:
    prompt = f"""You are a clinical NLP expert. Read the clinical note and extract
every clinical finding that should be coded.

Extract: confirmed diagnoses, chronic conditions, symptoms not part of a confirmed
diagnosis, relevant findings (BMI, labs), procedures performed.
Do NOT include: medications, normal findings, family history.

Respond ONLY in this JSON format:
{{
  "findings": [
    {{
      "term": "clinical term as written",
      "category": "diagnosis | symptom | chronic_condition | finding | procedure",
      "context": "brief quote from the note",
      "certainty": "confirmed | suspected | ruled_out | history"
    }}
  ]
}}

Clinical Note:
{clinical_note}
"""
    raw = call_ollama(prompt, "Evidence Extractor")
    return parse_json(raw)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2: Code Candidate Generator
# NOW uses UMLS-normalized terms + direct ICD-10 hints from UMLS crosswalk
# ═══════════════════════════════════════════════════════════════════════════

def agent_code_candidate_generator(enriched_findings: list, mcp: ICD10MCPClient) -> dict:
    """
    Enhanced Agent 2 that uses UMLS normalization data:
      - Uses canonical term (not shorthand) for MCP search → better results
      - If UMLS already provided direct ICD-10 hints, passes them to Qwen
        alongside MCP results → Qwen has more verified options to choose from
      - Includes SNOMED codes in output for future KG phase
    """
    print("\n  [Agent: Code Candidate Generator (UMLS + MCP grounded)]")
    candidates = []

    for finding in enriched_findings:
        term      = finding.get("term", "")
        certainty = finding.get("certainty", "confirmed")
        canonical = finding.get("canonical", term)  # UMLS canonical name
        cui       = finding.get("cui", None)
        icd10_hints = finding.get("icd10_hints", [])  # direct UMLS→ICD-10 mapping
        snomed_codes = finding.get("snomed_codes", [])

        if certainty == "ruled_out":
            continue

        print(f"    Coding: '{term}' → '{canonical}'...", end="", flush=True)

        # ── Step A: MCP search using CANONICAL term (not shorthand) ───────
        # This is the key improvement: "AFib" → search "Atrial fibrillation"
        search_results = mcp.search(canonical, limit=8)

        # Format MCP results
        mcp_text = ""
        for r in search_results:
            code     = r.get("code", r.get("raw", ""))
            desc     = r.get("description", r.get("long_description", ""))
            billable = r.get("is_billable", True)
            mcp_text += f"  - {code}: {desc} (billable: {billable})\n"

        # ── Step B: Also include direct UMLS→ICD-10 hints if available ────
        # These are direct crosswalk mappings from UMLS — very reliable
        hints_text = ""
        if icd10_hints:
            hints_text = "\nDirect ICD-10 mappings from UMLS crosswalk (highly reliable):\n"
            for h in icd10_hints[:5]:
                hints_text += f"  * {h.get('code')}: {h.get('description')}\n"

        if not mcp_text:
            mcp_text = "  No results found in ICD-10 database."

        # ── Step C: Qwen selects best code from ALL verified sources ───────
        prompt = f"""You are a certified medical coder (CCS, CPC).

Clinical finding: "{term}"
Canonical UMLS term: "{canonical}"
UMLS CUI: {cui or 'not found'}
Certainty: {certainty}

ICD-10-CM database search results (from MCP):
{mcp_text}
{hints_text}

Select the MOST SPECIFIC and APPROPRIATE ICD-10-CM code.
Rules:
- Prefer UMLS crosswalk codes when available — they are directly mapped
- Use combination codes when they exist
- Do not code symptoms integral to a confirmed diagnosis
- Use highest specificity (most digits)

Respond ONLY in this JSON format:
{{
  "selected_code": "the best ICD-10-CM code",
  "description": "official description of selected code",
  "source": "mcp | umls_crosswalk | both",
  "confidence": "high | medium | low",
  "reasoning": "why this code is most appropriate",
  "alternative_code": "second best code or null",
  "coding_notes": "guideline notes or null"
}}"""

        raw    = call_ollama(prompt, f"Code Selector [{canonical}]")
        result = parse_json(raw)

        # ── Step D: Final MCP validation of selected code ──────────────────
        selected     = result.get("selected_code", "")
        code_details = mcp.get_code(selected) if selected else {}

        verified_description = (
            code_details.get("long_description") or
            code_details.get("description") or
            result.get("description", "")
        )
        is_billable = code_details.get("is_billable", True)

        confidence = result.get("confidence", "medium")
        conf_icon  = "🟢" if confidence == "high" else "🟡" if confidence == "medium" else "🔴"
        print(f" → {selected} {conf_icon} ({confidence} confidence, {'✓' if is_billable else '⚠'} billable)")

        candidates.append({
            "original_term":       term,
            "canonical_term":      canonical,       # UMLS normalized
            "cui":                 cui,             # UMLS concept ID
            "certainty":           certainty,
            "primary_code":        selected,
            "primary_description": verified_description,
            "is_billable":         is_billable,
            "confidence":          confidence,      # Mohammed's request
            "code_source":         result.get("source", "mcp"),
            "alternative_code":    result.get("alternative_code"),
            "coding_notes":        result.get("coding_notes", ""),
            "reasoning":           result.get("reasoning", ""),
            "snomed_codes":        snomed_codes,    # for Phase 2 KG
            "umls_icd10_hints":    icd10_hints,     # direct UMLS mappings
            "mcp_verified":        True
        })

    return {"candidates": candidates}


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 3: Validator (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def agent_validator(clinical_note: str, candidates: list) -> dict:
    candidates_text = json.dumps(candidates, indent=2)
    prompt = f"""You are a senior medical coding auditor.

Validate the proposed ICD-10-CM codes. These codes were grounded using:
- UMLS normalization (canonical terms + CUI crosswalk)
- MCP real-time ICD-10-CM database lookup

Focus your validation on:
1. SEQUENCING — principal diagnosis listed first?
2. COMBINATION CODES — should any codes merge into one?
3. EXCLUSIONS — any mutually exclusive code pairs?
4. MISSING CODES — any conditions in the note not captured?
5. PHYSICIAN QUERIES — any ambiguities needing clarification?
6. CONFIDENCE FLAGS — review any low-confidence codes carefully

Respond ONLY in this JSON format:
{{
  "validated_codes": [
    {{
      "original_term": "term",
      "code": "ICD-10 code",
      "description": "description",
      "status": "approved | revised | flagged",
      "issue": "describe problem or null",
      "correction": "corrected code if revised or null"
    }}
  ],
  "missing_codes": [
    {{"condition": "missed condition", "suggested_code": "code", "reason": "why"}}
  ],
  "physician_queries": ["query text"]
}}

Original Note:
{clinical_note}

UMLS + MCP Verified Codes:
{candidates_text}
"""
    raw = call_ollama(prompt, "Validator")
    return parse_json(raw)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 4: Reconciler (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def agent_reconciler(clinical_note: str, validated: dict) -> dict:
    validated_text = json.dumps(validated, indent=2)
    prompt = f"""You are a medical coding supervisor producing the final claim-ready list.

1. Sequence codes: principal/primary diagnosis FIRST
2. Add missing codes from validator
3. Exclude invalid or ruled-out codes
4. Assign role: PRIMARY, SECONDARY, or COMPLICATION
5. Write a brief summary for the billing team
6. Flag if physician query needed before submission

Respond ONLY in this JSON format:
{{
  "final_codes": [
    {{
      "sequence": 1,
      "code": "ICD-10 code",
      "description": "description",
      "role": "PRIMARY | SECONDARY | COMPLICATION",
      "justification": "why this code and sequence"
    }}
  ],
  "claim_ready": true,
  "requires_physician_query": false,
  "queries": [],
  "coding_summary": "2-3 sentence summary for billing team"
}}

Original Note:
{clinical_note}

Validated Codes:
{validated_text}
"""
    raw = call_ollama(prompt, "Reconciler")
    return parse_json(raw)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(clinical_note: str, umls_api_key: str = UMLS_API_KEY):

    print("\n" + "═" * 65)
    print("  ICD-10 MULTI-AGENT PIPELINE  (UMLS + MCP Grounded)")
    print("═" * 65)
    print("\nCLINICAL NOTE:")
    print("-" * 65)
    print(clinical_note.strip())
    print("-" * 65)

    # Initialize clients
    umls = UMLSNormalizer(umls_api_key)
    mcp  = ICD10MCPClient()

    try:
        # ── Stage 1: Evidence extraction ──────────────────────────────────
        print("\n📋 STAGE 1: Extracting clinical evidence...")
        evidence = agent_evidence_extractor(clinical_note)
        findings = evidence.get("findings", [])
        print(f"  → Found {len(findings)} finding(s)")

        # ── Stage 1.5: UMLS Normalization ──────────────────────────────────
        # New step between Agent 1 and Agent 2
        print("\n🧬 STAGE 1.5: Normalizing terms via UMLS...")
        if umls_api_key == "YOUR_UMLS_API_KEY_HERE":
            print("  ⚠️  No UMLS API key set — skipping normalization, using raw terms")
            enriched_findings = findings
        else:
            enriched_findings = umls.normalize_all_findings(findings)
            normalized_count  = sum(1 for f in enriched_findings if f.get("cui"))
            hints_count       = sum(len(f.get("icd10_hints", [])) for f in enriched_findings)
            print(f"  → {normalized_count}/{len(findings)} terms normalized")
            print(f"  → {hints_count} direct ICD-10 hints from UMLS crosswalk")

        # ── Stage 2: Code candidate generation ────────────────────────────
        print("\n🔎 STAGE 2: Looking up codes (UMLS canonical + MCP database)...")
        candidates_result = agent_code_candidate_generator(enriched_findings, mcp)
        candidates        = candidates_result.get("candidates", [])
        high_conf  = sum(1 for c in candidates if c.get("confidence") == "high")
        mid_conf   = sum(1 for c in candidates if c.get("confidence") == "medium")
        low_conf   = sum(1 for c in candidates if c.get("confidence") == "low")
        print(f"  → Confidence: 🟢 {high_conf} high | 🟡 {mid_conf} medium | 🔴 {low_conf} low")

        # ── Stage 3: Validation ───────────────────────────────────────────
        print("\n✅ STAGE 3: Validating codes against guidelines...")
        validation     = agent_validator(clinical_note, candidates)
        validated      = validation.get("validated_codes", [])
        approved_count = sum(1 for c in validated if c.get("status") == "approved")
        revised_count  = sum(1 for c in validated if c.get("status") == "revised")
        flagged_count  = sum(1 for c in validated if c.get("status") == "flagged")
        missing        = validation.get("missing_codes", [])
        queries        = validation.get("physician_queries", [])
        print(f"  → Approved: {approved_count} | Revised: {revised_count} | Flagged: {flagged_count}")
        if missing:
            print(f"  → Found {len(missing)} missing code(s)")
        if queries:
            print(f"  → {len(queries)} physician query/queries needed")

        # ── Stage 4: Reconciliation ───────────────────────────────────────
        print("\n📊 STAGE 4: Reconciling final claim-ready code list...")
        final_result = agent_reconciler(clinical_note, validation)
        final_codes  = final_result.get("final_codes", [])
        print(f"  → Final list: {len(final_codes)} code(s)")

        # ── Print final results ───────────────────────────────────────────
        print("\n" + "═" * 65)
        print("  FINAL ICD-10 CODE LIST  (UMLS + MCP Verified)")
        print("═" * 65)

        for entry in final_codes:
            icon = "🔴" if entry.get("role") == "PRIMARY" else "🔵"
            print(f"\n  {icon} [{entry.get('sequence')}] {entry.get('code')}  —  {entry.get('description')}")
            print(f"      Role          : {entry.get('role')}")
            print(f"      Justification : {entry.get('justification')}")

        if final_result.get("requires_physician_query"):
            print("\n  ⚠️  PHYSICIAN QUERIES:")
            for q in final_result.get("queries", []):
                print(f"     • {q}")

        print(f"\n  📝 BILLING SUMMARY:")
        print(f"     {final_result.get('coding_summary', 'N/A')}")

        status = "✅ READY" if final_result.get("claim_ready") else "⛔ HOLD"
        print(f"\n  CLAIM STATUS: {status}")
        print("═" * 65)

        # Save full results including UMLS enrichment data
        output = {
            "evidence":          evidence,
            "umls_enriched":     enriched_findings,  # includes CUIs + SNOMED for KG
            "candidates":        candidates_result,
            "validation":        validation,
            "final":             final_result
        }
        with open("icd10_umls_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\n  Full results saved to icd10_umls_results.json")
        print("  (SNOMED codes saved for Phase 2 knowledge graph integration)")

        return output

    finally:
        mcp.close()
        print("\n  [MCP] Server stopped.")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_note = """
    Patient is a 67-year-old male admitted for chest pain and shortness of breath.
    History of T2DM with DKD stage 3 and HTN. EKG shows AFib with RVR.
    Troponin mildly elevated but repeat troponin negative — acute MI ruled out.
    CXR shows mild pulmonary edema. BMI documented at 34.2.
    Patient was rate-controlled and discharged stable with cardiology follow-up.
    """
    # Note: this note uses clinical shorthand (T2DM, DKD, HTN, AFib, RVR, CXR)
    # UMLS normalization will expand these to canonical terms before MCP search

    run_pipeline(sample_note)
