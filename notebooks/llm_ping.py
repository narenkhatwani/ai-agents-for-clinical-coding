"""Tiny OpenRouter diagnostic. Sends ~30-token requests in several payload variants (Naren's exact payload first)
and prints the HTTP status and error body of each, so a 400/404 can be traced to one setting.
Cost: a fraction of a cent. Uses Naren's config; the key comes from $env:OPENROUTER_API_KEY and is never printed.

    python llm_ping.py
"""
import sys
from pathlib import Path

import requests

import guideline_check as gc
import pipeline as p


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gc.ensure_ca_bundle(Path("../../icd10cm_sources/ca_bundle.pem"))
    cfg = p.get_llm_config()
    key = cfg.resolved_api_key()
    if not key:
        sys.exit(f"No key: set $env:{cfg.api_key_env} first")
    print(f"model={cfg.model} | zdr={cfg.openrouter_zdr} | temperature={cfg.temperature}")
    print(f"key: length {len(key)} | starts with sk-or: {key.startswith('sk-or')} | contains whitespace: {any(c.isspace() for c in key)}")
    url = cfg.api_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **cfg.api_extra_headers}
    base = {"model": cfg.model, "messages": [{"role": "user", "content": 'Reply with the JSON {"ok": true}'}], "max_tokens": 30}
    temp = {"temperature": cfg.temperature}
    jsn = {"response_format": {"type": "json_object"}}
    naren_provider = {"provider": {"zdr": cfg.openrouter_zdr, "order": ["Phala", "Together"], "allow_fallbacks": True}}
    variants = {
        "A  Naren's exact payload": {**base, **temp, **jsn, **naren_provider},
        "B  without provider block": {**base, **temp, **jsn},
        "C  without response_format": {**base, **temp, **naren_provider},
        "D  without temperature": {**base, **jsn, **naren_provider},
        "E  zero-retention only (no provider order)": {**base, **temp, **jsn, "provider": {"zdr": True}},
        "F  minimal request": base,
    }
    for name, payload in variants.items():
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            print(f"\n[{name}] HTTP {r.status_code} | content-type: {r.headers.get('content-type')}")
            print(f"    body: {r.text[:500]!r}")
        except Exception as exc:
            print(f"\n[{name}] request failed: {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
