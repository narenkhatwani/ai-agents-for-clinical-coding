"""
Simple Streamlit app: add SNOMED CT entities (looked up live via UMLS)
and inspect the cosine similarity between their embeddings.

Run:
    streamlit run snomed_similarity_app.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
ENV_FILE = PROJECT_ROOT / ".env"
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

from snomed_ontology import (  # noqa: E402 — needs sys.path set up first
    EMBED_MODEL,
    GAP_SEARCH_WINDOW,
    MAX_NEIGHBORHOOD_CANDIDATES,
    MAX_WORKERS,
    OLLAMA_BASE,
    biggest_gap_cutoff,
    configure,
    cosine_sim,
    embed,
    explore_neighborhood,
    get_concept_name,
    get_cui_for_sctid,
    get_sctid,
    load_umls_api_key,
    search_snomed,
    wu_palmer,
)

UMLS_API_KEY = load_umls_api_key(PROJECT_ROOT)
configure(UMLS_API_KEY)


# ── App ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SNOMED CT Similarity", layout="wide")
st.title("SNOMED CT Entity Similarity")
st.caption(
    f"Entities are resolved against live SNOMED CT (via UMLS). Two similarity scores are shown: "
    f"cosine similarity between `{EMBED_MODEL}` text embeddings (semantic/lexical), and Wu-Palmer "
    f"similarity over the SNOMED is-a hierarchy (purely graph-based, ignores wording entirely)."
)

if not UMLS_API_KEY:
    st.error(f"No UMLS_API_KEY found in {ENV_FILE}. Add it and restart the app.")
    st.stop()

if "entities" not in st.session_state:
    st.session_state.entities = []  # list of {cui, sctid, name, query}
if "candidates" not in st.session_state:
    st.session_state.candidates = []

# ── Add entity ────────────────────────────────────────────────────────────────
st.subheader("Add an entity")
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Search SNOMED CT", placeholder="e.g. atrial fibrillation", label_visibility="collapsed")
with col2:
    search_clicked = st.button("Search", use_container_width=True)

if search_clicked and query.strip():
    try:
        with st.spinner("Searching SNOMED CT via UMLS..."):
            st.session_state.candidates = search_snomed(query.strip())
        if not st.session_state.candidates:
            st.warning("No SNOMED CT concepts found for that term.")
    except requests.RequestException as e:
        st.error(f"UMLS request failed: {e}")

if st.session_state.candidates:
    labels = [f'{c["name"]}  (CUI {c["cui"]})' for c in st.session_state.candidates]
    choice = st.radio("Matching concepts — pick the one to add:", labels, index=0)
    if st.button("Add selected concept"):
        picked = st.session_state.candidates[labels.index(choice)]
        if any(e["cui"] == picked["cui"] for e in st.session_state.entities):
            st.info("That concept is already in your list.")
        else:
            with st.spinner("Fetching SCTID and computing embedding..."):
                sctid = get_sctid(picked["cui"])
                try:
                    embed(picked["name"])  # warm the cache
                except requests.RequestException as e:
                    st.error(
                        f"Could not reach Ollama at {OLLAMA_BASE} to embed this term: {e}\n"
                        "Make sure `ollama serve` is running and `nomic-embed-text` is pulled."
                    )
                    st.stop()
            st.session_state.entities.append(
                {"cui": picked["cui"], "sctid": sctid, "name": picked["name"], "query": query.strip()}
            )
            st.session_state.candidates = []
            st.rerun()

st.divider()

# ── Entity list ───────────────────────────────────────────────────────────────
st.subheader(f"Entities ({len(st.session_state.entities)})")

if not st.session_state.entities:
    st.info("No entities yet — search and add at least two to compare.")
else:
    table = pd.DataFrame(
        [
            {
                "Preferred term": e["name"],
                "CUI": e["cui"],
                "SCTID": e["sctid"] or "—",
                "Searched as": e["query"],
            }
            for e in st.session_state.entities
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    remove_labels = [f'{e["name"]} ({e["cui"]})' for e in st.session_state.entities]
    to_remove = st.multiselect("Remove entities", remove_labels)
    col_a, col_b = st.columns(2)
    with col_a:
        if to_remove and st.button("Remove selected"):
            st.session_state.entities = [
                e for e, lbl in zip(st.session_state.entities, remove_labels) if lbl not in to_remove
            ]
            st.rerun()
    with col_b:
        if st.button("Clear all"):
            st.session_state.entities = []
            st.rerun()

st.divider()

# ── Similarity ────────────────────────────────────────────────────────────────
n = len(st.session_state.entities)
if n < 2:
    st.subheader("Similarity")
    st.info("Add at least two entities to see similarity scores.")
else:
    names = [e["name"] for e in st.session_state.entities]
    sctids = [e["sctid"] for e in st.session_state.entities]

    try:
        vectors = [embed(e["name"]) for e in st.session_state.entities]
    except requests.RequestException as e:
        st.error(f"Could not reach Ollama at {OLLAMA_BASE}: {e}")
        st.stop()

    cos_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_sim[i, j] = cosine_sim(vectors[i], vectors[j])

    st.subheader("Cosine similarity (text embedding)")
    cos_df = pd.DataFrame(cos_sim, index=names, columns=names)
    st.dataframe(
        cos_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.3f}"),
        use_container_width=True,
    )

    missing_sctid = [names[i] for i in range(n) if not sctids[i]]
    if missing_sctid:
        st.warning(f"No SCTID resolved for: {', '.join(missing_sctid)} — Wu-Palmer can't be computed for these.")

    st.subheader("Wu-Palmer similarity (is-a hierarchy)")
    with st.spinner("Walking the SNOMED is-a hierarchy (first run per concept is slower, then cached)..."):
        wp_sim = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                score = wu_palmer(sctids[i], sctids[j])
                if score is not None:
                    wp_sim[i, j] = score

    wp_df = pd.DataFrame(wp_sim, index=names, columns=names)
    st.dataframe(
        wp_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format("{:.3f}", na_rep="—"),
        use_container_width=True,
    )

    if n == 2:
        col1, col2 = st.columns(2)
        col1.metric(f'Cosine: "{names[0]}" vs "{names[1]}"', f"{cos_sim[0, 1]:.3f}")
        wp_val = wp_sim[0, 1]
        col2.metric(f'Wu-Palmer: "{names[0]}" vs "{names[1]}"', f"{wp_val:.3f}" if not np.isnan(wp_val) else "—")

    st.caption("Pairwise table")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            wp_val = wp_sim[i, j]
            pairs.append(
                {
                    "Entity A": names[i],
                    "Entity B": names[j],
                    "Cosine similarity": round(float(cos_sim[i, j]), 4),
                    "Wu-Palmer similarity": round(float(wp_val), 4) if not np.isnan(wp_val) else None,
                }
            )
    pairs_df = pd.DataFrame(pairs).sort_values("Cosine similarity", ascending=False)
    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

st.divider()

# ── Neighborhood explorer (big-gap heuristic) ────────────────────────────────
st.subheader("Neighborhood explorer (big-gap heuristic)")
st.caption(
    f"Pick a seed entity — its SNOMED is-a neighborhood (parents, children, and — via shared "
    f"parents — siblings) is walked automatically, out to N hops, capped at {MAX_NEIGHBORHOOD_CANDIDATES} "
    f"concepts (some branches fan out into hundreds within 2-3 hops). Every discovered concept is "
    f"scored against the seed with both Wu-Palmer and cosine similarity, and **each metric gets its "
    f"own independent big-gap cutoff** — Wu-Palmer's neighborhood and cosine's neighborhood are ranked "
    f"and cut separately, since the two metrics can disagree about which concepts are 'close'."
)

seed_options = [e for e in st.session_state.entities if e["sctid"]]
if not seed_options:
    st.info("Add at least one entity with a resolved SCTID to explore its neighborhood.")
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        seed_labels = [e["name"] for e in seed_options]
        seed_choice = st.selectbox("Seed entity", seed_labels)
    with col2:
        max_hops = st.number_input("Max hops", min_value=1, max_value=5, value=3)

    seed = seed_options[seed_labels.index(seed_choice)]
    cache_key = (seed["sctid"], max_hops)

    if st.session_state.get("neighborhood_key") != cache_key:
        with st.spinner(f"Walking the is-a graph up to {max_hops} hops from \"{seed['name']}\"..."):
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                found = explore_neighborhood(seed["sctid"], max_hops, executor)
                candidate_sctids = list(found)

                if not candidate_sctids:
                    rows = []
                else:
                    wp_scores = list(executor.map(lambda s: wu_palmer(seed["sctid"], s) or 0.0, candidate_sctids))
                    names = list(executor.map(get_concept_name, candidate_sctids))
                    try:
                        seed_vec = embed(seed["name"])
                        candidate_vecs = list(executor.map(embed, names))
                        cos_scores = [round(cosine_sim(seed_vec, v), 4) for v in candidate_vecs]
                    except requests.RequestException as e:
                        st.warning(f"Could not reach Ollama at {OLLAMA_BASE} for cosine similarity: {e}")
                        cos_scores = [None] * len(candidate_sctids)

                    rows = list(zip(candidate_sctids, names, wp_scores, cos_scores))
                    rows.sort(key=lambda r: r[2], reverse=True)  # sort by Wu-Palmer, biggest_gap_cutoff needs it

        st.session_state.neighborhood = {"seed": seed, "found": found, "rows": rows}
        st.session_state.neighborhood_key = cache_key

    nb = st.session_state.neighborhood
    rows = nb["rows"]
    found = nb["found"]

    if not rows:
        st.warning("No neighboring concepts found — try increasing max hops.")
    else:

        def render_metric_neighborhood(metric_label: str, score_index: int):
            """Sort `rows` by one metric (Wu-Palmer=2, Cosine=3), apply that
            metric's own MAD-based big-gap cutoff, and render its table. Each
            metric is ranked and cut independently — Wu-Palmer and cosine can
            (and do) disagree about which concepts are "close", so sharing a
            single ranking would silently favor whichever metric happened to
            be used for sorting. Returns the set of SCTIDs kept for this metric."""
            sortable = [r for r in rows if r[score_index] is not None]
            sorted_rows = sorted(sortable, key=lambda r: r[score_index], reverse=True)
            scored = [(r[0], r[score_index]) for r in sorted_rows]
            cutoff_idx, gap, is_significant = biggest_gap_cutoff(scored)
            n_kept = (cutoff_idx + 1) if cutoff_idx is not None else len(sorted_rows)

            def _build_table(subset_rows, start_rank):
                table_rows = []
                for offset, (sctid, name, wp, cos) in enumerate(subset_rows):
                    i = start_rank + offset
                    this_score = sorted_rows[i][score_index]
                    next_score = sorted_rows[i + 1][score_index] if i + 1 < len(sorted_rows) else None
                    next_gap = round(this_score - next_score, 4) if next_score is not None else None
                    table_rows.append(
                        {
                            "Concept": name,
                            "SCTID": sctid,
                            "Hops": found[sctid],
                            "Cosine similarity": cos,
                            "Wu-Palmer": wp,
                            f"Gap to next ({metric_label})": next_gap,
                        }
                    )
                return pd.DataFrame(table_rows)

            st.markdown(f"**{metric_label} neighborhood**")
            if cutoff_idx is None:
                st.info("Only one neighbor found — no gap to detect.")
            elif is_significant:
                st.success(
                    f"Significant gap ({gap:.4f}, outlier vs. this neighborhood's typical gap size) falls between "
                    f"rank {n_kept} and {n_kept + 1} of {len(sorted_rows)} — showing the top {n_kept} below; "
                    f"the rest are collapsed."
                )
            else:
                st.warning(
                    f"No statistically significant {metric_label} gap found — scores decay smoothly with no "
                    f"clear cluster boundary. Falling back to the largest available gap ({gap:.4f}) between "
                    f"rank {n_kept} and {n_kept + 1} of {len(sorted_rows)}; treat this cutoff as a weak, "
                    f"best-effort guess rather than a confident one."
                )

            kept_df = _build_table(sorted_rows[:n_kept], 0)
            st.dataframe(
                kept_df.style.format({"Cosine similarity": "{:.4f}", "Wu-Palmer": "{:.4f}"}, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )

            remaining = sorted_rows[n_kept:]
            if remaining:
                with st.expander(f"Show {len(remaining)} more concepts below the {metric_label} gap"):
                    rest_df = _build_table(remaining, n_kept)
                    st.dataframe(
                        rest_df.style.format({"Cosine similarity": "{:.4f}", "Wu-Palmer": "{:.4f}"}, na_rep="—"),
                        use_container_width=True,
                        hide_index=True,
                    )

            return {sctid for sctid, *_ in sorted_rows[:n_kept]}

        wp_kept_sctids = render_metric_neighborhood("Wu-Palmer", score_index=2)
        st.divider()
        cos_kept_sctids = render_metric_neighborhood("Cosine similarity", score_index=3)

        st.divider()
        with st.expander("Optional: add discovered concepts to your entity list"):
            names_map = {sctid: name for sctid, name, _, _ in rows}
            suggested_sctids = wp_kept_sctids | cos_kept_sctids
            suggested = [names_map[sctid] for sctid in suggested_sctids]
            to_add = st.multiselect(
                "Defaults to concepts kept by either metric's cutoff — these will appear in the main "
                "Entities table/similarity matrices above",
                [names_map[sctid] for sctid, _, _, _ in rows],
                default=suggested,
            )
            if to_add and st.button("Add selected to entities"):
                label_to_sctid = {names_map[sctid]: sctid for sctid, _, _, _ in rows}
                added, skipped = 0, []
                with st.spinner("Resolving CUIs..."):
                    for label in to_add:
                        sctid = label_to_sctid[label]
                        if any(e["sctid"] == sctid for e in st.session_state.entities):
                            continue
                        cui = get_cui_for_sctid(sctid)
                        if not cui:
                            skipped.append(label)
                            continue
                        st.session_state.entities.append(
                            {"cui": cui, "sctid": sctid, "name": label, "query": f"(neighborhood of {nb['seed']['name']})"}
                        )
                        added += 1
                if added:
                    st.session_state.pop("neighborhood", None)
                    st.session_state.pop("neighborhood_key", None)
                    st.rerun()
                if skipped:
                    st.warning(f"Could not resolve: {', '.join(skipped)}")
