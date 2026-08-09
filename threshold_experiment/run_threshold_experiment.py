#!/usr/bin/env python3
"""
Threshold calibration for Stage 6 MiniLM cosine similarity.

Samples 20 random SNOMED CT clinical concepts from the local RF2 package
(same content as the SNOMED terminology API / browser), collects ontology
neighbors (2-level is-a ancestors + attribute targets used in Stage 6),
embeds labels with all-MiniLM-L6-v2, and reports similarity statistics to
justify a retain threshold.

Usage (from repo root):
  python threshold_experiment/run_threshold_experiment.py
  python threshold_experiment/run_threshold_experiment.py --n 20 --seed 42

Outputs:
  threshold_experiment/results/threshold_experiment_summary.json
  threshold_experiment/results/threshold_experiment_per_term.json
  threshold_experiment/results/threshold_experiment_report.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO_ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from snomed_ct import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HIGH_CONF_COSINE_SIM,
    DEFAULT_MIN_COSINE_SIM,
    TextEmbedder,
    attribute_related_concepts,
    build_snomed_index,
    find_snomed_root,
    is_a_ancestors_depth2,
)

OUT_DIR = Path(__file__).resolve().parent / "results"

# Prefer clinically meaningful semantic tags when sampling random terms
PREFERRED_TAGS = (
    "finding",
    "disorder",
    "procedure",
    "observable entity",
    "situation",
)


def _semantic_tag(fsn: Optional[str]) -> str:
    if not fsn:
        return ""
    if "(" in fsn and fsn.endswith(")"):
        return fsn[fsn.rfind("(") + 1 : -1].lower()
    return ""


def sample_concepts(
    index,
    n: int = 20,
    seed: int = 42,
    min_token_len: int = 4,
) -> List[Dict[str, str]]:
    """
    Random sample of active concepts with a preferred term + FSN, biased toward
    findings/disorders so Stage-6 neighbor walks are non-empty.
    """
    rng = random.Random(seed)
    pool: List[Dict[str, str]] = []
    for cid in index.active_concepts:
        pt_raw = index.concept_pt.get(cid)
        fsn_raw = index.concept_fsn.get(cid)
        if pt_raw is None or (isinstance(pt_raw, float) and pt_raw != pt_raw):
            continue
        pt = str(pt_raw).strip()
        fsn = "" if fsn_raw is None else str(fsn_raw).strip()
        if fsn.lower() in ("nan", "none"):
            fsn = ""
        if not pt or pt.lower() in ("nan", "none") or len(pt) < min_token_len:
            continue
        tag = _semantic_tag(fsn)
        if tag and tag not in PREFERRED_TAGS:
            continue
        pool.append(
            {
                "concept_id": cid,
                "preferred_term": pt,
                "fsn": fsn,
                "semantic_tag": tag or "unknown",
            }
        )

    if len(pool) < n:
        raise RuntimeError(
            f"Only {len(pool)} candidate concepts for sampling (need {n}). "
            "Check SNOMED index."
        )

    rng.shuffle(pool)
    selected: List[Dict[str, str]] = []
    for concept in pool:
        if is_a_ancestors_depth2(index, concept["concept_id"]):
            selected.append(concept)
        if len(selected) >= n:
            break
    if len(selected) < n:
        raise RuntimeError(
            f"Only found {len(selected)} concepts with is-a parents (need {n})."
        )
    return selected


def collect_neighbors(index, concept_id: str) -> List[Dict[str, Any]]:
    """Stage-6-style neighbors: 2-level is_a + attribute targets (/ancestors)."""
    neighbors: List[Dict[str, Any]] = []
    for row in is_a_ancestors_depth2(index, concept_id):
        neighbors.append({**row, "bucket": "is_a_ancestor"})
    for row in attribute_related_concepts(index, concept_id, include_inverse=False):
        neighbors.append({**row, "bucket": "attribute"})
    # de-dupe by concept_id keeping first (is_a preferred)
    seen = set()
    uniq = []
    for row in neighbors:
        cid = row.get("concept_id")
        if cid in seen:
            continue
        seen.add(cid)
        uniq.append(row)
    return uniq


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def summarize_sims(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "stdev": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    s = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "stdev": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "p10": round(percentile(s, 10), 4),
        "p25": round(percentile(s, 25), 4),
        "p75": round(percentile(s, 75), 4),
        "p90": round(percentile(s, 90), 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
    }


def recommend_threshold(
    top1_sims: List[float],
    top2_sims: List[float],
    parent_sims: List[float],
) -> Dict[str, Any]:
    """
    Propose retain thresholds from empirical distributions.

    Rationale used in the report:
    - top-1 / top-2 average ≈ "most relevant neighbor" band
    - mean of direct is_a parents ≈ typical true ancestor score
    - suggested floor = min(0.70, mean_top1 - 0.05) then snapped to {0.70, 0.75, 0.80}
      but never below 0.70 (project policy)
    """
    mean_top1 = statistics.mean(top1_sims) if top1_sims else float("nan")
    mean_top2 = statistics.mean(top2_sims) if top2_sims else float("nan")
    mean_parent = statistics.mean(parent_sims) if parent_sims else float("nan")
    med_top1 = statistics.median(top1_sims) if top1_sims else float("nan")

    # Empirical "most relevant" band midpoint
    band_center = statistics.mean(
        [x for x in (mean_top1, mean_top2, mean_parent) if x == x]  # not nan
    )

    candidates = [0.70, 0.75, 0.80]
    # Pick highest candidate ≤ max(band_center - 0.02, 0.70) so we stay below the
    # mean most-relevant sim but never under 0.70
    target = max(0.70, min(band_center - 0.02, 0.80))
    recommended = max(c for c in candidates if c <= target + 1e-9)
    # if band is very high, prefer 0.75 or 0.80
    if band_center >= 0.82:
        recommended = 0.80
    elif band_center >= 0.76:
        recommended = 0.75
    else:
        recommended = 0.70

    high_conf = 0.80 if recommended <= 0.75 else 0.85

    return {
        "mean_most_relevant_top1": round(mean_top1, 4),
        "mean_most_relevant_top2": round(mean_top2, 4),
        "mean_is_a_parent": round(mean_parent, 4),
        "median_most_relevant_top1": round(med_top1, 4),
        "empirical_band_center": round(band_center, 4),
        "recommended_min_similarity": recommended,
        "recommended_high_confidence": high_conf,
        "policy_floor": 0.70,
        "current_pipeline_min": DEFAULT_MIN_COSINE_SIM,
        "current_pipeline_high_conf": DEFAULT_HIGH_CONF_COSINE_SIM,
        "justification": (
            f"Across n={len(top1_sims)} sampled SNOMED terms, the average MiniLM "
            f"cosine of the single most-relevant ontology neighbor (top-1) is "
            f"{mean_top1:.3f}; mean over top-2 is {mean_top2:.3f}; mean direct "
            f"is-a parent is {mean_parent:.3f}. Band center ≈ {band_center:.3f}. "
            f"Recommended retain threshold is {recommended:.2f} "
            f"(≥ 0.70 policy floor); high-confidence tier ≥ {high_conf:.2f}."
        ),
    }


def run_experiment(
    n: int = 20,
    seed: int = 42,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    snomed_root = find_snomed_root(REPO_ROOT / "data")
    index = build_snomed_index(
        snomed_root=snomed_root,
        cache_path=REPO_ROOT / "data" / "snomed_index" / "snomed_index.pkl",
    )
    sample = sample_concepts(index, n=n, seed=seed)

    # Collect all labels for one encode pass
    texts: List[str] = []
    prepared: List[Tuple[Dict[str, str], List[Dict[str, Any]]]] = []
    for concept in sample:
        neighbors = collect_neighbors(index, concept["concept_id"])
        prepared.append((concept, neighbors))
        texts.append(concept["preferred_term"])
        for nb in neighbors:
            if nb.get("term"):
                texts.append(nb["term"])

    embedder = TextEmbedder(model_name=model_name, prefer_embeddings=True)
    embedder.encode_many(texts, show_progress=True)
    if embedder.method != "minilm":
        raise RuntimeError(
            "MiniLM required for this experiment. "
            "Install: pip install sentence-transformers"
        )

    per_term: List[Dict[str, Any]] = []
    all_neighbor_sims: List[float] = []
    all_is_a_sims: List[float] = []
    all_parent_sims: List[float] = []  # depth-1 is_a only
    top1_sims: List[float] = []
    top2_means: List[float] = []

    for concept, neighbors in prepared:
        anchor = concept["preferred_term"]
        scored = []
        for nb in neighbors:
            term = nb.get("term") or ""
            sim = float(embedder.similarity(anchor, term))
            sim = max(-1.0, min(1.0, sim))
            row = {
                **nb,
                "cosine_similarity": round(sim, 4),
                "cosine_distance": round(1.0 - sim, 4),
            }
            scored.append(row)
            all_neighbor_sims.append(sim)
            if nb.get("bucket") == "is_a_ancestor":
                all_is_a_sims.append(sim)
                if nb.get("depth") == 1:
                    all_parent_sims.append(sim)

        scored_sorted = sorted(
            scored, key=lambda r: r["cosine_similarity"], reverse=True
        )
        most_relevant = scored_sorted[:5]
        top1 = most_relevant[0]["cosine_similarity"] if most_relevant else float("nan")
        top2 = [r["cosine_similarity"] for r in most_relevant[:2]]
        if most_relevant:
            top1_sims.append(top1)
        if top2:
            top2_means.append(statistics.mean(top2))

        depth1 = [r for r in scored if r.get("bucket") == "is_a_ancestor" and r.get("depth") == 1]
        per_term.append(
            {
                "concept_id": concept["concept_id"],
                "preferred_term": concept["preferred_term"],
                "fsn": concept["fsn"],
                "semantic_tag": concept["semantic_tag"],
                "n_neighbors": len(scored),
                "n_is_a": sum(1 for r in scored if r.get("bucket") == "is_a_ancestor"),
                "n_attributes": sum(1 for r in scored if r.get("bucket") == "attribute"),
                "most_relevant": most_relevant,
                "top1_similarity": top1 if most_relevant else None,
                "top2_mean_similarity": round(statistics.mean(top2), 4) if top2 else None,
                "mean_is_a_parent_similarity": (
                    round(statistics.mean([r["cosine_similarity"] for r in depth1]), 4)
                    if depth1
                    else None
                ),
                "all_neighbors_scored": scored_sorted,
            }
        )

    rec = recommend_threshold(top1_sims, top2_means, all_parent_sims)

    summary = {
        "experiment": "snomed_minilm_threshold_calibration",
        "generated_at": datetime.now().isoformat(),
        "source": {
            "type": "local_SNOMED_CT_RF2",
            "note": (
                "Random sample from local US SNOMED CT Snapshot (RF2). "
                "Same concept/description model as SNOMED Terminology API."
            ),
            "snomed_root": str(snomed_root),
            "n_sampled": n,
            "seed": seed,
            "semantic_tag_filter": list(PREFERRED_TAGS),
        },
        "embedding": {
            "model": model_name,
            "method": embedder.method,
            "anchor": "snomed_preferred_term",
            "neighbors": "is_a depth≤2 + Stage-6 attribute targets (no inverse)",
        },
        "similarity_distributions": {
            "all_neighbors": summarize_sims(all_neighbor_sims),
            "is_a_ancestors_only": summarize_sims(all_is_a_sims),
            "is_a_parents_depth1": summarize_sims(all_parent_sims),
            "most_relevant_top1_per_term": summarize_sims(top1_sims),
            "most_relevant_top2_mean_per_term": summarize_sims(top2_means),
        },
        "threshold_recommendation": rec,
        "pass_rates_at_thresholds": {
            thr: {
                "fraction_top1_pass": round(
                    sum(1 for s in top1_sims if s >= thr) / max(len(top1_sims), 1), 4
                ),
                "fraction_parent_pass": round(
                    sum(1 for s in all_parent_sims if s >= thr)
                    / max(len(all_parent_sims), 1),
                    4,
                ),
                "fraction_all_neighbor_pass": round(
                    sum(1 for s in all_neighbor_sims if s >= thr)
                    / max(len(all_neighbor_sims), 1),
                    4,
                ),
            }
            for thr in (0.70, 0.75, 0.80)
        },
        "sample_concepts": [
            {
                "concept_id": t["concept_id"],
                "preferred_term": t["preferred_term"],
                "top1_similarity": t["top1_similarity"],
                "top1_neighbor": (t["most_relevant"][0]["term"] if t["most_relevant"] else None),
            }
            for t in per_term
        ],
    }
    return {"summary": summary, "per_term": per_term}


def write_markdown_report(summary: Dict[str, Any], path: Path) -> None:
    rec = summary["threshold_recommendation"]
    dist = summary["similarity_distributions"]
    lines = [
        "# SNOMED MiniLM threshold experiment",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Setup",
        "",
        f"- **Source:** {summary['source']['note']}",
        f"- **n terms:** {summary['source']['n_sampled']} (seed={summary['source']['seed']})",
        f"- **Model:** `{summary['embedding']['model']}`",
        f"- **Neighbors:** {summary['embedding']['neighbors']}",
        "",
        "## Average similarities (across sampled terms)",
        "",
        "| Quantity | mean | median | p25 | p75 | n |",
        "|----------|-----:|-------:|----:|----:|--:|",
    ]
    for key, label in (
        ("most_relevant_top1_per_term", "Most relevant neighbor (top-1)"),
        ("most_relevant_top2_mean_per_term", "Top-2 mean"),
        ("is_a_parents_depth1", "Is-a parent (depth 1)"),
        ("is_a_ancestors_only", "All is-a (depth ≤2)"),
        ("all_neighbors", "All neighbors (is-a + attributes)"),
    ):
        d = dist[key]
        lines.append(
            f"| {label} | {d['mean']} | {d['median']} | {d['p25']} | {d['p75']} | {d['n']} |"
        )

    lines += [
        "",
        "## Pass rates",
        "",
        "| Threshold | top-1 passes | is-a parent passes | any-neighbor passes |",
        "|----------:|-------------:|-------------------:|--------------------:|",
    ]
    for thr, row in summary["pass_rates_at_thresholds"].items():
        lines.append(
            f"| {thr} | {row['fraction_top1_pass']} | "
            f"{row['fraction_parent_pass']} | {row['fraction_all_neighbor_pass']} |"
        )

    lines += [
        "",
        "## Recommendation",
        "",
        f"- **Recommended min similarity:** **{rec['recommended_min_similarity']}**",
        f"- **Recommended high-confidence:** **{rec['recommended_high_confidence']}**",
        f"- Policy floor (never below): `{rec['policy_floor']}`",
        f"- Current pipeline: min=`{rec['current_pipeline_min']}`, "
        f"high-conf=`{rec['current_pipeline_high_conf']}`",
        "",
        rec["justification"],
        "",
        "## Sampled concepts (top-1 neighbor)",
        "",
    ]
    for row in summary["sample_concepts"]:
        lines.append(
            f"- `{row['concept_id']}` **{row['preferred_term']}** → "
            f"*{row['top1_neighbor']}* (sim={row['top1_similarity']})"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Number of random SNOMED terms")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformers model name",
    )
    args = parser.parse_args()

    print(f"Repo root: {REPO_ROOT}")
    print(f"Sampling n={args.n} SNOMED terms (seed={args.seed})...")
    payload = run_experiment(n=args.n, seed=args.seed, model_name=args.model)
    summary = payload["summary"]
    per_term = payload["per_term"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "threshold_experiment_summary.json"
    per_path = OUT_DIR / "threshold_experiment_per_term.json"
    report_path = OUT_DIR / "threshold_experiment_report.md"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_path.write_text(json.dumps(per_term, indent=2), encoding="utf-8")
    write_markdown_report(summary, report_path)

    rec = summary["threshold_recommendation"]
    print("\n" + "=" * 60)
    print(rec["justification"])
    print("=" * 60)
    print(f"Recommended min_similarity : {rec['recommended_min_similarity']}")
    print(f"Recommended high_confidence: {rec['recommended_high_confidence']}")
    print(f"\nWrote:\n  {summary_path}\n  {per_path}\n  {report_path}")


if __name__ == "__main__":
    main()
