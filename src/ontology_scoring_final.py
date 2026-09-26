#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final auditable ontology score for JOCO base reruns.

Detector confidence is intentionally NOT part of s_onto. Detector evidence
enters the global objective only through alpha * s_det, eliminating the
double-counting noted in the revised manuscript.

The released legacy implementation actually computed three non-detector
components:
    lexical mapping, coarse type/hierarchy compatibility, relation/context support.

Their legacy relative weights were 0.25 : 0.15 : 0.05 = 5 : 3 : 1.
After removing detector confidence, these active terms are renormalized to
sum to one:
    lexical   = 5/9 = 0.555555...
    hierarchy = 3/9 = 0.333333...
    context   = 1/9 = 0.111111...

No unsupported embedding-semantic or ontology-prior term is silently invented.
"""
from __future__ import annotations
from typing import Dict, Tuple

from ocosl_utils_v3 import lexical_score, label_to_type

FINAL_ONTOLOGY_SCORE_WEIGHTS = {
    "lexical": 5.0 / 9.0,
    "hierarchy": 3.0 / 9.0,
    "context": 1.0 / 9.0,
}

def build_ontology_score_clean(
    det_label: str,
    cand_label: str,
    synonym_map: Dict[str, str],
    type_map: Dict[str, str],
    rel_support: float = 0.0,
    weights: Tuple[float, float, float] = (
        FINAL_ONTOLOGY_SCORE_WEIGHTS["lexical"],
        FINAL_ONTOLOGY_SCORE_WEIGHTS["hierarchy"],
        FINAL_ONTOLOGY_SCORE_WEIGHTS["context"],
    ),
) -> float:
    """Compute ontology-only compatibility score s_onto in [0,1]."""
    w_lex, w_hier, w_ctx = [float(x) for x in weights]
    total = w_lex + w_hier + w_ctx
    if total <= 0:
        raise ValueError("Ontology-score weights must have positive sum.")
    w_lex, w_hier, w_ctx = (w_lex / total, w_hier / total, w_ctx / total)

    lex = float(lexical_score(det_label, cand_label))
    t_det = label_to_type(det_label, synonym_map, type_map)
    t_cand = label_to_type(cand_label, synonym_map, type_map)
    hier = 1.0 if t_det == t_cand else 0.4
    ctx = max(0.0, min(1.0, float(rel_support)))

    score = w_lex * lex + w_hier * hier + w_ctx * ctx
    return max(0.0, min(1.0, float(score)))
