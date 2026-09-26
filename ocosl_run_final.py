#!/usr/bin/env python3
"""Final OCOSL wrapper for the clean base-experiment reruns.

The already-completed RelTR V3 experiment remains frozen and is reproduced
separately by run_oi300_reltr_ocosl_v3.py.
"""
from __future__ import annotations

from ocosl_utils_v3 import (
    load_ontology_schema, build_relation_type_constraints, build_synonym_table,
    DEFAULT_TYPE_MAP, label_to_type, enrich_graph
)
from src.ontology_scoring_final import (
    build_ontology_score_clean, FINAL_ONTOLOGY_SCORE_WEIGHTS
)
from src.ocosl_constraints_final import (
    build_class_index, build_disjoint_pairs, make_label_compatible,
    solve_with_decomposition_repair, instance_statistics
)
from src.diagnostics_traceability import (
    diagnostic_violations, traceability_records, traceability_summary
)

FINAL_ALPHA = 0.40
FINAL_BETA = 0.30
FINAL_GAMMA = 0.20
FINAL_LAMBDA_CONFLICT = 0.0
FINAL_TIME_LIMIT_S = 10
FINAL_GAP_REL = 1.0e-4

def initialize(ontology_path):
    g, base = load_ontology_schema(str(ontology_path))
    rel_constraints = build_relation_type_constraints(g, base)
    synonyms = build_synonym_table(g)
    type_map = dict(DEFAULT_TYPE_MAP)
    class_index = build_class_index(g)
    compat = make_label_compatible(
        g, class_index, synonyms, type_map, label_to_type
    )
    disjoint = build_disjoint_pairs(g)
    return g, base, rel_constraints, synonyms, type_map, compat, disjoint

def clean_ontology_score(det_label, cand_label, synonym_map, type_map, rel_support=0.0):
    """Ontology-only score; detector confidence is deliberately excluded."""
    return build_ontology_score_clean(
        det_label=det_label,
        cand_label=cand_label,
        synonym_map=synonym_map,
        type_map=type_map,
        rel_support=rel_support,
    )

def rescore_objects_clean(objects, synonym_map, type_map, rel_support_by_object=None):
    """Replace DetObj.onto_scores in place with the clean ontology-only score."""
    rel_support_by_object = rel_support_by_object or {}
    for obj in objects:
        scores = {}
        for cand_label, _sdet in obj.candidates:
            scores[cand_label] = clean_ontology_score(
                obj.det_label,
                cand_label,
                synonym_map,
                type_map,
                rel_support=float(rel_support_by_object.get(obj.idx, 0.0)),
            )
        obj.onto_scores = scores
    return objects

def solve_final(
    objects, rel_cands, ontology_path,
    alpha=FINAL_ALPHA, beta=FINAL_BETA, gamma=FINAL_GAMMA,
    lambda_conflict=FINAL_LAMBDA_CONFLICT,
    time_limit_s=FINAL_TIME_LIMIT_S, rel_budget=0,
    gap_rel=FINAL_GAP_REL, image_id=None,
):
    g, base, rel_constraints, synonyms, type_map, compat, disjoint = initialize(
        ontology_path
    )
    rescore_objects_clean(objects, synonyms, type_map)

    stats = instance_statistics(
        objects, rel_cands, rel_constraints, compat, rel_budget, disjoint
    )

    x, y, log = solve_with_decomposition_repair(
        objects, rel_cands, rel_constraints, compat,
        alpha, beta, gamma, lambda_conflict,
        time_limit_s, rel_budget, disjoint, gap_rel
    )

    diag = diagnostic_violations(
        objects, x, y, rel_constraints, compat, rel_budget
    )
    log.violations_after_repair = diag["violations"]
    labels, rels = enrich_graph(objects, x, y)
    evidence = traceability_records(
        objects, x, y, rel_cands, rel_constraints, compat, image_id
    )

    return {
        "x_star": x,
        "y_star": y,
        "selected_labels": labels,
        "enriched_relations": rels,
        "solver_log": log,
        "instance_statistics": stats,
        "diagnostics": diag,
        "traceability": evidence,
        "traceability_summary": traceability_summary(evidence),
        "ontology_score_weights": dict(FINAL_ONTOLOGY_SCORE_WEIGHTS),
        "active_objective": {
            "alpha_detector": alpha,
            "beta_ontology": beta,
            "gamma_relation": gamma,
            "lambda_conflict": lambda_conflict,
        },
        "cbc": {"time_limit_s": time_limit_s, "gap_rel": gap_rel},
    }
