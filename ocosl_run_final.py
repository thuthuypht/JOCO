#!/usr/bin/env python3
"""Final solver wrapper matching the revised Algorithm 1 repository description."""
from ocosl_utils_v3 import load_ontology_schema,build_relation_type_constraints,build_synonym_table,DEFAULT_TYPE_MAP,label_to_type,enrich_graph
from src.ocosl_constraints_final import build_class_index,build_disjoint_pairs,make_label_compatible,solve_with_decomposition_repair
from src.diagnostics_traceability import diagnostic_violations,traceability_records,traceability_summary

def initialize(ontology_path):
    g,b=load_ontology_schema(str(ontology_path)); rc=build_relation_type_constraints(g,b); syn=build_synonym_table(g); tm=dict(DEFAULT_TYPE_MAP)
    ci=build_class_index(g); compat=make_label_compatible(g,ci,syn,tm,label_to_type); dis=build_disjoint_pairs(g)
    return g,b,rc,syn,tm,compat,dis

def solve_final(objects,rel_cands,ontology_path,alpha=.40,beta=.30,gamma=.20,lambda_conflict=.10,time_limit_s=10,rel_budget=0,image_id=None):
    g,b,rc,syn,tm,compat,dis=initialize(ontology_path)
    x,y,log=solve_with_decomposition_repair(objects,rel_cands,rc,compat,alpha,beta,gamma,lambda_conflict,time_limit_s,rel_budget,dis)
    diag=diagnostic_violations(objects,x,y,rc,compat,rel_budget); log.violations_after_repair=diag["violations"]
    labels,rels=enrich_graph(objects,x,y); evidence=traceability_records(objects,x,y,rel_cands,rc,compat,image_id)
    return {"x_star":x,"y_star":y,"selected_labels":labels,"enriched_relations":rels,"solver_log":log,
            "diagnostics":diag,"traceability":evidence,"traceability_summary":traceability_summary(evidence)}
