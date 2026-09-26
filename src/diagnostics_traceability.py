#!/usr/bin/env python3
"""Post-hoc violation and traceability metrics for JOCO."""
def selected_labels(objects,x_star):
    out={}
    for o in objects:
        for lbl,_ in o.candidates:
            if x_star.get((o.idx,lbl),0)==1:
                out[o.idx]=lbl; break
    return out

def selected_relations(y_star):
    return [(i,j,r) for (i,j,r),v in y_star.items() if v==1]

def diagnostic_violations(objects,x_star,y_star,rel_type_constraints,label_compatible,rel_budget=0):
    V=0; Q=0; details=[]; labels=selected_labels(objects,x_star)
    for o in objects:
        Q+=1; c=sum(x_star.get((o.idx,l),0) for l,_ in o.candidates)
        if c!=1: V+=1; details.append({"type":"one_label","object":o.idx,"count":c})
    pair_count={}
    rels=selected_relations(y_star)
    for i,j,r in rels:
        pair_count[(i,j)]=pair_count.get((i,j),0)+1
        dom,ran=rel_type_constraints.get(r,(None,None))
        if dom not in (None,"DetectedObject"):
            Q+=1
            if not label_compatible(labels.get(i,""),dom):
                V+=1; details.append({"type":"domain","i":i,"j":j,"rel":r,"required":dom})
        if ran not in (None,"DetectedObject"):
            Q+=1
            if not label_compatible(labels.get(j,""),ran):
                V+=1; details.append({"type":"range","i":i,"j":j,"rel":r,"required":ran})
    for pair,c in pair_count.items():
        Q+=1
        if c>1: V+=1; details.append({"type":"pair_cardinality","pair":pair,"count":c})
    C=rel_budget if rel_budget>0 else max(1,2*len(objects)); Q+=1
    if len(rels)>C: V+=1; details.append({"type":"relation_budget","selected":len(rels),"budget":C})
    return {"violations":V,"applicable_checks":Q,
            "constraint_compliance_pct":100*(1-V/Q) if Q else 100.0,
            "details":details}

def traceability_records(objects,x_star,y_star,rel_cands,rel_type_constraints,label_compatible,image_id=None):
    labels=selected_labels(objects,x_star)
    rel_score={(r.i,r.j,r.rel):float(r.score) for r in rel_cands}
    rows=[]
    for o in objects:
        lbl=labels.get(o.idx)
        if lbl is not None:
            rows.append({"image_id":image_id,"assertion_type":"label","subject":o.idx,
                         "predicate":"rdf:type","object":lbl,"detector_score":float(o.det_conf),
                         "ontology_score":float(o.onto_scores.get(lbl,0.0)),
                         "relation_score":None,"domain_range_ok":True,"evidence_items":2})
    for i,j,r in selected_relations(y_star):
        dom,ran=rel_type_constraints.get(r,(None,None))
        d=True if dom in (None,"DetectedObject") else label_compatible(labels.get(i,""),dom)
        q=True if ran in (None,"DetectedObject") else label_compatible(labels.get(j,""),ran)
        rows.append({"image_id":image_id,"assertion_type":"relation","subject":i,"predicate":r,"object":j,
                     "detector_score":None,"ontology_score":None,"relation_score":rel_score.get((i,j,r)),
                     "domain_range_ok":bool(d and q),
                     "evidence_items":int((i,j,r) in rel_score)+int(d and q)})
    return rows

def traceability_summary(records):
    n=len(records)
    if not n:
        return {"assertions":0,"traceability_rate_pct":0.0,
                "explanation_completeness_pct":0.0,"avg_evidence_items_per_assertion":0.0}
    tr=sum(int(r.get("evidence_items",0))>0 for r in records)
    comp=sum(int(r.get("evidence_items",0))>=1 for r in records)
    avg=sum(int(r.get("evidence_items",0)) for r in records)/n
    return {"assertions":n,"traceability_rate_pct":100*tr/n,
            "explanation_completeness_pct":100*comp/n,
            "avg_evidence_items_per_assertion":avg}
