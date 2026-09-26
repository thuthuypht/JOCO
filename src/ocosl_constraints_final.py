#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hierarchy-aware OCOSL constraints and decomposition/repair fallback."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Set
import time, re
from rdflib import RDF, RDFS, OWL, URIRef

@dataclass
class FallbackLog:
    status: str
    trigger: str
    used_fallback: bool
    solver_time_s: float
    repair_time_s: float
    total_time_s: float
    n_components: int
    objective_value: Optional[float] = None
    violations_after_repair: Optional[int] = None

def _norm(x: object) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")

def _local(uri: object) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[-1] if "#" in s else s.rsplit("/", 1)[-1]

def build_class_index(g) -> Dict[str, URIRef]:
    idx: Dict[str, URIRef] = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue
        idx[_norm(_local(cls))] = cls
        for lbl in g.objects(cls, RDFS.label):
            idx[_norm(lbl)] = cls
    return idx

def is_subclass_or_same(g, class_index: Dict[str, URIRef], child: str, parent: str) -> bool:
    if parent in (None, "DetectedObject"):
        return True
    cu = class_index.get(_norm(child))
    pu = class_index.get(_norm(parent))
    if cu is None or pu is None:
        return False
    if cu == pu:
        return True
    seen: Set[URIRef] = set()
    stack = [cu]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for par in g.objects(cur, RDFS.subClassOf):
            if not isinstance(par, URIRef):
                continue
            if par == pu:
                return True
            stack.append(par)
    return False

def build_disjoint_pairs(g) -> Set[frozenset]:
    pairs: Set[frozenset] = set()
    for a, b in g.subject_objects(OWL.disjointWith):
        if isinstance(a, URIRef) and isinstance(b, URIRef):
            pairs.add(frozenset((_local(a), _local(b))))
    return pairs

def candidate_names(label: str, synonym_map: dict, type_map: dict) -> List[str]:
    out = [str(label)]
    for k in [label, str(label).lower(), _norm(label)]:
        if isinstance(synonym_map, dict) and k in synonym_map and synonym_map[k] is not None:
            out.append(_local(synonym_map[k]))
        if isinstance(type_map, dict) and k in type_map and type_map[k] is not None:
            out.append(str(type_map[k]))
    dedup = []
    for x in out:
        if x not in dedup:
            dedup.append(x)
    return dedup

def make_label_compatible(g, class_index, synonym_map, type_map,
                          legacy_label_to_type: Optional[Callable] = None):
    def compatible(label: str, required_type: Optional[str]) -> bool:
        if required_type in (None, "DetectedObject"):
            return True
        if required_type == "PhysicalObject":
            return True
        for name in candidate_names(label, synonym_map, type_map):
            if is_subclass_or_same(g, class_index, name, required_type):
                return True
        if legacy_label_to_type is not None:
            try:
                mapped = legacy_label_to_type(label, synonym_map, type_map)
                if mapped == required_type or is_subclass_or_same(g, class_index, mapped, required_type):
                    return True
            except Exception:
                pass
        return False
    return compatible

def _components(objects, rel_cands):
    nodes = [o.idx for o in objects]
    adj = {i: set() for i in nodes}
    for rc in rel_cands:
        if rc.feasible and rc.i in adj and rc.j in adj:
            adj[rc.i].add(rc.j); adj[rc.j].add(rc.i)
    comps=[]; unseen=set(nodes)
    while unseen:
        start=min(unseen); unseen.remove(start); stack=[start]; comp=[]
        while stack:
            u=stack.pop(); comp.append(u)
            for v in sorted(adj[u]):
                if v in unseen:
                    unseen.remove(v); stack.append(v)
        comps.append(sorted(comp))
    return comps

def solve_ilp_hierarchy(objects, rel_cands, rel_type_constraints, label_compatible,
                        alpha=0.40, beta=0.30, gamma=0.20, lambda_conflict=0.0,
                        time_limit_s=10, rel_budget=0, disjoint_pairs=None,
                        gap_rel=1.0e-4, msg=False):
    import pulp
    prob = pulp.LpProblem("OCOSL_HierarchyAware", pulp.LpMaximize)
    x={}
    for o in objects:
        for lbl,_ in o.candidates:
            safe=re.sub(r"[^A-Za-z0-9_]+","_",str(lbl))
            x[(o.idx,lbl)] = pulp.LpVariable(f"x_{o.idx}_{safe}",0,1,cat=pulp.LpBinary)
    for o in objects:
        prob += pulp.lpSum(x[(o.idx,lbl)] for lbl,_ in o.candidates)==1

    disjoint_pairs = disjoint_pairs or set()
    for o in objects:
        labels=[lbl for lbl,_ in o.candidates]
        for a in range(len(labels)):
            for b in range(a+1,len(labels)):
                if frozenset((labels[a],labels[b])) in disjoint_pairs:
                    prob += x[(o.idx,labels[a])] + x[(o.idx,labels[b])] <= 1

    y={}
    for rc in rel_cands:
        if rc.feasible:
            safe=re.sub(r"[^A-Za-z0-9_]+","_",str(rc.rel))
            y[(rc.i,rc.j,rc.rel)] = pulp.LpVariable(f"y_{rc.i}_{rc.j}_{safe}",0,1,cat=pulp.LpBinary)

    budget=rel_budget if rel_budget>0 else max(1,2*len(objects))
    if y:
        prob += pulp.lpSum(y.values()) <= int(budget)

    pairs={}
    for k in y:
        pairs.setdefault((k[0],k[1]),[]).append(k)
    for keys in pairs.values():
        prob += pulp.lpSum(y[k] for k in keys) <= 1

    by_idx={o.idx:o for o in objects}
    for (i,j,r),var in y.items():
        dom,ran=rel_type_constraints.get(r,(None,None))
        if dom not in (None,"DetectedObject"):
            allowed=[x[(i,lbl)] for lbl,_ in by_idx[i].candidates if label_compatible(lbl,dom)]
            prob += var <= pulp.lpSum(allowed)
        if ran not in (None,"DetectedObject"):
            allowed=[x[(j,lbl)] for lbl,_ in by_idx[j].candidates if label_compatible(lbl,ran)]
            prob += var <= pulp.lpSum(allowed)

    terms=[]
    for o in objects:
        for lbl,sdet in o.candidates:
            sonto=o.onto_scores.get(lbl,sdet)
            terms.append((alpha*float(sdet)+beta*float(sonto))*x[(o.idx,lbl)])
    score={(r.i,r.j,r.rel):float(r.score) for r in rel_cands if r.feasible}
    for k,var in y.items():
        terms.append(gamma*score.get(k,0.0)*var)
    # Conflicts are hard feasibility constraints in the clean rerun.
    # Therefore no additional soft conflict penalty is active.
    prob += pulp.lpSum(terms)

    solver=pulp.PULP_CBC_CMD(msg=msg, timeLimit=int(time_limit_s), gapRel=float(gap_rel))
    prob.solve(solver)
    status=pulp.LpStatus.get(prob.status,str(prob.status))
    x_star={(i,l):int((pulp.value(v) or 0)>0.5) for (i,l),v in x.items()}
    y_star={(i,j,r):int((pulp.value(v) or 0)>0.5) for (i,j,r),v in y.items()}
    return x_star,y_star,status,pulp.value(prob.objective)

def deterministic_repair(objects, rel_cands, rel_type_constraints, label_compatible,
                         alpha=0.40,beta=0.30,gamma=0.20,rel_budget=0):
    chosen={}; x={}
    for o in objects:
        ranked=[(alpha*float(s)+beta*float(o.onto_scores.get(lbl,s)),str(lbl),lbl)
                for lbl,s in o.candidates]
        best=max(ranked)[2]; chosen[o.idx]=best
        for lbl,_ in o.candidates: x[(o.idx,lbl)]=int(lbl==best)
    budget=rel_budget if rel_budget>0 else max(1,2*len(objects))
    y={}; seen=set()
    for rc in sorted([r for r in rel_cands if r.feasible],
                     key=lambda z:(-float(z.score),z.i,z.j,str(z.rel))):
        if len(y)>=budget or (rc.i,rc.j) in seen: continue
        dom,ran=rel_type_constraints.get(rc.rel,(None,None))
        if dom not in (None,"DetectedObject") and not label_compatible(chosen[rc.i],dom): continue
        if ran not in (None,"DetectedObject") and not label_compatible(chosen[rc.j],ran): continue
        y[(rc.i,rc.j,rc.rel)]=1; seen.add((rc.i,rc.j))
    return x,y

def solve_with_decomposition_repair(objects, rel_cands, rel_type_constraints, label_compatible,
                                    alpha=0.40,beta=0.30,gamma=0.20,lambda_conflict=0.0,
                                    time_limit_s=10,rel_budget=0,disjoint_pairs=None,
                                    gap_rel=1.0e-4):
    start=time.perf_counter()
    try:
        x,y,status,obj=solve_ilp_hierarchy(objects,rel_cands,rel_type_constraints,label_compatible,
            alpha,beta,gamma,lambda_conflict,time_limit_s,rel_budget,disjoint_pairs,gap_rel)
        solve_dt=time.perf_counter()-start
        if status=="Optimal":
            return x,y,FallbackLog(status,"none",False,solve_dt,0.0,solve_dt,1,obj)
        trigger="timeout_or_nonoptimal"
    except Exception:
        solve_dt=time.perf_counter()-start; trigger="solver_exception"

    repair_start=time.perf_counter()
    comps=_components(objects,rel_cands); by_idx={o.idx:o for o in objects}
    x_all={}; y_all={}
    for comp in comps:
        s=set(comp); c_objs=[by_idx[i] for i in comp]
        c_rels=[r for r in rel_cands if r.i in s and r.j in s]
        try:
            cx,cy,cs,_=solve_ilp_hierarchy(c_objs,c_rels,rel_type_constraints,label_compatible,
                alpha,beta,gamma,lambda_conflict,max(1,int(time_limit_s/max(1,len(comps)))),0,disjoint_pairs,gap_rel)
            if cs!="Optimal": raise RuntimeError(cs)
        except Exception:
            cx,cy=deterministic_repair(c_objs,c_rels,rel_type_constraints,label_compatible,
                                       alpha,beta,gamma,0)
        x_all.update(cx); y_all.update(cy)

    budget=rel_budget if rel_budget>0 else max(1,2*len(objects))
    chosen=[k for k,v in y_all.items() if v==1]
    if len(chosen)>budget:
        score={(r.i,r.j,r.rel):float(r.score) for r in rel_cands}
        keep=set(sorted(chosen,key=lambda k:score.get(k,0.0),reverse=True)[:budget])
        y_all={k:int(k in keep) for k in y_all}
    repair_dt=time.perf_counter()-repair_start
    total=time.perf_counter()-start
    return x_all,y_all,FallbackLog("FallbackRepaired",trigger,True,solve_dt,repair_dt,total,len(comps),None)


def instance_statistics(objects, rel_cands, rel_type_constraints, label_compatible,
                        rel_budget=0, disjoint_pairs=None):
    """Count binary variables and instantiated hard constraints without solving."""
    disjoint_pairs = disjoint_pairs or set()
    label_vars = sum(len(o.candidates) for o in objects)
    feasible_rels = [r for r in rel_cands if r.feasible]
    relation_vars = len(feasible_rels)

    constraints = len(objects)  # exactly-one label
    for o in objects:
        labels = [lbl for lbl, _ in o.candidates]
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                if frozenset((labels[a], labels[b])) in disjoint_pairs:
                    constraints += 1

    if relation_vars:
        constraints += 1  # global relation budget
    constraints += len({(r.i, r.j) for r in feasible_rels})  # <=1 relation/pair

    domain_constraints = 0
    range_constraints = 0
    for r in feasible_rels:
        dom, ran = rel_type_constraints.get(r.rel, (None, None))
        if dom not in (None, "DetectedObject"):
            domain_constraints += 1
        if ran not in (None, "DetectedObject"):
            range_constraints += 1
    constraints += domain_constraints + range_constraints

    budget = rel_budget if rel_budget > 0 else max(1, 2 * len(objects))
    return {
        "n_objects": int(len(objects)),
        "n_label_variables": int(label_vars),
        "n_relation_variables": int(relation_vars),
        "n_binary_variables": int(label_vars + relation_vars),
        "n_constraints": int(constraints),
        "n_domain_constraints": int(domain_constraints),
        "n_range_constraints": int(range_constraints),
        "relation_budget": int(budget),
    }
