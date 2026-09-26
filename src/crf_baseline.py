#!/usr/bin/env python3
"""Pairwise CRF-style baseline with deterministic ICM MAP approximation."""
def pairwise_crf_icm(objects,rel_cands,rel_type_constraints,label_compatible,
                     alpha=.40,beta=.30,gamma=.20,incompatibility_penalty=.20,max_iter=25,rel_budget=0):
    by={o.idx:o for o in objects}; labels={}
    for o in objects:
        labels[o.idx]=max(o.candidates,key=lambda z:(alpha*float(z[1])+beta*float(o.onto_scores.get(z[0],z[1])),str(z[0])))[0]
    incident={o.idx:[] for o in objects}
    for rc in rel_cands:
        if rc.feasible:
            incident.setdefault(rc.i,[]).append(rc); incident.setdefault(rc.j,[]).append(rc)
    def local(node,cand):
        o=by[node]; sdet=dict(o.candidates)[cand]
        score=alpha*float(sdet)+beta*float(o.onto_scores.get(cand,sdet))
        for rc in incident.get(node,[]):
            li,lj=(cand,labels.get(rc.j)) if rc.i==node else (labels.get(rc.i),cand)
            dom,ran=rel_type_constraints.get(rc.rel,(None,None))
            od=True if dom in (None,"DetectedObject") else label_compatible(li,dom)
            orng=True if ran in (None,"DetectedObject") else label_compatible(lj,ran)
            score += gamma*float(rc.score) if od and orng else -incompatibility_penalty*float(rc.score)
        return score
    for _ in range(max_iter):
        changed=0
        for node in sorted(labels):
            o=by[node]; best=max((local(node,l),str(l),l) for l,_ in o.candidates)[2]
            if best!=labels[node]: labels[node]=best; changed+=1
        if changed==0: break
    scored=[]
    for rc in rel_cands:
        if not rc.feasible: continue
        dom,ran=rel_type_constraints.get(rc.rel,(None,None))
        od=True if dom in (None,"DetectedObject") else label_compatible(labels[rc.i],dom)
        orng=True if ran in (None,"DetectedObject") else label_compatible(labels[rc.j],ran)
        scored.append((float(rc.score)+(0.15 if od and orng else -0.15),rc))
    best={}
    for s,rc in sorted(scored,key=lambda z:z[0],reverse=True):
        best.setdefault((rc.i,rc.j),(s,rc))
    budget=rel_budget if rel_budget>0 else max(1,2*len(objects))
    selected=sorted(best.values(),key=lambda z:z[0],reverse=True)[:budget]
    x={(o.idx,l):int(l==labels[o.idx]) for o in objects for l,_ in o.candidates}
    y={(rc.i,rc.j,rc.rel):1 for _,rc in selected}
    return x,y
