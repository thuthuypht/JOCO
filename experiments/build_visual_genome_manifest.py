#!/usr/bin/env python3
"""Build the exact/reproducible Visual Genome subset manifest and counts."""
import argparse,json,random,re
from collections import Counter
from pathlib import Path
def norm(s): return re.sub(r"\s+"," ",str(s).strip().lower())
def main():
    p=argparse.ArgumentParser(); p.add_argument("--objects-json",required=True)
    p.add_argument("--relationships-json",required=True); p.add_argument("--image-ids",default="")
    p.add_argument("--out-dir",default="manifests/visual_genome"); p.add_argument("--n",type=int,default=400)
    p.add_argument("--vocab-size",type=int,default=150); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); objects=json.loads(Path(a.objects_json).read_text()); rels=json.loads(Path(a.relationships_json).read_text())
    ob={str(x["image_id"]):x for x in objects}; rb={str(x["image_id"]):x for x in rels}
    elig=sorted(set(ob)&set(rb))
    if a.image_ids: ids=[x.strip() for x in Path(a.image_ids).read_text().splitlines() if x.strip()]
    else: ids=sorted(random.Random(a.seed).sample(elig,min(a.n,len(elig))))
    c=Counter()
    for iid in ids:
        for o in ob[iid].get("objects",[]):
            names=o.get("names") or ([o["name"]] if "name" in o else [])
            if names: c[norm(names[0])]+=1
    vocab=[x for x,_ in c.most_common(a.vocab_size)]; vs=set(vocab)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"vg400_image_ids.txt").write_text("\n".join(ids)+"\n")
    (out/"vg150_object_vocabulary.txt").write_text("\n".join(vocab)+"\n")
    B=R=0
    with (out/"vg400_relationship_gt.jsonl").open("w") as fh:
        for iid in ids:
            objs={}
            for o in ob[iid].get("objects",[]):
                names=o.get("names") or ([o["name"]] if "name" in o else [])
                if not names: continue
                label=norm(names[0])
                if label in vs: objs[str(o["object_id"])]={"label":label,"x":o.get("x"),"y":o.get("y"),"w":o.get("w"),"h":o.get("h")}
            B+=len(objs); rr=[]
            for r in rb[iid].get("relationships",[]):
                sid=str((r.get("subject") or {}).get("object_id","")); oid=str((r.get("object") or {}).get("object_id",""))
                if sid in objs and oid in objs: rr.append({"subject_id":sid,"predicate":norm(r.get("predicate","")),"object_id":oid})
            R+=len(rr); fh.write(json.dumps({"image_id":iid,"objects":objs,"relationships":rr})+"\n")
    summary={"images":len(ids),"boxes_B":B,"relations_R":R,"vocabulary_size":len(vocab),"seed":a.seed,
             "manifest_source":"provided" if a.image_ids else "deterministic_generated"}
    (out/"vg400_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
