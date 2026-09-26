#!/usr/bin/env python3
"""Synthetic dense-scene solver stress test for n=20,40,60 (solver-only)."""
import argparse,random,time,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from ocosl_utils_v3 import DetObj,RelCand
from src.ocosl_constraints_final import solve_ilp_hierarchy
def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default="results/dense_scene_stress.csv"); p.add_argument("--seed",type=int,default=42); a=p.parse_args(); rng=random.Random(a.seed); rows=[]
    for n in [20,40,60]:
        objs=[]
        for i in range(n):
            c=[("person",.70),("physical object",.20),("vehicle",.10)]; objs.append(DetObj(i,(0,0,10,10),"person",.70,c,{l:s for l,s in c}))
        rels=[RelCand(i,j,"near",rng.uniform(.7,1),True) for i in range(n) for j in range(n) if i!=j and rng.random()<.15]
        t=time.perf_counter(); x,y,status,obj=solve_ilp_hierarchy(objs,rels,{},lambda l,r:True,time_limit_s=10,rel_budget=2*n); dt=time.perf_counter()-t
        rows.append({"n":n,"label_variables":n*3,"relation_variables":len(rels),"total_binary_variables":n*3+len(rels),"solver_time_s":dt,"status":status,"selected_relations":sum(y.values())})
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
