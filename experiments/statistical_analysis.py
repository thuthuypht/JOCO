#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired permutation test from per-image CSV."""
import argparse,json,numpy as np,pandas as pd
def boot(x,n=5000,seed=42):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); b=[rng.choice(x,len(x),replace=True).mean() for _ in range(n)]
    return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]
def perm(a,b,n=20000,seed=42):
    d=np.asarray(b,float)-np.asarray(a,float)
    if np.allclose(d,0): return 1.0
    obs=abs(d.mean()); rng=np.random.default_rng(seed); c=0
    for _ in range(n):
        if abs((d*rng.choice([-1.,1.],len(d))).mean())>=obs-1e-15: c+=1
    return (c+1)/(n+1)
def main():
    p=argparse.ArgumentParser(); p.add_argument("csv"); p.add_argument("--baseline",required=True); p.add_argument("--proposed",required=True)
    p.add_argument("--out",default=""); p.add_argument("--bootstrap",type=int,default=5000); p.add_argument("--permutations",type=int,default=20000); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); df=pd.read_csv(a.csv); x=df[a.baseline].astype(float).values; y=df[a.proposed].astype(float).values
    o={"n":len(df),"baseline_mean":float(x.mean()),"proposed_mean":float(y.mean()),"mean_effect":float((y-x).mean()),
       "baseline_ci95":boot(x,a.bootstrap,a.seed),"proposed_ci95":boot(y,a.bootstrap,a.seed),"paired_permutation_p":float(perm(x,y,a.permutations,a.seed))}
    print(json.dumps(o,indent=2))
    if a.out: Path(a.out).write_text(json.dumps(o,indent=2))
if __name__=="__main__":
    from pathlib import Path
    main()
