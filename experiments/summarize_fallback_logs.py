#!/usr/bin/env python3
"""Aggregate fallback JSONL logs into Table-5 fields."""
import argparse,json,pandas as pd
def main():
    p=argparse.ArgumentParser(); p.add_argument("jsonl"); p.add_argument("--out",default="fallback_summary.csv"); a=p.parse_args()
    rows=[json.loads(x) for x in open(a.jsonl,encoding="utf-8") if x.strip()]; df=pd.DataFrame(rows)
    if df.empty: raise SystemExit("No log rows")
    out={"images":len(df),"fallback_count":int(df.get("used_fallback",False).astype(bool).sum())}
    trig=df.get("trigger",pd.Series(["none"]*len(df))); out["timeout_or_nonoptimal"]=int((trig=="timeout_or_nonoptimal").sum()); out["solver_exception"]=int((trig=="solver_exception").sum())
    for c in ["repair_time_s","total_time_s","objective_loss_pct","violations_after_repair"]:
        if c in df: out["avg_"+c]=float(pd.to_numeric(df[c],errors="coerce").mean())
    pd.DataFrame([out]).to_csv(a.out,index=False); print(pd.DataFrame([out]).to_string(index=False))
if __name__=="__main__": main()
