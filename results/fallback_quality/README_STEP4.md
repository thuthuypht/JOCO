# JOCO Step 4 — forced fallback-branch experiment

The Step-4 experiment completed on 30 difficult instances (10 per dataset).

## Observed result

- Reference solve status: Optimal for all 30 instances.
- Forced fallback-branch objective loss: 0.0% on all 30 instances.
- Fallback violations after the branch: 0.
- Constraint compliance: 100%.
- Mean reference solve time: ~0.049 s.
- Mean forced-fallback-branch time: ~0.055 s.

## Important interpretation

The selected candidate graphs each formed a **single connected component**
(`components_mean = 1.0` for all datasets). Consequently, the forced fallback
branch solved that single component optimally rather than invoking the final
deterministic-repair step.

Therefore this experiment is useful evidence that the released fallback branch
preserves feasibility and objective value on these difficult instances, but it
does **not by itself measure the quality of deterministic repair under an actual
timeout/infeasibility condition**.

The natural Step-3B rerun had fallback count 0/1200, so the manuscript should not
invent timeout or infeasibility counts. Natural fallback count, timeout count,
and infeasibility/pruning count are all zero in that run.

A separate repair-only quality stress test should be used before finalizing
Table 5 if the manuscript wishes to report objective degradation of the repair
heuristic itself.
