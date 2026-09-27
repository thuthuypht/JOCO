# JOCO Step 2 dense-scene stress-test interpretation

The stress test evaluates synthetic dense candidate graphs at n = 20, 40, and 60
objects, with seeds 42, 123, and 2025.

## Main results

- All 9 runs returned `Optimal`.
- No fallback was triggered.
- The configured CBC relative-gap tolerance was `gapRel = 1e-4`.
- Therefore the solver terminated within the configured relative-gap criterion
  (`final relative gap <= 1e-4` for runs reported Optimal).
- The original notebook column `objective_gap_if_optimal = 0.0` is a convention
  inserted by the notebook, not a separately parsed CBC log value. In the paper,
  report the bounded statement above rather than claiming a measured numerical
  gap of exactly zero.

The first n=20/seed=42 run shows a one-time solver startup effect
(~0.373 s versus ~0.019 s for the other two n=20 runs). For transparency,
the repository retains all per-run values and reports both mean±std and median.

## Final stress-test summary

Use `dense_scene_stress_final_summary.csv` for manuscript reporting.
