# Remaining runs required before removing every red manuscript warning

The new files close repository-level implementation gaps, but they do not fabricate missing results.
Generate and commit actual outputs for:

1. Table 4: solver-only time and integer fallback count for COCO/VG/OI base experiments.
2. Table 5: fallback trigger breakdown, repair time, total fallback-case time, objective loss, size-bin analysis.
3. Visual Genome: exact 400-image manifest and 150-class vocabulary. If the original subset cannot be recovered,
   regenerate it and rerun/update the manuscript.
4. Pairwise CRF: per-image outputs, 95% bootstrap CIs and paired tests.
5. Detector-confidence-removed ontology-score ablation.
6. Dense-scene stress test.
7. Figure 4 redraw after CRF and uncertainty outputs exist.
8. Base-run violation/traceability evidence logs.

Do not replace [FROM LOGS] with assumed values.
