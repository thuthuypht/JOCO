# JOCO Step 3B — corrected full base rerun

The full corrected rerun completed **1,200/1,200 images** with zero recorded execution errors:

- COCO: 500 images
- Visual Genome: 400 images
- Open Images: 300 images

## Important empirical finding

Under the released base candidate-generation implementation, Detector top-1,
Detector + local ontology score, and OCOSL produce identical object-label
Precision/Recall/F1 on all three datasets. This is not a runtime error.

The released `build_candidates()` function starts from the detector top-1 label,
optionally adds an ontology alias, and adds `physical object`; it does not expose
true alternative detector class scores for each box. Consequently, the clean
ontology score and global solver usually cannot switch a detection to a different
benchmark class.

Therefore the manuscript must NOT retain earlier claims that the base geometry
pipeline improves Label F1 (e.g., 0.88→0.90, 0.81→0.84, 0.76→0.79) unless a
genuinely different candidate generator is implemented and rerun.

## Corrected object-label results

- COCO micro-F1: 0.573705
- Visual Genome micro-F1: 0.268072
- Open Images micro-F1: 0.222101

Detector, local ontology, and OCOSL are identical for these metrics.

## Base geometry relation results

The detector-compatible / geometry-expressible evaluation denominator is very
small after protocol filtering:

- Visual Genome: 18 eligible relation triplets out of 8,797 raw relationships
- Open Images: 2 eligible relation triplets out of 867 raw object-object relations

No exact relation matches were obtained in this base geometry rerun, so the
base geometry relation F1 is 0. The stronger OI-trained RelTR V3 experiment
should remain the primary quantitative relation-accuracy robustness experiment.

## Constraint / runtime results

- Violations: 0 on all 1,200 images
- Constraint compliance: 100%
- Fallback count: 0
- Median solver time:
  - COCO: 0.00729 s
  - Visual Genome: 0.00753 s
  - Open Images: 0.00660 s

The average per-image traceability percentage is below 100% only because images
with zero detector outputs have no assertions to explain. Conditional on images
with at least one detected object, traceability is 100% for all three datasets.

## Scientific interpretation

These results support the claims that the final solver is fast, feasible,
constraint-consistent, and traceable. They do **not** support a claim of improved
base object-label F1 over Detector top-1, nor a positive base geometry relation F1.

The separate RelTR V3 experiment remains important because it evaluates a learned
relation generator on all 867 Open Images relations and showed nearly unchanged
relation F1 after hierarchy-aware OCOSL filtering.
