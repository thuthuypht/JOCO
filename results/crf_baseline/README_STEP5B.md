# JOCO Step 5B — full pairwise CRF baseline

The experiment completed all 1,200 images with zero execution errors.

## Main object-label result

For all three datasets, Pairwise CRF and OCOSL produced identical object-label
Precision/Recall/F1 under the released candidate pool:

- COCO micro-F1: 0.573705 for both.
- Visual Genome micro-F1: 0.268072 for both.
- Open Images micro-F1: 0.222101 for both.

The paired per-image F1 effect (CRF - OCOSL) was exactly 0 for each dataset and
the two-sided paired permutation p-value was 1.0.

The 95% bootstrap confidence intervals in `step5b_crf_table_ready.csv` are for
the **mean per-image F1**, not for micro-F1.

## Relation result

The geometry-compatible relation evaluation denominator is very small after
detector-space filtering:
- Visual Genome: 14 images with at least one eligible relation.
- Open Images: 1 image with at least one eligible relation.
- COCO: no relation GT in the instance annotations.

Both OCOSL and CRF obtained relation F1 = 0 under this base geometry protocol.
The separate OI-trained RelTR V3 experiment remains the primary learned-relation
robustness evaluation and retains all 867 OI object-object relations.

## Runtime and constraints

CRF inference was extremely fast (roughly 0.0002 s/image mean).
CRF violations were 0 and mean constraint compliance was 100%.

## Scientific interpretation

This result does not support an accuracy advantage of the pairwise CRF or of
OCOSL under the released object-label candidate pool. The useful distinction is
methodological: OCOSL provides an explicit hard-constrained global ILP, global
relation budget, auditable feasibility, and the released decomposition/repair
fallback, while the CRF baseline uses deterministic ICM approximate MAP inference.
