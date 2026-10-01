# JOCO Step 6B — ontology-score double-counting ablation

The full ablation completed 1,200/1,200 images with zero execution errors.

## Compared formulations
- Clean score: detector confidence is excluded from `s_onto`; the implemented
  non-detector components are lexical / hierarchy / context in normalized 5:3:1 ratio.
- Legacy score: detector confidence is also included inside `s_onto` with
  weights 0.55 / 0.25 / 0.15 / 0.05.

## Main result
Across COCO-500, Visual Genome-400, and Open Images-300:
- micro Precision/Recall/F1 are identical between clean and legacy variants;
- selected-label changes = 0;
- selected-relation symmetric differences = 0;
- paired per-image F1 effect = 0;
- paired permutation p-value = 1.0;
- violations = 0 for both variants;
- fallback count = 0 for both variants.

Thus, removing detector confidence from `s_onto` eliminates conceptual
double-counting without changing any selected solution on the 1,200-image
evaluation set.

Objective magnitudes differ because the objective definitions differ and must
not be interpreted as a performance gain/loss.
