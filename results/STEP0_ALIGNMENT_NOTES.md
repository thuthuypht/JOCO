# STEP 0 — final code/config alignment

## Fixed for the remaining base COCO/VG/OI reruns

1. Detector confidence is removed from `s_onto`; it enters only through
   `alpha * s_det`.
2. No unsupported embedding-semantic or ontology-prior component is invented.
3. The actual legacy non-detector ratio `.25:.15:.05 = 5:3:1` is retained and
   normalized to lexical `5/9`, hierarchy `3/9`, context `1/9`.
4. Domain/range checking uses transitive `rdfs:subClassOf` closure.
5. CBC receives `gapRel=1e-4` and `timeLimit=10`.
6. No active soft conflict penalty is used; conflicts are hard constraints.
7. Fallback remains connected-component decomposition plus deterministic repair.
8. Actual candidate-graph variable/constraint counts are exposed for Table 4.

## Manuscript update required after reruns

JOCO_v9 Table 2 and the ontology-score equation must be changed to match this
implementation. In particular, remove the six-component ontology-score claim
unless the missing components are actually implemented and evaluated.

## Frozen RelTR V3

Do not overwrite the completed RelTR-V3 300-image result files. This Step-0
alignment applies to the remaining base geometry/CRF/ablation experiments.
