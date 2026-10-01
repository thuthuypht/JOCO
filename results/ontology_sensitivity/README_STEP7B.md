# JOCO Step 7B — full ontology-sensitivity experiment

The full experiment completed 19,200/19,200 variant rows with zero execution errors.

## Protocol
- Baseline ontology.
- Missing knowledge: remove 10%, 20%, and 30% of the active
  label-to-ontology-type mappings and relation domain/range constraints.
- Erroneous knowledge: replace 5% and 10% of the same active mappings/constraints
  with incorrect values.
- Seeds: 42, 123, 2025.
- Selected solutions are checked against the unperturbed reference ontology.

## Main result
Across the tested perturbation levels:
- Object-label micro-F1 remained unchanged:
  - COCO: 0.573705
  - Visual Genome: 0.268072
  - Open Images: 0.222101
- Base geometry relation micro-F1 remained 0 under the previously defined
  geometry-compatible evaluation protocol.
- Reference-ontology diagnostic violations remained 0.
- Reference compliance remained 100%.
- No fallback was triggered.

Missing-knowledge perturbations caused no selected-label or selected-relation
changes in this experiment. Erroneous-knowledge perturbations changed selected
relation sets for some seeds, but did not change object-label F1, relation F1,
or reference-ontology violation counts.

The result should be interpreted as empirical robustness over the tested
perturbation range and the released candidate pool, not as a general robustness
guarantee against arbitrary ontology corruption.
