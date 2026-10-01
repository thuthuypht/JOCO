# JOCO Step 7A V2 — corrected ontology-sensitivity smoke test

The corrected smoke test completed all 144 expected variant rows with zero
execution errors.

## Why V2 was required
The earlier V1 perturbed `synonym_map`, but the released ontology produced an
empty synonym/altLabel table. V2 instead perturbs the active
detector/object-label -> ontology-type mapping (`type_map`) together with
relation domain/range constraints.

## V2 perturbations
- Missing knowledge: remove 10%, 20%, and 30%.
- Erroneous knowledge: corrupt 5% and 10%.
- Seeds: 42, 123, 2025.

Across the 9 smoke images, V2 actually perturbed:
- removed type mappings: 486 row-level perturbation counts across all variants,
- corrupted type mappings: 135,
- removed relation constraints: 621,
- corrupted relation constraints: 162.

The smoke test showed no object-label changes and no reference-ontology
violations, while some erroneous conditions changed selected relation sets.
This justifies running the full Step 7B sensitivity experiment.
