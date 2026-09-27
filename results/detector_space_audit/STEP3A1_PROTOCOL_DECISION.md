# JOCO Step 3A.1 detector-space audit

The 9-image Step-3A smoke test completed without runtime errors, but its
full-vocabulary VG/OI object-label F1 values are not the manuscript protocol.

The revised manuscript states that Visual Genome and Open Images object-label
evaluation must be restricted to GT instances whose ontology-mapped concept lies
within the 80-class YOLOv8n detector output space.

This audit creates the exact non-fuzzy eligible class lists required to implement
that protocol.

Class-level detector-space coverage:
- COCO: 79/79
- Visual Genome: 22/150
- Open Images: 28/209

Visual Genome relation mapping:
- unique predicates mapped: 17/652
- raw relation instances covered by the current predicate map:
  4073/8796 (46.31%)

The next full rerun must:
1. filter VG/OI object-label GT to these eligible detector-space classes;
2. compare predicted/GT labels after the same ontology canonicalization;
3. separately report full raw GT counts and eligible evaluation counts;
4. retain all relation GT in the declared relation denominator only where the
   corresponding relation protocol explicitly requires it (e.g. RelTR V3).
