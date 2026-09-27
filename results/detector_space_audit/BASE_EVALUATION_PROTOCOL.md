# Base-experiment evaluation protocol after Step 3A.1

The detector is YOLOv8n trained on COCO-80. Therefore object-label metrics on
Visual Genome and Open Images are computed only on GT instances whose
deterministic ontology/exact-label mapping reaches the detector output space.

Frozen class-level coverage:
- COCO: 79/79 classes occurring in COCO-500 are eligible.
- Visual Genome: 22/150 selected VG classes are detector-eligible.
- Open Images: 28/209 classes occurring in OI-300 are detector-eligible.

For the geometry-driven base relation experiment, relation evaluation is also
restricted to predicates expressible by the geometry relation generator and to
relations whose two GT endpoints are detector-eligible. Raw full-dataset counts
are reported separately. The stronger OI-trained RelTR V3 experiment remains
unchanged and keeps all 867 OI object-object relations in its denominator.

This separation prevents the COCO-80 detector from being penalized for object
classes or semantic interaction predicates that its base geometry pipeline
cannot generate.
