# JOCO Step 1 dataset audit artifacts

These files freeze the dataset subsets and audit statistics used for the remaining reruns.

## Audited subsets

- COCO-500: 500 images, 3,505 GT boxes, 79 used GT classes.
- Visual Genome-400: new deterministic manifest (seed=42), 400 images,
  9,597 raw object instances and 8,797 raw relationships; top-150 object
  vocabulary stored in `manifests/vg150_class_list.csv`.
- Open Images-300: 300 images, 5,823 bbox rows, 867 object-object relations,
  209 bbox classes and 19 relation labels.

## Important Visual Genome note

The original VG-400 manifest was not recoverable. The new deterministic
`vg400_image_ids.txt` is therefore the authoritative manifest for all NEW
Visual Genome reruns. Existing manuscript VG numerical results must be
replaced by results generated on this manifest.

## COCO relations

MS COCO instance annotations do not provide Visual-Genome-style relation
ground truth, so `relations_R` is intentionally blank for COCO.
