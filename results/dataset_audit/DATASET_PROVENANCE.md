# JOCO Step-1 dataset provenance

## COCO
Source: official COCO 2017 train/val annotations archive.
Subset: first 500 entries of the `images` array in `instances_val2017.json`,
matching the released `ocosl_run_v4.py` selection rule.

## Visual Genome
Source: Visual Genome v1.4 object and relationship annotations plus image metadata.
The exact original VG-400 image-ID manifest was not preserved in the public artifacts.
Therefore Step 1 creates a new deterministic authoritative VG-400 subset:
seed=42, sampled from images with metadata, at least one object, and at least one relationship.
All Visual Genome numerical results in the revised manuscript must later be rerun on this manifest.

## Open Images
Uses the already established OI-300 manifest and annotations from the private Kaggle input.
