# JOCO RelTR + OCOSL V3 — Open Images 300

This package archives the final controlled experiment used to answer the reviewer request for a stronger learned relation generator.

## Final result

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Raw OI-RelTR | 0.336729 | 0.686275 | 0.451784 |
| RelTR-only, shared candidate pool | 0.461538 | 0.456747 | 0.459130 |
| RelTR + hierarchy-aware OCOSL V3 | 0.461449 | 0.455594 | 0.458503 |

- Micro-F1 difference: **-0.0006278**
- Paired permutation p-value: **1.0**
- Per-image improved / equal / worse: **1 / 298 / 1**
- Fallback count: **0**
- Average OCOSL solver time: **0.00640 s/image**

## Why V3

The original solver tested relation domain/range types using flat `label_to_type(...) == type` equality. The final V3 implementation instead follows the ontology hierarchy using the transitive closure of `rdfs:subClassOf`. This fixes valid cases such as `Boat ⊑ Vehicle` and `Glasses ⊑ Clothing` without changing the ontology, thresholds, objective weights, relation budget, or evaluation IoU.

## Files

- `run_oi300_reltr_ocosl_v3.py`: GitHub-ready reproducibility runner.
- `JOCO_RelTR_OCOSL_V3_300_results.ipynb`: compact results/reproducibility notebook.
- `results/`: final 300-image outputs and diagnostics.

## Kaggle rerun

The runner can auto-locate the OI-300 dataset and JOCO code when they are attached as Kaggle inputs.

```bash
python run_oi300_reltr_ocosl_v3.py \
  --prepare-reltr \
  --output-dir /kaggle/working/JOCO/results_v3 \
  --n-images 300 \
  --verify-reference
```

Required JOCO implementation files:
- `ocosl_run_v4.py`
- `ocosl_utils_v3.py`
- `JOCO_verified_ontology_schema_fixed.owl`

Required Open Images subset:
- `oi300_ids.txt`
- `images/*.jpg`
- `openimages_relationships.csv`

`--prepare-reltr` clones the official RelTR repository and downloads the official OI checkpoint/metadata used by the experiment.
