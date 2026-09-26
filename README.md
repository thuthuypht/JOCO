# JOCO — OCOSL reproducibility repository

This repository supports the revised manuscript **“Ontology-Constrained Combinatorial Optimization for Semantic Image Labeling via Scene Graph Reasoning.”**

## Active artifacts

- `JOCO_verified_ontology_schema_fixed.owl` — verified ontology used by the revised experiments.
- `ocosl_run_final.py` — final solver wrapper: hierarchy-aware relation constraints, decomposition/repair fallback, diagnostics and traceability.
- `src/ocosl_constraints_final.py` — hierarchy-aware ILP constraints and fallback implementation.
- `src/crf_baseline.py` — pairwise CRF-style structured-inference baseline.
- `src/diagnostics_traceability.py` — violation, compliance, traceability and explanation-evidence metrics.
- `configs/final_reported_config.yaml` — reported base and RelTR-V3 settings.
- `experiments/` — Visual Genome manifest builder, statistical analysis, fallback summarizer and dense-scene stress test.
- `manifests/oi300_ids.txt` — exact 300 Open Images IDs represented in the final RelTR-V3 output.
- `results/` — final RelTR-V3 outputs plus a checklist of experimental outputs.

## RelTR V3 stronger-generator experiment

The controlled 300-image experiment compares the same OI-trained RelTR candidate pool before and after hierarchy-aware OCOSL inference.

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Raw OI-RelTR | 0.336729 | 0.686275 | 0.451784 |
| RelTR-only, shared candidate pool | 0.461538 | 0.456747 | 0.459130 |
| RelTR + hierarchy-aware OCOSL V3 | 0.461449 | 0.455594 | 0.458503 |

Micro-F1 difference = **-0.0006278**; paired permutation **p=1.0**; per-image improved/equal/worse = **1/298/1**; fallback count = **0**.

Run:
```bash
python run_oi300_reltr_ocosl_v3.py   --prepare-reltr   --output-dir /kaggle/working/JOCO/results_v3   --n-images 300   --verify-reference
```

## Base reported configuration

See `configs/final_reported_config.yaml`. The revised manuscript reports, for the geometry-driven base experiments, top-k=3, detector threshold 0.25, relation threshold 0.70, duplicate IoU 0.60, relation budget C=2n, alpha=0.40, beta=0.30, gamma=0.20, lambda=0.10, and CBC time limit 10 s. The RelTR robustness experiment has its own explicitly separated settings in the same file.

## Important reproducibility status

The repository distinguishes **released code** from **completed experimental evidence**. Files/scripts have been added for the Visual Genome manifest, pairwise CRF baseline, fallback analysis, dense-scene stress test, violation metrics, traceability logs, and paired statistical tests. Their manuscript values must come from actual runs; no missing result is fabricated.

Before removing every red warning in the manuscript, read:

`results/REMAINING_REQUIRED_RUNS.md`

## Legacy files

`ocosl_run_v3.py`, `ocosl_utils_v3.py`, `v5_ocosl_ontology_schema.owl`, and `v6_ocosl_ontology_schema_openimages.owl` are retained for provenance. The revised active entry point is `ocosl_run_final.py`, while the RelTR-V3 experiment uses `run_oi300_reltr_ocosl_v3.py`.

## Environment

```bash
pip install -r requirements_final.txt
```

## Data

The manuscript uses MS COCO, Visual Genome, and an Open Images subset. Public dataset files are not redistributed here. Exact processed subset identifiers should be stored under `manifests/`; the final OI-300 manifest is included. The Visual Genome builder can either consume the original manifest or generate a deterministic replacement that must then be used consistently in rerun results.
