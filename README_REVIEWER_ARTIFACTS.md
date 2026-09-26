# JOCO reviewer-artifact supplement

This package supplements the public JOCO repository with files requested during review.

## Included exact final artifacts
- Verified final OWL ontology
- RelTR V3 runner, notebook and final 300-image outputs
- Exact OI-300 image-ID manifest derived from the final output CSV
- Human-readable reported configuration

## Added implementation modules
- hierarchy-aware `rdfs:subClassOf` relation constraints
- connected-component decomposition + deterministic repair fallback
- pairwise CRF-style structured baseline
- violation and traceability metrics
- Visual Genome manifest/count builder
- bootstrap CI + paired permutation analysis
- fallback-log summarizer
- dense-scene solver stress test

## Scientific-status warning
Code availability does not replace missing experimental evidence. Read
`results/REMAINING_REQUIRED_RUNS.md`. Only remove manuscript red warnings after the
corresponding scripts have actually been run and their outputs committed.
