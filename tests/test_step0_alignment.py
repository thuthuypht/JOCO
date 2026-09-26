from src.ontology_scoring_final import FINAL_ONTOLOGY_SCORE_WEIGHTS

def test_clean_score_weights_sum_one():
    assert abs(sum(FINAL_ONTOLOGY_SCORE_WEIGHTS.values()) - 1.0) < 1e-12
