"""Decision logic for keeping stored vectors consistent with the configured embedding model."""

from scripts.reembed import VECTOR_COLUMNS, plan_change

MINILM, MPNET, BGE = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
)


def dims(d):
    return {col: d for col in VECTOR_COLUMNS}


def test_nothing_to_do_when_model_and_dimension_match():
    plan = plan_change(dims(768), MPNET, MPNET, 768)
    assert not plan.changes_vectors and not plan.record_only


def test_dimension_change_resizes_every_vector_column():
    plan = plan_change(dims(384), MINILM, MPNET, 768)
    assert plan.resize == VECTOR_COLUMNS and not plan.clear


def test_legacy_database_without_metadata_is_resized_on_dimension_change():
    plan = plan_change(dims(384), None, MPNET, 768)  # built before system_meta existed
    assert plan.resize == VECTOR_COLUMNS


def test_same_dimension_different_model_clears_vectors():
    plan = plan_change(dims(384), MINILM, BGE, 384)
    assert plan.clear and not plan.resize


def test_legacy_database_with_matching_dimension_only_records_the_model():
    plan = plan_change(dims(384), None, MINILM, 384)
    assert plan.record_only and not plan.changes_vectors
