from perturbatlas.reproducibility import run_metadata


def test_run_metadata_contains_core_fields():
    metadata = run_metadata({"seed": 1})
    assert "timestamp_utc" in metadata
    assert "git_commit" in metadata
    assert "python" in metadata
    assert metadata["config"]["seed"] == 1
