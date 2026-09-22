import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rateless_har.cli import grid_jobs, load_config
from rateless_har.demo import make_features, make_windows
from rateless_har.experiment import evaluate, METHODS


@pytest.fixture
def config():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/default.json")
    cfg["dataset"] = "SYNTHETIC_TEST"
    return cfg


@pytest.mark.parametrize("method", METHODS)
def test_all_methods(tmp_path, config, method):
    make_features(tmp_path / "run", per_role=30)
    summary = evaluate(tmp_path / "run", tmp_path / "out", config, method)
    rows = pd.read_csv(tmp_path / "out/windows.csv")
    assert len(rows) == 30 and (rows.k <= 6).all()
    assert (rows.latency_ms <= rows.deadline_ms).all()
    assert (rows.emitted >= rows.k).all()
    assert (rows.energy_uj >= 0).all()
    assert summary["source"] == "simulation"
    if method == "ours":
        assert summary["k_star"] is None  # Too few pilot samples for a singleton target.
        assert not rows.budget_feasible.any()


def test_grid_is_complete(config):
    jobs = grid_jobs(config, "all")
    assert len(jobs) == 252
    assert len(set(j[0] for j in jobs)) == 252
    assert len(grid_jobs(config, "main")) == 224


def test_repeatability_and_no_test_label_leakage(tmp_path, config):
    run = tmp_path / "run"
    make_features(run, per_role=40)
    evaluate(run, tmp_path / "a", config)
    evaluate(run, tmp_path / "b", config)
    assert (tmp_path / "a/windows.csv").read_bytes() == (tmp_path / "b/windows.csv").read_bytes()
    with np.load(run / "features.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    data["y_test"] = (data["y_test"] + 1) % 3
    np.savez_compressed(run / "features.npz", **data)
    evaluate(run, tmp_path / "c", config)
    a, c = pd.read_csv(tmp_path / "a/windows.csv"), pd.read_csv(tmp_path / "c/windows.csv")
    columns = [
        "prediction_set",
        "target_k",
        "emitted",
        "k",
        "latency_ms",
        "energy_uj",
        "debt",
        "resolved",
        "winner",
    ]
    pd.testing.assert_frame_equal(a[columns], c[columns])


def test_short_qat_training_pipeline(tmp_path, config):
    pytest.importorskip("torch")
    from rateless_har.train import train

    data = tmp_path / "windows.npz"
    make_windows(data, windows_per_subject=12)
    cfg = copy.deepcopy(config["training"])
    cfg.update(epochs=2, batch_size=32)
    summary = train(data, tmp_path / "run", cfg)
    assert summary["role_counts"]["test"] == 24
    assert summary["profile"]["parameters"] > 0 and summary["profile"]["macs"] > 0
    with (
        np.load(data, allow_pickle=False) as raw,
        np.load(tmp_path / "run/normalization.npz", allow_pickle=False) as norm,
    ):
        take = np.isin(raw["subject"], summary["split"]["train"])
        np.testing.assert_allclose(
            norm["mean"],
            raw["x"][take].astype(float).mean((0, 2), keepdims=True),
            rtol=1e-5,
            atol=1e-7,
        )
    evaluate(tmp_path / "run", tmp_path / "evaluation", config)
    assert json.loads((tmp_path / "evaluation/summary.json").read_text())["features_sha256"]
