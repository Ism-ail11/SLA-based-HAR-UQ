import importlib.util
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pytest
from rateless_har.data import (
    Recording,
    read_wisdm,
    read_pamap2,
    read_realworld,
    window_recording,
    subject_split,
    normalization,
)
from rateless_har.hardware import integrate_energy, analyze_windows


def test_subject_splits_and_normalization():
    subjects = np.repeat(np.arange(10).astype(str), 10)
    split = subject_split(subjects, 7)
    assert split == subject_split(subjects, 7)
    groups = [set(v) for v in split.values()]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 10
    bad = dict(split)
    bad["test"] = split["train"]
    with pytest.raises(ValueError):
        subject_split(subjects, explicit=bad)
    mean, std = normalization(np.array([[[1, 3], [4, 4]]], np.float32))
    np.testing.assert_allclose(mean.ravel(), [2, 4])
    assert std[0, 1, 0] > 0


def test_wisdm_parser(tmp_path):
    f = tmp_path / "phone_accel.txt"
    f.write_text("1600,A,1000000000,1,2,3;\n1600,A,1050000000,4,5,6;\n")
    r = list(read_wisdm(f))[0]
    assert r.subject == "1600" and r.values.shape == (2, 3)
    np.testing.assert_allclose(r.time, [0, 0.05])


def test_pamap2_parser(tmp_path):
    f = tmp_path / "subject101.dat"
    values = np.tile(np.arange(54), (2, 1)).astype(float)
    values[:, 0] = [0, 0.01]
    np.savetxt(f, values)
    r = list(read_pamap2(f))[0]
    np.testing.assert_array_equal(r.values[0], [4, 5, 6, 10, 11, 12])
    assert r.subject == "101"


def test_realworld_alignment(tmp_path):
    frame = pd.DataFrame(
        dict(attr_time=[1000, 1020, 1040], attr_x=[1, 2, 3], attr_y=[4, 5, 6], attr_z=[7, 8, 9])
    )
    frame.to_csv(tmp_path / "acc.csv", index=False)
    frame.to_csv(tmp_path / "gyr.csv", index=False)
    r = list(read_realworld(tmp_path / "acc.csv", tmp_path / "gyr.csv", "1", "walking", "chest"))[0]
    assert r.values.shape == (3, 6)
    np.testing.assert_allclose(r.time, [0, 0.02, 0.04])


def test_windows_gaps_and_purity():
    t = np.r_[np.arange(200) / 50, 10 + np.arange(200) / 50]
    r = Recording("1", "session", "chest", t, np.ones((400, 6)), np.repeat("walking", 400), 50)
    rows = list(window_recording(r))
    assert len(rows) == 6 and all(row[0].shape == (6, 100) for row in rows)
    assert [row[-1] for row in rows] == [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    r.labels[:100] = np.tile(["walking", "sitting"], 50)
    assert len(list(window_recording(r))) == 4


def test_hardware_energy_and_deadline():
    trace = pd.DataFrame(dict(time_s=[0, 0.1, 0.2], voltage_v=[3, 3, 3], current_a=[0.001] * 3))
    assert integrate_energy(trace, 0, 0.2) == pytest.approx(600)
    windows = pd.DataFrame(
        dict(
            window_id=[0, 1, 2],
            ready_s=[0, 0, 0],
            prediction_s=[0.1, 0.2, np.nan],
            deadline_ms=[150] * 3,
            covered=[1, 1, 1],
            set_size=[1, 1, 3],
        )
    )
    analyzed = analyze_windows(windows)
    assert analyzed.sla_success.tolist() == [True, False, False]
    with pytest.raises(ValueError):
        integrate_energy(trace, 0, 0.3)


def test_archive_traversal_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "download_data", root / "scripts/download_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        module.extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
