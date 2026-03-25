"""Tests for experiment serialization and ExperimentManager (``core.experiment_manager``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.experiment_manager import Experiment, ExperimentManager


@pytest.fixture
def recent_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "recent_experiments.json"
    monkeypatch.setattr("core.experiment_manager.RECENT_FILE", path)
    return path


def test_experiment_to_json_roundtrip() -> None:
    exp = Experiment(
        name="Roundtrip",
        description="d",
        principal_investigator="pi",
        image_stack_path="/data",
        image_count=2,
        image_stack_files=["/data/a.tif", "/data/b.tif"],
    )
    exp.rois = {"roi_1": {"x": 1, "y": 2, "width": 3, "height": 4}, "roi_2": None}
    payload = exp.to_json()
    assert payload["version"] == "1.0"
    restored = Experiment.from_json(payload)
    assert restored.name == exp.name
    assert restored.image_stack_files == exp.image_stack_files
    assert restored.rois["roi_1"]["width"] == 3
    assert restored.rois["roi_2"] is None


def test_from_json_legacy_single_roi() -> None:
    data = {
        "version": "1.0",
        "experiment": {
            "name": "Legacy",
            "roi": {"x": 0, "y": 0, "width": 10, "height": 10},
            "image_stack": {"path": "", "file_list": [], "count": 0},
        },
    }
    exp = Experiment.from_json(data)
    assert exp.rois["roi_1"] == data["experiment"]["roi"]
    assert exp.rois["roi_2"] is None


def test_neuron_detection_serialize_deserialize_roundtrip() -> None:
    exp = Experiment(name="N")
    loc = np.array([[1, 2], [3, 4]], dtype=np.int32)
    traj = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([True, False], dtype=bool)
    exp.set_neuron_detection_data(
        neuron_locations=loc,
        neuron_trajectories=traj,
        quality_mask=mask,
        detection_params={"k": 1},
        roi_origin=np.array([0, 1], dtype=np.intp),
    )
    payload = exp.to_json()
    blob = json.loads(json.dumps(payload))
    exp2 = Experiment.from_json(blob)
    nd = exp2.get_neuron_detection_data()
    assert nd is not None
    np.testing.assert_array_equal(nd["neuron_locations"], loc)
    np.testing.assert_array_equal(nd["neuron_trajectories"], traj)
    np.testing.assert_array_equal(nd["quality_mask"], mask)
    assert nd["detection_params"] == {"k": 1}
    assert list(nd["roi_origin"]) == [0, 1]


def test_create_new_experiment_requires_name(recent_file: Path) -> None:
    mgr = ExperimentManager()
    with pytest.raises(ValueError, match="name cannot be empty"):
        mgr.create_new_experiment({"name": "   "})


def test_create_new_experiment_sets_analysis_type(recent_file: Path) -> None:
    mgr = ExperimentManager()
    exp = mgr.create_new_experiment({"name": "X", "analysis_type": "SCN"})
    assert exp.settings["processing"]["analysis_type"] == "SCN"


def test_validate_experiment_file(tmp_path: Path, recent_file: Path) -> None:
    mgr = ExperimentManager()
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert mgr.validate_experiment_file(str(bad)) is False

    good = tmp_path / "good.nexp"
    good.write_text(
        json.dumps({"version": "1.0", "experiment": {"name": "Ok", "image_stack": {}}}),
        encoding="utf-8",
    )
    assert mgr.validate_experiment_file(str(good)) is True


def test_save_and_load_experiment(tmp_path: Path, recent_file: Path) -> None:
    mgr = ExperimentManager()
    exp = Experiment(name="FileExp", description="")
    path = tmp_path / "e.nexp"
    assert mgr.save_experiment(exp, str(path)) is True
    loaded = mgr.load_experiment(str(path))
    assert loaded.name == "FileExp"


def test_recent_list_filters_missing_files(tmp_path: Path, recent_file: Path) -> None:
    mgr = ExperimentManager()
    missing = str(tmp_path / "gone.nexp")
    recent_file.write_text(
        json.dumps(
            {
                "recent": [
                    {"path": missing, "name": "gone", "last_opened": "2099-01-01T00:00:00"},
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert mgr.get_recent_experiments() == []
