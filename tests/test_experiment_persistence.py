from pathlib import Path

import numpy as np

import core.experiment_manager as experiment_manager_module
from core.experiment_manager import Experiment, ExperimentManager
from core.roi import ROI, ROIShape


def test_experiment_to_json_and_from_json_roundtrip_preserves_critical_fields():
    experiment = Experiment(name="demo")
    roi_1 = ROI(x=1, y=2, width=5, height=6, shape=ROIShape.ELLIPSE).to_dict()
    roi_2 = ROI(
        x=3,
        y=4,
        width=6,
        height=4,
        shape=ROIShape.POLYGON,
        points=[(3, 4), (8, 4), (8, 8), (3, 8)],
    ).to_dict()
    experiment.rois = {"roi_1": roi_1, "roi_2": roi_2}
    experiment.analysis_results = {"runs": [{"summary": "ok"}]}

    neuron_locations = np.array([[1, 2], [3, 4]], dtype=np.int32)
    neuron_trajectories = np.arange(6, dtype=np.float32).reshape(2, 3)
    quality_mask = np.array([True, False], dtype=bool)
    roi_origin = np.array([0, 1], dtype=np.intp)
    experiment.set_neuron_detection_data(
        neuron_locations=neuron_locations,
        neuron_trajectories=neuron_trajectories,
        quality_mask=quality_mask,
        detection_params={"threshold": 0.5},
        roi_origin=roi_origin,
    )

    payload = experiment.to_json()

    nd = payload["experiment"]["neuron_detection"]
    assert nd["neuron_locations"] == neuron_locations.tolist()
    assert nd["neuron_trajectories"]["shape"] == [2, 3]
    assert nd["neuron_trajectories"]["dtype"] == str(neuron_trajectories.dtype)
    assert "data" in nd["neuron_trajectories"]

    restored = Experiment.from_json(payload)
    restored_data = restored.get_neuron_detection_data()

    assert restored.rois["roi_1"] == roi_1
    assert restored.rois["roi_2"] == roi_2
    assert np.array_equal(restored_data["neuron_locations"], neuron_locations)
    assert np.array_equal(restored_data["neuron_trajectories"], neuron_trajectories)
    assert np.array_equal(restored_data["quality_mask"], quality_mask)
    assert np.array_equal(restored_data["roi_origin"], roi_origin)


def test_experiment_manager_save_load_and_validate(tmp_path, monkeypatch):
    recent_file = tmp_path / "config" / "recent_experiments.json"
    recent_file.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(experiment_manager_module, "RECENT_FILE", recent_file)

    manager = ExperimentManager()
    experiment = manager.create_new_experiment({"name": "Persistence Test"})
    file_path = tmp_path / "exp.json"

    assert manager.save_experiment(experiment, str(file_path)) is True
    assert file_path.exists()
    assert manager.validate_experiment_file(str(file_path)) is True

    loaded = manager.load_experiment(str(file_path))

    assert loaded.name == "Persistence Test"
    recents = manager.get_recent_experiments()
    assert recents
    assert Path(recents[0]["path"]).resolve() == file_path.resolve()
