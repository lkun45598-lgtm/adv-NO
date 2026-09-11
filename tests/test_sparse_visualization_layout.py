import importlib.util
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_PATH = PROJECT_ROOT / "3_flow_reconstruction" / "no" / "adv_training" / "plot_sparse_visualization.py"


def load_plot_module():
    sys.path.insert(0, str(PLOT_PATH.parent))
    spec = importlib.util.spec_from_file_location("plot_sparse_visualization", PLOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pre_layout_labels_and_slice_caption():
    module = load_plot_module()
    assert module.format_slice_caption(14, 30, "Sigma layer") == "Sigma layer 15 of 30"
    assert module.axis_labels("xi-grid index", "eta-grid index") == (
        "xi-grid index",
        "eta-grid index",
    )


def test_read_full_sample_preserves_complete_domain(tmp_path):
    module = load_plot_module()
    data_path = tmp_path / "sample.h5"
    field = np.arange(2 * 100 * 110 * 30, dtype=np.float32).reshape(1, 2, 100, 110, 30)
    valid = np.ones((1, 1, 100, 110, 30), dtype=np.uint8)
    with h5py.File(data_path, "w") as h5:
        split = h5.create_group("test")
        split.create_dataset("field", data=field)
        split.create_dataset("valid_mask", data=valid)

    loaded_field, loaded_valid = module.read_full_sample(data_path, "test", 0)

    assert loaded_field.shape == (2, 100, 110, 30)
    assert loaded_valid.shape == (1, 100, 110, 30)
    np.testing.assert_array_equal(loaded_field, field[0])


def test_full_domain_prediction_uses_tiled_predict_without_cropping(monkeypatch):
    module = load_plot_module()
    field = torch.zeros((1, 2, 100, 110, 30), dtype=torch.float32)
    valid = torch.ones((1, 1, 100, 110, 30), dtype=torch.float32)
    captured = {}

    def fake_tiled_predict(model, model_input, patch, depth, stride, tile_batch):
        captured["shape"] = tuple(model_input.shape)
        captured["tiles"] = (patch, depth, stride, tile_batch)
        return torch.full((1, 2, 100, 110, 30), 2.0)

    monkeypatch.setattr(module, "tiled_predict", fake_tiled_predict)
    prediction, observed = module.predict_full_domain(
        model=object(),
        field=field,
        valid=valid,
        mask_ratio=0.5,
        mean_t=torch.zeros((1, 2, 1, 1, 1)),
        generator=torch.Generator().manual_seed(25),
        patch=64,
        depth=16,
        stride=32,
        tile_batch=8,
    )

    assert captured == {"shape": (1, 4, 100, 110, 30), "tiles": (64, 16, 32, 8)}
    assert prediction.shape == (2, 100, 110, 30)
    assert observed.shape == (100, 110, 30)


def test_fixed_error_color_limit_is_shared_across_components():
    module = load_plot_module()
    limits = module.resolve_error_limits(np.asarray([0.08, 0.06]), 0.12)
    np.testing.assert_array_equal(limits, np.asarray([0.12, 0.12]))
    with pytest.raises(ValueError, match="error-vmax"):
        module.resolve_error_limits(np.asarray([0.08, 0.06]), 0.0)


def test_publication_column_titles_and_axis_visibility():
    module = load_plot_module()
    assert module.panel_titles(0.5) == (
        "Ground Truth",
        "Sparse Observations",
        "adv-NO Reconstruction",
        "Absolute Error",
    )
    for row in range(2):
        for col in range(4):
            assert module.axis_visibility(row, col) == (row == 1, col == 0)


def test_hide_redundant_axes_keeps_only_left_and_bottom_labels():
    module = load_plot_module()
    figure, axes = plt.subplots(2, 4)
    try:
        module.apply_axis_visibility(axes, "x index", "y index")
        for row in range(2):
            for col in range(4):
                assert bool(axes[row, col].get_xlabel()) is (row == 1)
                assert bool(axes[row, col].get_ylabel()) is (col == 0)
                assert any(label.get_visible() for label in axes[row, col].get_xticklabels()) is (row == 1)
                assert any(label.get_visible() for label in axes[row, col].get_yticklabels()) is (col == 0)
    finally:
        plt.close(figure)


def test_parser_exposes_publication_layout_controls():
    module = load_plot_module()
    args = module.build_arg_parser().parse_args(
        [
            "--data", "input.h5",
            "--checkpoint", "checkpoint.pt",
            "--config", "config.json",
            "--output", "output.png",
            "--x-label", "xi-grid index",
            "--y-label", "eta-grid index",
            "--slice-label", "Sigma layer",
            "--error-vmax", "0.12",
            "--patch", "64",
            "--depth", "16",
            "--stride", "32",
            "--tile-batch", "8",
        ]
    )
    assert args.x_label == "xi-grid index"
    assert args.y_label == "eta-grid index"
    assert args.slice_label == "Sigma layer"
    assert args.error_vmax == 0.12
    assert (args.patch, args.depth, args.stride, args.tile_batch) == (64, 16, 32, 8)
