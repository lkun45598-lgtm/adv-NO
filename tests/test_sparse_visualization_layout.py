import importlib.util
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.collections import QuadMesh


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


def test_era5_spatial_grid_uses_global_degree_edges(tmp_path):
    module = load_plot_module()
    data_path = tmp_path / "era5.h5"
    metadata = {
        "task": "D",
        "spatial_scale": 4,
        "source_spatial_crop": [720, 1440],
        "processed_shape": [219, 2, 180, 360, 8],
    }
    with h5py.File(data_path, "w") as h5:
        h5.attrs["metadata_json"] = __import__("json").dumps(metadata)

    longitude, latitude, shading = module.load_spatial_grid(
        data_path, expected_shape=(180, 360)
    )

    assert longitude.shape == (361,)
    assert latitude.shape == (181,)
    assert shading == "flat"
    np.testing.assert_allclose(longitude[[0, -1]], [0.0, 360.0])
    np.testing.assert_allclose(latitude[[0, -1]], [90.0, -90.0])
    assert np.all(np.diff(longitude) > 0)
    assert np.all(np.diff(latitude) < 0)


def test_pre_spatial_grid_downsamples_curvilinear_coordinates(tmp_path):
    module = load_plot_module()
    coordinate_dir = tmp_path / "stat_var"
    coordinate_dir.mkdir()
    source_shape = (8, 12)
    longitude = 112.0 + np.arange(12)[None, :] + 0.1 * np.arange(8)[:, None]
    latitude = 20.0 + np.arange(8)[:, None] + 0.01 * np.arange(12)[None, :]
    np.save(coordinate_dir / "lon_rho.npy", longitude)
    np.save(coordinate_dir / "lat_rho.npy", latitude)
    np.save(coordinate_dir / "pm.npy", np.ones(source_shape))
    np.save(coordinate_dir / "pn.npy", np.ones(source_shape))
    np.save(coordinate_dir / "mask_rho.npy", np.ones(source_shape))

    data_path = tmp_path / "pre.h5"
    metadata = {
        "task": "A",
        "source": str(tmp_path / "unused_source"),
        "spatial_scale": 4,
        "source_spatial_crop": list(source_shape),
        "processed_shape": [1, 2, 2, 3, 1],
    }
    with h5py.File(data_path, "w") as h5:
        h5.attrs["metadata_json"] = __import__("json").dumps(metadata)

    coarse_lon, coarse_lat, shading = module.load_spatial_grid(
        data_path,
        expected_shape=(2, 3),
        pre_coordinate_dir=coordinate_dir,
    )

    assert coarse_lon.shape == (2, 3)
    assert coarse_lat.shape == (2, 3)
    assert shading == "nearest"
    np.testing.assert_allclose(coarse_lon[0, 0], longitude[:4, :4].mean())
    np.testing.assert_allclose(coarse_lat[-1, -1], latitude[4:8, 8:12].mean())


def test_geographic_field_is_drawn_as_coordinate_aware_quadmesh():
    module = load_plot_module()
    figure, axis = plt.subplots()
    try:
        image = module.plot_spatial_field(
            axis,
            longitude=np.linspace(0.0, 360.0, 5),
            latitude=np.linspace(90.0, -90.0, 3),
            values=np.arange(8, dtype=float).reshape(2, 4),
            shading="flat",
            cmap="viridis",
            vmin=0.0,
            vmax=7.0,
        )
        assert isinstance(image, QuadMesh)
        np.testing.assert_allclose(axis.get_xlim(), (0.0, 360.0))
        np.testing.assert_allclose(axis.get_ylim(), (-90.0, 90.0))
    finally:
        plt.close(figure)


def test_column_titles_are_above_and_do_not_intersect_image_axes():
    module = load_plot_module()
    figure, axes = plt.subplots(2, 4, figsize=(18, 7))
    try:
        title_artists = module.add_column_titles(
            figure,
            axes,
            ("Ground Truth", "Sparse Observations", "adv-NO Reconstruction", "Absolute Error"),
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for title_artist, axis in zip(title_artists, axes[0]):
            title_box = title_artist.get_window_extent(renderer=renderer)
            assert title_box.y0 > axis.get_window_extent(renderer=renderer).y1
    finally:
        plt.close(figure)


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


def test_predict_sample_reports_selected_generator_source(monkeypatch, tmp_path):
    module = load_plot_module()

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state):
            self.state = state

        def eval(self):
            return self

    checkpoint = {
        "par": {
            "nx": 64,
            "nz": 16,
            "inp_shift": torch.zeros((1, 4, 1, 1, 1)),
        }
    }
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(module, "Unet3D", lambda **kwargs: FakeModel())
    monkeypatch.setattr(
        module,
        "select_generator_state",
        lambda checkpoint, prefer_ema: ({"weight": torch.tensor(1.0)}, "generator_ema"),
    )
    monkeypatch.setattr(
        module,
        "read_full_sample",
        lambda *args: (
            np.zeros((2, 100, 110, 30), dtype=np.float32),
            np.ones((1, 100, 110, 30), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        module,
        "predict_full_domain",
        lambda *args, **kwargs: (
            np.zeros((2, 100, 110, 30), dtype=np.float32),
            np.ones((100, 110, 30), dtype=bool),
        ),
    )

    *_, source = module.predict_sample(
        data_path=tmp_path / "input.h5",
        checkpoint_path=tmp_path / "checkpoint.pt",
        config_path=None,
        sample=0,
        slice_index=14,
        mask_ratio=0.5,
        seed=25,
        device_name="cpu",
        patch=64,
        depth=16,
    )

    assert source == "generator_ema"


def test_fixed_error_color_limit_is_shared_across_components():
    module = load_plot_module()
    limits = module.resolve_error_limits(np.asarray([0.08, 0.06]), 0.12)
    np.testing.assert_array_equal(limits, np.asarray([0.12, 0.12]))
    with pytest.raises(ValueError, match="error-vmax"):
        module.resolve_error_limits(np.asarray([0.08, 0.06]), 0.0)


def test_error_scale_caption_matches_fixed_or_dynamic_policy():
    module = load_plot_module()
    assert "fixed cross-rate scale" in module.error_scale_caption(0.12)
    assert "per-figure scale" in module.error_scale_caption(None)


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


def test_long_velocity_row_labels_wrap_without_abbreviation():
    module = load_plot_module()
    assert module.format_row_label("Zonal Wind Velocity (u)") == (
        "Zonal Wind\nVelocity (u)"
    )
    assert module.format_row_label("Eastward Velocity (u)") == (
        "Eastward Velocity (u)"
    )


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


def test_parser_uses_publication_era5_row_labels():
    module = load_plot_module()
    args = module.build_arg_parser().parse_args(
        [
            "--data", "input.h5",
            "--checkpoint", "checkpoint.pt",
            "--config", "config.json",
            "--output", "output.png",
        ]
    )
    assert args.u_label == "Zonal Wind Velocity (u)"
    assert args.v_label == "Meridional Wind Velocity (v)"


def test_parser_defaults_to_physical_coordinate_labels():
    module = load_plot_module()
    args = module.build_arg_parser().parse_args(
        [
            "--data", "input.h5",
            "--checkpoint", "checkpoint.pt",
            "--config", "config.json",
            "--output", "output.png",
        ]
    )
    assert args.x_label is None
    assert args.y_label is None


def test_geographic_tick_formatters_show_cardinal_directions():
    module = load_plot_module()
    assert module.format_longitude_tick(0.0, None) == "0°"
    assert module.format_longitude_tick(112.5, None) == "112.5°E"
    assert module.format_longitude_tick(-30.0, None) == "30°W"
    assert module.format_latitude_tick(-90.0, None) == "90°S"
    assert module.format_latitude_tick(0.0, None) == "0°"
    assert module.format_latitude_tick(23.125, None) == "23.1°N"


def test_geographic_extent_caption_has_directional_ranges():
    module = load_plot_module()
    assert module.geographic_extent_caption(
        np.asarray([0.0, 360.0]), np.asarray([-90.0, 90.0])
    ) == "Longitude 0°E–360°E | Latitude 90°S–90°N"
    assert module.geographic_extent_caption(
        np.asarray([112.32, 115.67]), np.asarray([20.90, 23.12])
    ) == "Longitude 112.32–115.67°E | Latitude 20.90–23.12°N"
