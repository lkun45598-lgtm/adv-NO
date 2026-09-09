import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


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
    assert module.format_slice_caption(4, 16, "Sigma layer") == "Sigma layer 5 of 16"
    assert module.axis_labels("xi-grid index", "eta-grid index") == (
        "xi-grid index",
        "eta-grid index",
    )


def test_fixed_error_color_limit_is_shared_across_components():
    module = load_plot_module()
    limits = module.resolve_error_limits(np.asarray([0.08, 0.06]), 0.12)
    np.testing.assert_array_equal(limits, np.asarray([0.12, 0.12]))
    with pytest.raises(ValueError, match="error-vmax"):
        module.resolve_error_limits(np.asarray([0.08, 0.06]), 0.0)


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
        ]
    )
    assert args.x_label == "xi-grid index"
    assert args.y_label == "eta-grid index"
    assert args.slice_label == "Sigma layer"
    assert args.error_vmax == 0.12
