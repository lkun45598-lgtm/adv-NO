"""Render a paper-ready PRE or ERA5 sparse-reconstruction visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tcunet import Unet3D  # noqa: E402
from evaluate_sparse_metrics import tiled_predict  # noqa: E402
from train_sparse_adv_no import make_mask, select_generator_state  # noqa: E402


FIELD_CMAP = "RdBu_r"
ERROR_CMAP = "magma"


def panel_titles(mask_ratio: float) -> tuple[str, str, str, str]:
    """Return panel labels whose observation rate matches the rendered mask."""
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask ratio must be in [0, 1], got {mask_ratio}")
    return (
        "Ground Truth",
        "Sparse Observations",
        "adv-NO Reconstruction",
        "Absolute Error",
    )


def visualization_labels(title: str, u_label: str, v_label: str) -> tuple[str, str, str]:
    """Return caller-supplied labels for the task-specific figure header."""
    return title, u_label, v_label


def format_slice_caption(slice_index: int, slice_count: int, slice_label: str) -> str:
    """Format the fifth-axis label without assuming it is time."""
    if not 0 <= slice_index < slice_count:
        raise ValueError(f"slice index must be in [0, {slice_count}), got {slice_index}")
    return f"{slice_label} {slice_index + 1} of {slice_count}"


def axis_labels(x_label: str, y_label: str) -> tuple[str, str]:
    """Return explicit horizontal and vertical grid-axis labels."""
    return x_label, y_label


def axis_visibility(row: int, col: int, row_count: int = 2) -> tuple[bool, bool]:
    """Return whether an axis should display x and y coordinates."""
    return row == row_count - 1, col == 0


def apply_axis_visibility(axes: np.ndarray, x_label: str, y_label: str) -> None:
    """Show coordinates only along the bottom and left edges."""
    row_count, col_count = axes.shape
    for row in range(row_count):
        for col in range(col_count):
            show_x, show_y = axis_visibility(row, col, row_count)
            axis = axes[row, col]
            axis.set_xlabel(x_label if show_x else "")
            axis.set_ylabel(y_label if show_y else "")
            axis.tick_params(
                axis="x",
                bottom=show_x,
                labelbottom=show_x,
                length=3 if show_x else 0,
            )
            axis.tick_params(
                axis="y",
                left=show_y,
                labelleft=show_y,
                length=3 if show_y else 0,
            )


def resolve_error_limits(error_limits: np.ndarray, error_vmax: float | None) -> np.ndarray:
    """Return a shared non-negative error scale for all rows and missing rates."""
    observed_max = float(np.nanmax(np.asarray(error_limits, dtype=np.float64)))
    if not np.isfinite(observed_max) or observed_max <= 0.0:
        observed_max = np.finfo(np.float64).eps
    if error_vmax is None:
        value = observed_max
    else:
        if error_vmax <= 0.0:
            raise ValueError("error-vmax must be positive")
        value = float(error_vmax)
    return np.full(np.asarray(error_limits).shape, value, dtype=np.float64)


def ground_truth_field_limits(truth: np.ndarray) -> np.ndarray:
    """Derive stable symmetric component scales from the selected truth field."""
    limits = np.nanmax(np.abs(np.asarray(truth, dtype=np.float64)), axis=(1, 2))
    return np.maximum(limits, np.finfo(np.float64).eps)


def mask_invalid_land(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Set non-ocean cells to NaN without modifying the source array."""
    result = np.asarray(values, dtype=np.float32).copy()
    valid_mask = np.broadcast_to(np.asarray(valid, dtype=bool), result.shape)
    result[~valid_mask] = np.nan
    return result


def mask_sparse_observations(values: np.ndarray, observed: np.ndarray,
                             valid: np.ndarray) -> np.ndarray:
    """Render only observed values at valid ocean cells."""
    ocean_observed = np.asarray(observed, dtype=bool) & np.asarray(valid, dtype=bool)
    return np.where(ocean_observed[None], values, np.nan)


def read_full_sample(data_path: Path, split: str, sample: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one complete processed field without spatial or depth cropping."""
    with h5py.File(data_path, "r") as h5:
        if split not in h5:
            raise KeyError(f"split {split!r} is not present in {data_path}")
        fields = h5[split]["field"]
        valid_masks = h5[split]["valid_mask"]
        if not 0 <= sample < fields.shape[0]:
            raise IndexError(f"sample must be in [0, {fields.shape[0]})")
        field = np.asarray(fields[sample], dtype=np.float32)
        valid = np.asarray(valid_masks[sample], dtype=np.float32)
    return field, valid


@torch.no_grad()
def predict_full_domain(
    model,
    field: torch.Tensor,
    valid: torch.Tensor,
    mask_ratio: float,
    mean_t: torch.Tensor,
    generator: torch.Generator,
    patch: int,
    depth: int,
    stride: int,
    tile_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask and reconstruct a complete sample with overlapping model tiles."""
    model_input, observed_t = make_mask(
        field,
        valid,
        mask_ratio,
        mean_t=mean_t,
        generator=generator,
    )
    prediction_t = tiled_predict(
        model,
        model_input,
        patch=patch,
        depth=depth,
        stride=stride,
        tile_batch=tile_batch,
    )
    prediction = prediction_t[0].detach().cpu().numpy()
    observed = observed_t[0, 0].detach().cpu().numpy().astype(bool)
    return prediction, observed


@torch.no_grad()
def predict_sample(
    data_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    sample: int,
    slice_index: int,
    mask_ratio: float,
    seed: int,
    device_name: str,
    raw_generator: bool = False,
    split: str = "test",
    patch: int | None = None,
    depth: int | None = None,
    stride: int = 32,
    tile_batch: int = 8,
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = json.loads(config_path.read_text()) if config_path else {}
    par = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in checkpoint["par"].items()
    }
    model = Unet3D(
        dim=int(config.get("dim", 16)),
        Par=par,
        dim_mults=(1, 2, 4, 8),
        channels=4,
    ).to(device)
    generator_state, _ = select_generator_state(checkpoint, prefer_ema=not raw_generator)
    model.load_state_dict(generator_state)
    model.eval()

    field, valid = read_full_sample(data_path, split, sample)
    _, height, width, full_depth = field.shape
    checkpoint_patch = int(par["nx"])
    checkpoint_depth = int(par["nz"])
    tile_patch = checkpoint_patch if patch is None else patch
    tile_depth = checkpoint_depth if depth is None else depth
    if (tile_patch, tile_depth) != (checkpoint_patch, checkpoint_depth):
        raise ValueError(
            "tile patch/depth must match checkpoint dimensions "
            f"{(checkpoint_patch, checkpoint_depth)}, got {(tile_patch, tile_depth)}"
        )
    if tile_patch > min(height, width) or tile_depth > full_depth:
        raise ValueError(
            f"checkpoint patch {(tile_patch, tile_depth)} exceeds full sample shape {field.shape}"
        )
    if not 0 <= slice_index < full_depth:
        raise IndexError(f"slice index must be in [0, {full_depth})")

    field_t = torch.from_numpy(field[None]).to(device)
    valid_t = torch.from_numpy(valid[None]).to(device)
    generator = torch.Generator(device=device.type).manual_seed(seed + sample)
    mean_t = par["inp_shift"][:, :2]
    prediction, observed = predict_full_domain(
        model,
        field_t,
        valid_t,
        mask_ratio,
        mean_t=mean_t,
        generator=generator,
        patch=tile_patch,
        depth=tile_depth,
        stride=stride,
        tile_batch=tile_batch,
    )
    return field, prediction, observed, valid, generator_source


def render(args):
    target, prediction, observed, valid, generator_source = predict_sample(
        args.data,
        args.checkpoint,
        args.config,
        args.sample,
        args.slice_index,
        args.mask_ratio,
        args.seed,
        args.device,
        args.raw_generator,
        args.split,
        args.patch,
        args.depth,
        args.stride,
        args.tile_batch,
    )
    slice_index = args.slice_index
    valid_slice = valid[0, :, :, slice_index]
    truth = mask_invalid_land(target[:, :, :, slice_index], valid_slice)
    sparse = mask_sparse_observations(truth, observed[:, :, slice_index], valid_slice)
    reconstruction = mask_invalid_land(prediction[:, :, :, slice_index], valid_slice)
    absolute_error = np.abs(reconstruction - truth)
    titles = panel_titles(args.mask_ratio)

    field_limits = ground_truth_field_limits(truth)
    error_limits = np.nanmax(absolute_error, axis=(1, 2))
    error_scales = resolve_error_limits(error_limits, args.error_vmax)
    x_label, y_label = axis_labels(args.x_label, args.y_label)
    title, u_label, v_label = visualization_labels(args.title, args.u_label, args.v_label)
    _, height, width, full_depth = target.shape
    domain_aspect = width / height
    figure_width = args.figure_width or 18.0
    figure_height = args.figure_height or (6.8 if domain_aspect >= 1.5 else 8.8)
    top = 0.79 if domain_aspect >= 1.5 else 0.82
    bottom = 0.15 if domain_aspect >= 1.5 else 0.11

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    figure = plt.figure(figsize=(figure_width, figure_height), constrained_layout=False)
    grid = GridSpec(
        2,
        6,
        figure=figure,
        width_ratios=(1.0, 1.0, 1.0, 0.035, 1.0, 0.035),
        left=0.075,
        right=0.975,
        bottom=bottom,
        top=top,
        wspace=0.16,
        hspace=0.18,
    )
    axes = np.empty((2, 4), dtype=object)
    field_color_axes = []
    error_color_axes = []
    for row in range(2):
        axes[row, 0] = figure.add_subplot(grid[row, 0])
        axes[row, 1] = figure.add_subplot(grid[row, 1])
        axes[row, 2] = figure.add_subplot(grid[row, 2])
        field_color_axes.append(figure.add_subplot(grid[row, 3]))
        axes[row, 3] = figure.add_subplot(grid[row, 4])
        error_color_axes.append(figure.add_subplot(grid[row, 5]))

    for row, component in enumerate(("u", "v")):
        field_images = (truth[row], sparse[row], reconstruction[row])
        field_image = None
        for col, values in enumerate(field_images):
            axis = axes[row, col]
            field_image = axis.imshow(
                values,
                cmap=FIELD_CMAP,
                vmin=-field_limits[row],
                vmax=field_limits[row],
                interpolation="nearest",
                origin="lower",
                aspect="equal",
            )
            if row == 0:
                axis.set_title(titles[col], pad=10, fontsize=14, fontweight="bold")

        field_colorbar = figure.colorbar(field_image, cax=field_color_axes[row])
        field_colorbar.ax.tick_params(labelsize=10, length=3)
        field_colorbar.set_label(
            f"{component} (m s$^{{-1}}$)", fontsize=10, rotation=270, labelpad=15
        )

        axis = axes[row, 3]
        error_image = axis.imshow(
            absolute_error[row],
            cmap=ERROR_CMAP,
            vmin=0,
            vmax=float(error_scales[row]),
            interpolation="nearest",
            origin="lower",
            aspect="equal",
        )
        if row == 0:
            axis.set_title(titles[3], pad=10, fontsize=14, fontweight="bold")
        error_colorbar = figure.colorbar(error_image, cax=error_color_axes[row])
        error_colorbar.ax.tick_params(labelsize=10, length=3)
        error_colorbar.set_label(
            f"|{component} error| (m s$^{{-1}}$)",
            fontsize=10,
            rotation=270,
            labelpad=17,
        )

    apply_axis_visibility(axes, x_label, y_label)
    figure.canvas.draw()
    for row, row_label in enumerate((u_label, v_label)):
        bounds = axes[row, 0].get_position()
        figure.text(
            0.022,
            (bounds.y0 + bounds.y1) / 2,
            row_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#1F2937",
        )

    figure.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.915,
        f"Held-out test sample {args.sample} | full {height} x {width} domain "
        f"| {format_slice_caption(slice_index, full_depth, args.slice_label)} "
        f"| Missing rate: {args.mask_ratio:.0%} "
        f"| Observed fraction: {1.0 - args.mask_ratio:.0%} "
        f"| Weights: {generator_source}",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )
    figure.text(
        0.5,
        0.035,
        "Ground truth, observations, and reconstruction share each row's velocity scale; "
        "absolute-error panels use a fixed cross-rate scale.",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the physical canvas fixed across missing rates for stable paper/PPT layout.
    figure.savefig(args.output, dpi=300, facecolor="white")
    plt.close(figure)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--slice-index", "--time-index", dest="slice_index", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-generator", action="store_true",
                        help="Use raw generator weights instead of EMA weights when present")
    parser.add_argument("--x-label", default="Longitude index")
    parser.add_argument("--y-label", default="Latitude index")
    parser.add_argument("--slice-label", default="Time step")
    parser.add_argument("--error-vmax", type=float,
                        help="Fixed absolute-error colorbar upper limit in physical units")
    parser.add_argument("--patch", type=int,
                        help="Horizontal tile size; defaults to checkpoint nx")
    parser.add_argument("--depth", type=int,
                        help="Depth/time tile size; defaults to checkpoint nz")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--tile-batch", type=int, default=8)
    parser.add_argument("--figure-width", type=float)
    parser.add_argument("--figure-height", type=float)
    parser.add_argument(
        "--title",
        default="ERA5 Horizontal Wind-Field Reconstruction from Sparse Observations",
    )
    parser.add_argument("--u-label", default="Zonal wind component (u)")
    parser.add_argument("--v-label", default="Meridional wind component (v)")
    return parser


def main():
    render(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
