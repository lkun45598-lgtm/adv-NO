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
from matplotlib.ticker import FuncFormatter, MultipleLocator

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
        "Absolute Error\n(|Reconstruction - Ground Truth|)",
    )


def colorbar_labels() -> tuple[str, str]:
    """Return unit-labelled captions for field and absolute-error colorbars."""
    return "Velocity (m s$^{-1}$)", "Absolute error (m s$^{-1}$)"


def configure_colorbar_axis(colorbar, side: str) -> None:
    """Place colorbar ticks and its label on the requested vertical side."""
    if side not in {"left", "right"}:
        raise ValueError(f"colorbar side must be 'left' or 'right', got {side!r}")
    colorbar.ax.yaxis.set_ticks_position(side)
    colorbar.ax.yaxis.set_label_position(side)


def create_publication_axes(figure, domain_aspect: float):
    """Create the two-row publication grid with unclipped colorbar margins."""
    top = 0.88 if domain_aspect >= 1.5 else 0.89
    bottom = 0.10 if domain_aspect >= 1.5 else 0.08
    grid = GridSpec(
        2,
        8,
        figure=figure,
        # Spacers on both sides of each colorbar keep its tick labels and
        # unit caption outside the data panels.  Field colorbar ticks are on
        # the left, so they face the reconstruction panel without reaching
        # the error panel.
        width_ratios=(1.0, 1.0, 1.0, 0.12, 0.045, 0.12, 1.0, 0.045),
        left=0.075,
        right=0.94,
        bottom=bottom,
        top=top,
        wspace=0.20,
        hspace=0.18,
    )
    axes = np.empty((2, 4), dtype=object)
    field_color_axes = []
    error_color_axes = []
    for row in range(2):
        axes[row, 0] = figure.add_subplot(grid[row, 0])
        axes[row, 1] = figure.add_subplot(grid[row, 1])
        axes[row, 2] = figure.add_subplot(grid[row, 2])
        field_color_axes.append(figure.add_subplot(grid[row, 4]))
        axes[row, 3] = figure.add_subplot(grid[row, 6])
        error_color_axes.append(figure.add_subplot(grid[row, 7]))
    return axes, field_color_axes, error_color_axes


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


def _read_metadata(data_path: Path) -> dict:
    """Read the preprocessing metadata stored in an output HDF5 file."""
    with h5py.File(data_path, "r") as h5:
        raw = h5.attrs.get("metadata_json")
    if raw is None:
        raise KeyError(f"{data_path} does not contain metadata_json")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _coarse_curvilinear_centers(
    longitude: np.ndarray,
    latitude: np.ndarray,
    weights: np.ndarray,
    wet: np.ndarray,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate PRE's RHO-point coordinates onto the processed coarse grid."""
    if longitude.shape != latitude.shape or longitude.shape != weights.shape:
        raise ValueError("PRE coordinate and weight arrays must have identical shapes")
    if wet.shape != longitude.shape:
        raise ValueError("PRE wet mask must match coordinate arrays")
    height, width = longitude.shape
    if height % scale or width % scale:
        raise ValueError(f"PRE coordinate shape {(height, width)} is not divisible by {scale}")
    low_height, low_width = height // scale, width // scale
    block_shape = (low_height, scale, low_width, scale)
    block_weights = np.asarray(weights, dtype=np.float64).reshape(block_shape)
    block_wet = np.asarray(wet, dtype=bool).reshape(block_shape)
    effective_weights = np.where(block_wet, block_weights, 0.0)
    denominator = effective_weights.sum(axis=(1, 3))
    fallback_denominator = block_weights.sum(axis=(1, 3))

    def reduce_coordinate(values: np.ndarray) -> np.ndarray:
        block_values = np.asarray(values, dtype=np.float64).reshape(block_shape)
        numerator = (block_values * effective_weights).sum(axis=(1, 3))
        fallback = (block_values * block_weights).sum(axis=(1, 3))
        return np.divide(
            numerator,
            denominator,
            out=np.divide(
                fallback,
                fallback_denominator,
                out=np.full((low_height, low_width), np.nan, dtype=np.float64),
                where=fallback_denominator > 0,
            ),
            where=denominator > 0,
        )

    return reduce_coordinate(longitude), reduce_coordinate(latitude)


def load_spatial_grid(
    data_path: Path,
    expected_shape: tuple[int, int],
    pre_coordinate_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load physical coordinates matching a processed PRE or ERA5 sample.

    ERA5 returns one-dimensional cell edges for ``pcolormesh(shading='flat')``.
    PRE returns two-dimensional coarse cell centers for its curvilinear RHO grid.
    """
    metadata = _read_metadata(data_path)
    task = metadata.get("task")
    height, width = expected_shape
    if task == "D":
        if (height, width) != (180, 360):
            raise ValueError(f"ERA5 geographic grid expects (180, 360), got {expected_shape}")
        # The source has 0.25-degree centers, drops the final -90-degree row,
        # and averages 4x4 blocks.  The resulting cell edges are exactly 1 degree.
        longitude_edges = np.arange(width + 1, dtype=np.float64)
        latitude_edges = 90.0 - np.arange(height + 1, dtype=np.float64)
        return longitude_edges, latitude_edges, "flat"
    if task != "A":
        raise ValueError(f"Unsupported task in metadata: {task!r}")

    source = Path(pre_coordinate_dir) if pre_coordinate_dir else Path(metadata["source"])
    coordinate_dir = source / "processed/stat_var" if source.name != "stat_var" else source
    longitude_path = coordinate_dir / "lon_rho.npy"
    latitude_path = coordinate_dir / "lat_rho.npy"
    pm_path = coordinate_dir / "pm.npy"
    pn_path = coordinate_dir / "pn.npy"
    mask_path = coordinate_dir / "mask_rho.npy"
    missing = [str(path) for path in (longitude_path, latitude_path, pm_path, pn_path, mask_path)
               if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "PRE geographic coordinates require source static files; missing: "
            + ", ".join(missing)
        )
    longitude = np.asarray(np.load(longitude_path), dtype=np.float64)
    latitude = np.asarray(np.load(latitude_path), dtype=np.float64)
    pm = np.asarray(np.load(pm_path), dtype=np.float64)
    pn = np.asarray(np.load(pn_path), dtype=np.float64)
    wet = np.asarray(np.load(mask_path), dtype=np.float64) > 0
    source_height, source_width = map(int, metadata["source_spatial_crop"])
    longitude = longitude[:source_height, :source_width]
    latitude = latitude[:source_height, :source_width]
    pm = pm[:source_height, :source_width]
    pn = pn[:source_height, :source_width]
    wet = wet[:source_height, :source_width]
    if (source_height, source_width) != (height * 4, width * 4):
        raise ValueError(
            "PRE coordinate crop does not match processed shape: "
            f"source {(source_height, source_width)}, processed {expected_shape}"
        )
    weights = 1.0 / (pm * pn)
    coarse_longitude, coarse_latitude = _coarse_curvilinear_centers(
        longitude, latitude, weights, wet, scale=4
    )
    return coarse_longitude, coarse_latitude, "nearest"


def plot_spatial_field(
    axis,
    longitude: np.ndarray,
    latitude: np.ndarray,
    values: np.ndarray,
    *,
    shading: str,
    cmap: str,
    vmin: float,
    vmax: float,
):
    """Plot one field with geographic coordinates instead of array indices."""
    return axis.pcolormesh(
        longitude,
        latitude,
        values,
        shading=shading,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )


def add_column_titles(figure, axes: np.ndarray, titles: tuple[str, ...]):
    """Place column titles in figure space above axes, avoiding image overlap."""
    if len(titles) != axes.shape[1]:
        raise ValueError(f"Expected {axes.shape[1]} column titles, got {len(titles)}")
    figure.canvas.draw()
    artists = []
    for axis, title in zip(axes[0], titles):
        position = axis.get_position()
        artists.append(
            figure.text(
                (position.x0 + position.x1) / 2,
                position.y1 + 0.065,
                title,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
                color="#111827",
            )
        )
    return artists


def geographic_extent_caption(longitude: np.ndarray, latitude: np.ndarray) -> str:
    """Format the plotted longitude/latitude range for the figure subtitle."""
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    lon_min, lon_max = float(np.nanmin(longitude)), float(np.nanmax(longitude))
    lat_min, lat_max = float(np.nanmin(latitude)), float(np.nanmax(latitude))

    def directional_range(low: float, high: float, negative: str, positive: str) -> str:
        if low >= 0:
            return f"{low:.2f}–{high:.2f}°{positive}"
        if high <= 0:
            return f"{abs(high):.2f}–{abs(low):.2f}°{negative}"
        return f"{abs(low):.0f}°{negative}–{high:.0f}°{positive}"

    longitude_range = directional_range(lon_min, lon_max, "W", "E")
    latitude_range = directional_range(lat_min, lat_max, "S", "N")
    if np.allclose([lon_min, lon_max], [0.0, 360.0]):
        longitude_range = "0°E–360°E"
    return f"Longitude {longitude_range} | Latitude {latitude_range}"


def _format_degree(value: float) -> str:
    precision = 0 if np.isclose(value, round(value)) else 1
    return f"{value:.{precision}f}°"


def format_longitude_tick(value: float, _position) -> str:
    """Format a longitude tick with an unambiguous east/west suffix."""
    if np.isclose(value, 0.0):
        return "0°"
    return _format_degree(abs(value)) + ("E" if value > 0 else "W")


def format_latitude_tick(value: float, _position) -> str:
    """Format a latitude tick with an unambiguous north/south suffix."""
    if np.isclose(value, 0.0):
        return "0°"
    return _format_degree(abs(value)) + ("N" if value > 0 else "S")


def geographic_tick_interval(shading: str) -> float | None:
    """Return the major geographic tick spacing for each grid representation."""
    if shading == "nearest":
        return 1.0
    if shading == "flat":
        return None
    raise ValueError(f"unsupported geographic shading: {shading!r}")


def apply_geographic_tick_formatters(
    axes: np.ndarray, major_interval: float | None = None
) -> None:
    """Apply cardinal-direction labels and an optional fixed tick interval."""
    longitude_formatter = FuncFormatter(format_longitude_tick)
    latitude_formatter = FuncFormatter(format_latitude_tick)
    for axis in axes.flat:
        axis.xaxis.set_major_formatter(longitude_formatter)
        axis.yaxis.set_major_formatter(latitude_formatter)
        if major_interval is not None:
            if major_interval <= 0:
                raise ValueError("major_interval must be positive")
            axis.xaxis.set_major_locator(MultipleLocator(major_interval))
            axis.yaxis.set_major_locator(MultipleLocator(major_interval))


def format_row_label(label: str) -> str:
    """Wrap long wind-component labels without abbreviating their meaning."""
    if len(label) > 22 and " Wind Velocity " in label:
        return label.replace(" Wind Velocity ", " Wind\nVelocity ", 1)
    return label


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


def error_scale_caption(error_vmax: float | None) -> str:
    """Describe whether absolute-error limits support cross-rate comparison."""
    if error_vmax is None:
        return "absolute-error panels use a per-figure scale."
    return "absolute-error panels use a fixed cross-rate scale."


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
    generator_state, generator_source = select_generator_state(
        checkpoint, prefer_ema=not raw_generator
    )
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
    target, prediction, observed, valid, _generator_source = predict_sample(
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
    title, u_label, v_label = visualization_labels(args.title, args.u_label, args.v_label)
    _, height, width, _full_depth = target.shape
    longitude, latitude, shading = load_spatial_grid(
        args.data,
        expected_shape=(height, width),
        pre_coordinate_dir=args.coordinate_dir,
    )
    default_x_label = "Longitude"
    default_y_label = "Latitude"
    x_label, y_label = axis_labels(
        args.x_label or default_x_label,
        args.y_label or default_y_label,
    )
    domain_aspect = width / height
    figure_width = args.figure_width or 18.0
    figure_height = args.figure_height or (6.8 if domain_aspect >= 1.5 else 8.8)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    figure = plt.figure(figsize=(figure_width, figure_height), constrained_layout=False)
    axes, field_color_axes, error_color_axes = create_publication_axes(
        figure, domain_aspect
    )

    field_colorbar_label, error_colorbar_label = colorbar_labels()

    for row, component in enumerate(("u", "v")):
        field_images = (truth[row], sparse[row], reconstruction[row])
        field_image = None
        for col, values in enumerate(field_images):
            axis = axes[row, col]
            field_image = plot_spatial_field(
                axis,
                longitude,
                latitude,
                values,
                cmap=FIELD_CMAP,
                vmin=-field_limits[row],
                vmax=field_limits[row],
                shading=shading,
            )

        field_colorbar = figure.colorbar(field_image, cax=field_color_axes[row])
        configure_colorbar_axis(field_colorbar, side="left")
        field_colorbar.ax.tick_params(labelsize=10, length=3)
        field_colorbar.set_label(
            field_colorbar_label, fontsize=10, rotation=270, labelpad=15
        )

        axis = axes[row, 3]
        error_image = plot_spatial_field(
            axis,
            longitude,
            latitude,
            absolute_error[row],
            cmap=ERROR_CMAP,
            vmin=0,
            vmax=float(error_scales[row]),
            shading=shading,
        )
        error_colorbar = figure.colorbar(error_image, cax=error_color_axes[row])
        configure_colorbar_axis(error_colorbar, side="right")
        error_colorbar.ax.tick_params(labelsize=10, length=3)
        error_colorbar.set_label(
            error_colorbar_label,
            fontsize=10,
            rotation=270,
            labelpad=17,
        )

    apply_axis_visibility(axes, x_label, y_label)
    apply_geographic_tick_formatters(
        axes, major_interval=geographic_tick_interval(shading)
    )
    for axis in axes.flat:
        axis.set_aspect("auto")
    add_column_titles(figure, axes, titles)
    figure.canvas.draw()
    for row, row_label in enumerate((u_label, v_label)):
        bounds = axes[row, 0].get_position()
        figure.text(
            0.022,
            (bounds.y0 + bounds.y1) / 2,
            format_row_label(row_label),
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
        y=0.995,
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
    parser.add_argument("--x-label", default=None,
                        help="Override the physical horizontal coordinate label")
    parser.add_argument("--y-label", default=None,
                        help="Override the physical vertical coordinate label")
    parser.add_argument("--coordinate-dir", type=Path, default=None,
                        help="PRE processed/stat_var directory containing lon/lat arrays")
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
    parser.add_argument("--u-label", default="Zonal Wind Velocity (u)")
    parser.add_argument("--v-label", default="Meridional Wind Velocity (v)")
    return parser


def main():
    render(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
