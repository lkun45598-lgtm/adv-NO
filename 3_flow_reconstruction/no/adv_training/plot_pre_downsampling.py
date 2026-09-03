"""Visualize candidate 2x-time / 4x-space downsampling methods for PRE u/v.

The script only reads the raw PRE arrays and writes a PNG comparison.  It does
not alter the prepared training dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F


SCALE = 4


def temporal_mean(values: np.ndarray) -> np.ndarray:
    """Average a two-frame sequence while retaining only finite values."""
    valid = np.isfinite(values)
    numerator = np.where(valid, values, 0.0).sum(axis=0)
    denominator = valid.sum(axis=0)
    result = np.full(values.shape[1:], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def block_reduce(values: np.ndarray, wet: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average two time frames and each valid 4x4 horizontal block."""
    height, width = values.shape[-2:]
    valid = np.isfinite(values) & wet[None, :, :]
    values5 = values.reshape(2, height // SCALE, SCALE, width // SCALE, SCALE)
    valid5 = valid.reshape(2, height // SCALE, SCALE, width // SCALE, SCALE)
    weights5 = weights.reshape(height // SCALE, SCALE, width // SCALE, SCALE)
    weights5 = weights5[None, :, :, :, :]
    numerator = np.where(valid5, values5 * weights5, 0.0).sum(axis=(0, 2, 4))
    denominator = np.where(valid5, weights5, 0.0).sum(axis=(0, 2, 4))
    result = np.full((height // SCALE, width // SCALE), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def bicubic_reduce(values: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Compute a mask-normalized bicubic baseline for visual comparison only."""
    height, width = values.shape[-2:]
    valid = np.isfinite(values) & wet[None, :, :]
    numerator = np.where(valid, values, 0.0).sum(axis=0)
    denominator = valid.sum(axis=0).astype(np.float64)
    numerator_t = torch.from_numpy(numerator.astype(np.float32))[None, None]
    denominator_t = torch.from_numpy(denominator.astype(np.float32))[None, None]
    with torch.no_grad():
        small_num = F.interpolate(
            numerator_t, size=(height // SCALE, width // SCALE), mode="bicubic", align_corners=False
        )[0, 0].numpy()
        small_den = F.interpolate(
            denominator_t, size=(height // SCALE, width // SCALE), mode="bicubic", align_corners=False
        )[0, 0].numpy()
    result = np.full((height // SCALE, width // SCALE), np.nan, dtype=np.float64)
    np.divide(small_num, small_den, out=result, where=small_den > 0.5)
    return result


def enlarge(values: np.ndarray) -> np.ndarray:
    """Enlarge an LR field only for side-by-side display; no interpolation."""
    return np.repeat(np.repeat(values, SCALE, axis=0), SCALE, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/data/PRE_ocean_data/processed"))
    parser.add_argument("--time-index", type=int, default=3000)
    parser.add_argument("--sigma-index", type=int, default=29)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pre_downsampling_comparison.png"),
    )
    args = parser.parse_args()

    u_raw = np.load(args.source / "dyn_var/u_eastward.npy", mmap_mode="r")
    v_raw = np.load(args.source / "dyn_var/v_northward.npy", mmap_mode="r")
    wet = np.load(args.source / "stat_var/mask_rho.npy", mmap_mode="r") > 0
    pm = np.load(args.source / "stat_var/pm.npy", mmap_mode="r")
    pn = np.load(args.source / "stat_var/pn.npy", mmap_mode="r")
    s_rho = np.load(args.source / "stat_var/s_rho.npy", mmap_mode="r")
    if not 0 <= args.time_index < u_raw.shape[0] - 1:
        raise ValueError(f"--time-index must lie in [0, {u_raw.shape[0] - 2}]")
    if not 0 <= args.sigma_index < u_raw.shape[1]:
        raise ValueError(f"--sigma-index must lie in [0, {u_raw.shape[1] - 1}]")

    height = (u_raw.shape[2] // SCALE) * SCALE
    width = (u_raw.shape[3] // SCALE) * SCALE
    wet = np.asarray(wet[:height, :width])
    area = np.asarray(1.0 / (pm[:height, :width] * pn[:height, :width]), dtype=np.float64)

    def load_pair(raw: np.ndarray) -> np.ndarray:
        result = np.asarray(
            raw[args.time_index:args.time_index + 2, args.sigma_index, :height, :width],
            dtype=np.float64,
        )
        result[:, ~wet] = np.nan
        return result

    u = load_pair(u_raw)
    v = load_pair(v_raw)
    hr_u, hr_v = temporal_mean(u), temporal_mean(v)
    current_u = enlarge(u[0, ::SCALE, ::SCALE])
    current_v = enlarge(v[0, ::SCALE, ::SCALE])
    arithmetic_u, arithmetic_v = block_reduce(u, wet, np.ones_like(area)), block_reduce(v, wet, np.ones_like(area))
    area_u, area_v = block_reduce(u, wet, area), block_reduce(v, wet, area)
    bicubic_u, bicubic_v = bicubic_reduce(u, wet), bicubic_reduce(v, wet)

    methods = [
        ("HR: 2-time valid mean", hr_u, hr_v),
        ("Current: point stride\\n(t0, every 4th point)", current_u, current_v),
        ("2t + 4x4 arithmetic mean", enlarge(arithmetic_u), enlarge(arithmetic_v)),
        ("2t + 4x4 area-weighted mean", enlarge(area_u), enlarge(area_v)),
        ("2t + 4x4 bicubic baseline", enlarge(bicubic_u), enlarge(bicubic_v)),
    ]

    def ocean_only(values: np.ndarray) -> np.ndarray:
        return np.where(wet, values, np.nan)

    u_limit = max(float(np.nanpercentile(np.abs(hr_u), 99)), 1e-6)
    v_limit = max(float(np.nanpercentile(np.abs(hr_v), 99)), 1e-6)
    speed_limit = max(float(np.nanpercentile(np.hypot(hr_u, hr_v), 99)), 1e-6)
    cmap_div = plt.get_cmap("RdBu_r").copy()
    cmap_div.set_bad("#202124")
    cmap_mag = plt.get_cmap("viridis").copy()
    cmap_mag.set_bad("#202124")
    row_defs = [
        ("u eastward (m/s)", cmap_div, TwoSlopeNorm(vmin=-u_limit, vcenter=0, vmax=u_limit)),
        ("v northward (m/s)", cmap_div, TwoSlopeNorm(vmin=-v_limit, vcenter=0, vmax=v_limit)),
        ("speed |u,v| (m/s)", cmap_mag, Normalize(vmin=0, vmax=speed_limit)),
    ]

    fig, axes = plt.subplots(3, len(methods), figsize=(21, 11), layout="constrained")
    for column, (title, u_field, v_field) in enumerate(methods):
        fields = (u_field, v_field, np.hypot(u_field, v_field))
        for row, (label, cmap, norm) in enumerate(row_defs):
            ax = axes[row, column]
            image = ax.imshow(
                ocean_only(fields[row]), cmap=cmap, norm=norm, origin="lower", interpolation="nearest"
            )
            if row == 0:
                ax.set_title(title, fontsize=11)
            if column == 0:
                ax.set_ylabel(label, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == len(methods) - 1:
                colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
                colorbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        f"PRE u/v downsampling comparison | raw frames {args.time_index} and {args.time_index + 1}, "
        f"sigma index {args.sigma_index} (s_rho={s_rho[args.sigma_index]:.4f}) | "
        f"400x441 -> crop 400x440 -> 100x110",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
