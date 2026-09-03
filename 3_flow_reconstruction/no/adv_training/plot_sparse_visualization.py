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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tcunet import Unet3D  # noqa: E402
from train_sparse_adv_no import make_mask, select_generator_state  # noqa: E402


FIELD_CMAP = "RdBu_r"
ERROR_CMAP = "magma"


def panel_titles(mask_ratio: float) -> tuple[str, str, str, str]:
    """Return panel labels whose observation rate matches the rendered mask."""
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask ratio must be in [0, 1], got {mask_ratio}")
    return (
        "Reference field",
        f"Sparse observations ({1.0 - mask_ratio:.0%} observed)",
        "adv-NO reconstruction",
        "Absolute reconstruction error",
    )


def visualization_labels(title: str, u_label: str, v_label: str) -> tuple[str, str, str]:
    """Return caller-supplied labels for the task-specific figure header."""
    return title, u_label, v_label


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


@torch.no_grad()
def predict_sample(
    data_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    sample: int,
    time_index: int,
    mask_ratio: float,
    seed: int,
    device_name: str,
    raw_generator: bool = False,
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

    with h5py.File(data_path, "r") as h5:
        fields = h5["test"]["field"]
        valid_masks = h5["test"]["valid_mask"]
        if not 0 <= sample < fields.shape[0]:
            raise IndexError(f"sample must be in [0, {fields.shape[0]})")
        _, _, height, width, depth = fields.shape
        patch = int(par["nx"])
        model_depth = int(par["nz"])
        if patch > min(height, width) or model_depth > depth:
            raise ValueError(f"checkpoint patch {(patch, model_depth)} exceeds test shape {fields.shape}")
        if not 0 <= time_index < depth:
            raise IndexError(f"time_index must be in [0, {depth})")
        y0 = (height - patch) // 2
        x0 = (width - patch) // 2
        z0 = max(0, min(time_index - model_depth // 2, depth - model_depth))
        field = np.asarray(fields[sample, :, y0:y0 + patch, x0:x0 + patch, z0:z0 + model_depth], dtype=np.float32)
        valid = np.asarray(valid_masks[sample, :, y0:y0 + patch, x0:x0 + patch, z0:z0 + model_depth], dtype=np.float32)

    field_t = torch.from_numpy(field[None]).to(device)
    valid_t = torch.from_numpy(valid[None]).to(device)
    generator = torch.Generator(device=device.type).manual_seed(seed + sample)
    mean_t = par["inp_shift"][:, :2]
    model_input, observed_t = make_mask(
        field_t,
        valid_t,
        mask_ratio,
        mean_t=mean_t,
        generator=generator,
    )
    prediction = model(model_input, torch.ones(1, device=device))[0].cpu().numpy()
    observed = observed_t[0, 0].cpu().numpy().astype(bool)
    target = field
    return target, prediction, observed, valid, z0, model_depth


def render(args):
    target, prediction, observed, valid, z0, model_depth = predict_sample(
        args.data,
        args.checkpoint,
        args.config,
        args.sample,
        args.time_index,
        args.mask_ratio,
        args.seed,
        args.device,
        args.raw_generator,
    )
    slice_index = args.time_index - z0
    valid_slice = valid[0, :, :, slice_index]
    truth = mask_invalid_land(target[:, :, :, slice_index], valid_slice)
    sparse = mask_sparse_observations(truth, observed[:, :, slice_index], valid_slice)
    reconstruction = mask_invalid_land(prediction[:, :, :, slice_index], valid_slice)
    absolute_error = np.abs(reconstruction - truth)
    titles = panel_titles(args.mask_ratio)

    field_limits = np.maximum(
        np.nanmax(np.abs(truth), axis=(1, 2)),
        np.nanmax(np.abs(reconstruction), axis=(1, 2)),
    )
    error_limits = np.nanmax(absolute_error, axis=(1, 2))
    title, u_label, v_label = visualization_labels(args.title, args.u_label, args.v_label)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    figure, axes = plt.subplots(2, 4, figsize=(15.6, 7.2), constrained_layout=False)
    figure.subplots_adjust(left=0.055, right=0.965, bottom=0.105, top=0.825,
                          wspace=0.30, hspace=0.34)

    for row, component in enumerate(("u", "v")):
        field_images = (truth[row], sparse[row], reconstruction[row])
        for col, values in enumerate(field_images):
            axis = axes[row, col]
            image = axis.imshow(values, cmap=FIELD_CMAP, vmin=-field_limits[row], vmax=field_limits[row],
                                interpolation="nearest", origin="lower", aspect="equal")
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
            colorbar.ax.tick_params(labelsize=7)
            colorbar.set_label(f"{component} (m s$^{{-1}}$)", fontsize=8, rotation=270,
                               labelpad=11)
            axis.set_title(titles[col], pad=6)
            axis.set_xlabel("Longitude index")
            axis.set_ylabel("Latitude index")
            axis.tick_params(length=2)

        axis = axes[row, 3]
        image = axis.imshow(absolute_error[row], cmap=ERROR_CMAP, vmin=0,
                            vmax=float(error_limits[row]), interpolation="nearest",
                            origin="lower", aspect="equal")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(f"|{component} error| (m s$^{{-1}}$)", fontsize=8,
                           rotation=270, labelpad=11)
        axis.set_title(titles[3], pad=6)
        axis.set_xlabel("Longitude index")
        axis.set_ylabel("Latitude index")
        axis.tick_params(length=2)

    figure.text(0.012, 0.635, u_label, rotation=90,
                ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2937")
    figure.text(0.012, 0.340, v_label, rotation=90,
                ha="center", va="center", fontsize=10, fontweight="bold", color="#1F2937")

    figure.suptitle(
        title,
        fontsize=15, fontweight="bold", y=0.965,
    )
    figure.text(
        0.5, 0.925,
        f"Held-out test sample {args.sample} | central {target.shape[1]} x {target.shape[2]} tile "
        f"| time step {args.time_index + 1} of {target.shape[3]} "
        f"| missing rate: {args.mask_ratio:.0%} "
        f"| observed fraction: {1.0 - args.mask_ratio:.0%}",
        ha="center", va="center", fontsize=10, color="#334155",
    )
    figure.text(0.5, 0.035,
                "Color limits are shared across reference, observations, and reconstruction; "
                "error maps show absolute differences.",
                ha="center", va="center", fontsize=8, color="#475569")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-generator", action="store_true",
                        help="Use raw generator weights instead of EMA weights when present")
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
