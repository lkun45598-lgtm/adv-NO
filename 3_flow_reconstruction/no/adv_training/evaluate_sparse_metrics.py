"""Run tiled sparse-field inference and report comprehensive masked metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tcunet import Unet3D  # noqa: E402
from train_sparse_adv_no import make_mask, select_generator_state  # noqa: E402
from metrics_sparse import masked_metrics, macro_average, vector_epe  # noqa: E402


def tile_starts(length: int, size: int, stride: int) -> list[int]:
    if size <= 0 or stride <= 0 or size > length:
        raise ValueError(f"invalid tile size/stride for axis length {length}: {size}, {stride}")
    starts = list(range(0, length - size + 1, stride))
    last = length - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def nearest_fill(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fill each channel from its nearest observed 3D voxel."""
    values = np.asarray(values, dtype=np.float32)
    observed = np.broadcast_to(np.asarray(observed, dtype=bool), values.shape)
    result = values.copy()
    common = observed[0]
    if not common.any():
        return result
    indices = distance_transform_edt(~common, return_distances=False, return_indices=True)
    nearest = (indices[0], indices[1], indices[2])
    for channel in range(values.shape[0]):
        result[channel][~observed[channel]] = values[channel][nearest][~observed[channel]]
    return result


def gaussian_fill(values: np.ndarray, observed: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Normalized Gaussian interpolation that keeps observed values exact."""
    values = np.asarray(values, dtype=np.float32)
    observed = np.broadcast_to(np.asarray(observed, dtype=bool), values.shape)
    result = values.copy()
    for channel in range(values.shape[0]):
        mask = observed[channel].astype(np.float32)
        numerator = gaussian_filter(np.where(observed[channel], values[channel], 0.0), sigma=sigma)
        denominator = gaussian_filter(mask, sigma=sigma)
        filled = numerator / np.maximum(denominator, np.finfo(np.float32).eps)
        result[channel][~observed[channel]] = filled[~observed[channel]]
    return result


@torch.no_grad()
def tiled_predict(model, model_input, patch: int, depth: int, stride: int, tile_batch: int):
    _, _, height, width, full_depth = model_input.shape
    y_starts = tile_starts(height, patch, stride)
    x_starts = tile_starts(width, patch, stride)
    z_starts = tile_starts(full_depth, depth, max(1, depth // 2))
    tiles = []
    locations = []
    output = torch.zeros((1, 2, height, width, full_depth), device=model_input.device)
    weights = torch.zeros_like(output)

    def flush():
        if not tiles:
            return
        batch = torch.cat(tiles, dim=0)
        prediction = model(batch, torch.ones(batch.shape[0], device=batch.device))
        for index, (y0, x0, z0) in enumerate(locations):
            output[..., y0:y0 + patch, x0:x0 + patch, z0:z0 + depth] += prediction[index:index + 1]
            weights[..., y0:y0 + patch, x0:x0 + patch, z0:z0 + depth] += 1.0
        tiles.clear()
        locations.clear()

    for y0 in y_starts:
        for x0 in x_starts:
            for z0 in z_starts:
                tiles.append(model_input[..., y0:y0 + patch, x0:x0 + patch, z0:z0 + depth])
                locations.append((y0, x0, z0))
                if len(tiles) >= tile_batch:
                    flush()
    flush()
    return output / weights.clamp_min(1.0)


def _training_ranges(h5_path: Path, split: str = "train") -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        fields = h5[split]["field"]
        valid = h5[split]["valid_mask"]
        ranges = []
        for channel in range(fields.shape[1]):
            minimum, maximum = np.inf, -np.inf
            for index in range(fields.shape[0]):
                values = fields[index, channel]
                mask = valid[index, 0].astype(bool) & np.isfinite(values)
                if mask.any():
                    minimum = min(minimum, float(values[mask].min()))
                    maximum = max(maximum, float(values[mask].max()))
            span = maximum - minimum
            ranges.append(span if np.isfinite(span) and span > 0 else 1.0)
    return np.asarray(ranges, dtype=np.float64)


def _clean_json(value):
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    par = checkpoint["par"]
    par = {key: value.to(device) if torch.is_tensor(value) else value for key, value in par.items()}
    model = Unet3D(dim=int(config.get("dim", 16)), Par=par, dim_mults=(1, 2, 4, 8), channels=4).to(device)
    generator_state, generator_source = select_generator_state(
        checkpoint, prefer_ema=not args.raw_generator
    )
    model.load_state_dict(generator_state)
    model.eval()
    mean_t = par["inp_shift"][:, :2]

    ranges = _training_ranges(args.data)
    target_records, valid_records, observed_records = [], [], []
    predictions = {"model": [], "nearest": [], "gaussian": []}
    with h5py.File(args.data, "r") as h5:
        split = h5[args.split]
        total = min(split["field"].shape[0], args.max_samples) if args.max_samples else split["field"].shape[0]
        for index in range(total):
            field = np.asarray(split["field"][index], dtype=np.float32)
            valid = np.asarray(split["valid_mask"][index], dtype=np.float32)
            field_t = torch.from_numpy(field[None]).to(device)
            valid_t = torch.from_numpy(valid[None]).to(device)
            generator = torch.Generator(device=device.type).manual_seed(args.seed + index)
            model_input, observed_t = make_mask(field_t, valid_t, args.mask_ratio, mean_t=mean_t, generator=generator)
            prediction = tiled_predict(model, model_input, args.patch, args.depth, args.stride, args.tile_batch)
            observed = observed_t[0].cpu().numpy().astype(bool)
            predictions["model"].append(prediction[0].cpu().numpy())
            predictions["nearest"].append(nearest_fill(field, observed))
            predictions["gaussian"].append(gaussian_fill(field, observed, args.gaussian_sigma))
            target_records.append(field)
            valid_records.append(valid.astype(bool))
            observed_records.append(observed)
            if (index + 1) % 10 == 0 or index + 1 == total:
                print(f"inference {index + 1}/{total}", flush=True)

    target = np.concatenate(target_records, axis=3)
    valid = np.concatenate(valid_records, axis=3)
    observed = np.concatenate(observed_records, axis=3)
    results = {
        "metadata": {
            "data": str(args.data), "checkpoint": str(args.checkpoint), "split": args.split,
            "samples": total, "mask_ratio": args.mask_ratio, "seed": args.seed,
            "patch": args.patch, "depth": args.depth, "stride": args.stride,
            "device": str(device), "data_range": ranges.tolist(),
            "generator_weights": generator_source,
        },
        "methods": {},
    }
    masks = {"all": valid, "observed": observed, "missing": valid & ~observed}
    for method, records in predictions.items():
        prediction = np.concatenate(records, axis=3)
        results["methods"][method] = {}
        for name, score_mask in masks.items():
            metrics = masked_metrics(prediction, target, valid=valid, score_mask=score_mask,
                                     data_range=ranges, ssim_window=args.ssim_window)
            metrics["macro"] = macro_average(metrics)
            metrics["epe"] = vector_epe(prediction, target, valid=valid, score_mask=score_mask)
            results["methods"][method][name] = metrics
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_clean_json(results), indent=2))
    print(json.dumps(_clean_json(results), indent=2))


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--tile-batch", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--ssim-window", type=int, default=7)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--raw-generator", action="store_true",
                        help="Use raw generator weights instead of EMA weights when present")
    return parser


def main():
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
