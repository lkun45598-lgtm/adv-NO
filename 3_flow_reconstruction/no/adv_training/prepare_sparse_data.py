"""Prepare PRE (task A) or ERA5 (task D) data for sparse adv-NO reconstruction.

The output is one HDF5 file with ``train``, ``val`` and ``test`` groups.  Each
group contains ``field`` in ``[N, channels, height, width, depth]`` format and
``valid_mask`` in ``[N, 1, height, width, depth]`` format.  Temporal records
are split chronologically (8:1:1) after 2x temporal averaging and conservative
4x4 area-weighted spatial averaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def split_bounds(n: int) -> dict[str, tuple[int, int]]:
    """Return chronological 8:1:1 bounds, keeping every record assigned once."""
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }


def paired_record_starts(record_count: int) -> np.ndarray:
    """Return the starts of non-overlapping two-record temporal averages."""
    return np.arange(record_count // 2, dtype=np.int64) * 2


def conservative_downsample(values: np.ndarray, wet: np.ndarray,
                            spatial_weights: np.ndarray, scale: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce two time records and square horizontal blocks without using land.

    ``values`` has shape ``[2, channels, depth, height, width]``.  The result
    is ``[channels, depth, height / scale, width / scale]``.  Components are
    averaged independently, while the validity mask requires every output
    component to be finite at a given depth.
    """
    if values.ndim != 5 or values.shape[0] != 2:
        raise ValueError(f"Expected [2, channels, depth, height, width], got {values.shape}")
    height, width = values.shape[-2:]
    if height % scale or width % scale:
        raise ValueError(f"Spatial shape {(height, width)} is not divisible by {scale}")
    if wet.shape != (height, width) or spatial_weights.shape != (height, width):
        raise ValueError("wet mask and spatial weights must match values' horizontal shape")
    if not np.all(np.isfinite(spatial_weights)) or np.any(spatial_weights <= 0):
        raise ValueError("spatial weights must be finite and positive")

    low_height, low_width = height // scale, width // scale
    value_blocks = values.reshape(2, values.shape[1], values.shape[2], low_height, scale, low_width, scale)
    finite_blocks = (np.isfinite(values) & wet[None, None, None, :, :]).reshape(value_blocks.shape)
    weight_blocks = spatial_weights.reshape(low_height, scale, low_width, scale)
    weight_blocks = weight_blocks[None, None, None, :, :, :, :]
    numerator = np.where(finite_blocks, value_blocks * weight_blocks, 0.0).sum(axis=(0, 4, 6))
    denominator = np.where(finite_blocks, weight_blocks, 0.0).sum(axis=(0, 4, 6))
    field = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=field, where=denominator > 0)

    wet_blocks = wet.reshape(low_height, scale, low_width, scale)
    total_area = weight_blocks[0, 0, 0].sum(axis=(1, 3))
    wet_area = np.where(wet_blocks, weight_blocks[0, 0, 0], 0.0).sum(axis=(1, 3))
    wet_fraction = np.zeros((low_height, low_width), dtype=np.float32)
    np.divide(wet_area, total_area, out=wet_fraction, where=total_area > 0)
    valid = np.isfinite(field).all(axis=0)
    return field, valid, wet_fraction


def downsample_regular_global_uv_pair(values: np.ndarray, latitude_weights: np.ndarray,
                                      scale: int = 4) -> np.ndarray:
    """Fast, equivalent reduction for finite ``[2, 2, latitude, longitude]`` ERA5 pairs."""
    if values.ndim != 4 or values.shape[:2] != (2, 2):
        raise ValueError(f"Expected [2, 2, latitude, longitude], got {values.shape}")
    height, width = values.shape[-2:]
    if height % scale or width % scale or latitude_weights.shape != (height,):
        raise ValueError("ERA5 spatial shape and latitude weights must align with the scale")
    if not np.all(np.isfinite(values)):
        raise ValueError("Fast ERA5 reduction accepts finite values only")
    blocks = values.mean(axis=0).reshape(2, height // scale, scale, width // scale, scale)
    weights = latitude_weights.reshape(height // scale, scale)
    numerator = (blocks * weights[None, :, :, None, None]).sum(axis=(2, 4))
    denominator = (weights.sum(axis=1) * scale)[None, :, None]
    return numerator / denominator


def downsample_regular_global_uv_pair_torch(values: torch.Tensor, latitude_weights: np.ndarray,
                                            scale: int = 4) -> torch.Tensor:
    """Torch equivalent of the finite ERA5 reduction without copying mmap slices."""
    if values.ndim != 4 or tuple(values.shape[:2]) != (2, 2):
        raise ValueError(f"Expected [2, 2, latitude, longitude], got {tuple(values.shape)}")
    height, width = values.shape[-2:]
    if height % scale or width % scale or latitude_weights.shape != (height,):
        raise ValueError("ERA5 spatial shape and latitude weights must align with the scale")
    if not torch.isfinite(values).all():
        raise ValueError("Fast ERA5 reduction accepts finite values only")
    blocks = values.mean(dim=0).reshape(2, height // scale, scale, width // scale, scale)
    weights = torch.as_tensor(latitude_weights, dtype=values.dtype, device=values.device)
    numerator = (blocks * weights.reshape(1, height // scale, scale, 1, 1)).sum(dim=(2, 4))
    denominator = (weights.reshape(height // scale, scale).sum(dim=1) * scale)[None, :, None]
    return numerator / denominator


def downsample_masked_uv_pair_torch(values: torch.Tensor, wet: torch.Tensor,
                                    spatial_weights: torch.Tensor,
                                    scale: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast PRE reduction for land-only NaNs, with the generic path handling exceptions."""
    if values.ndim != 5 or tuple(values.shape[:2]) != (2, 2):
        raise ValueError(f"Expected [2, 2, depth, height, width], got {tuple(values.shape)}")
    height, width = values.shape[-2:]
    if height % scale or width % scale or tuple(wet.shape) != (height, width):
        raise ValueError("PRE spatial shape and wet mask must align with the scale")
    if tuple(spatial_weights.shape) != (height, width):
        raise ValueError("PRE spatial weights must match the wet mask")
    wet = wet.to(device=values.device, dtype=torch.bool)
    spatial_weights = spatial_weights.to(device=values.device, dtype=values.dtype)
    finite = torch.isfinite(values)
    if ((~finite) & wet[None, None, None]).any():
        raise ValueError("Fast PRE reduction accepts finite values over wet cells only")

    low_height, low_width = height // scale, width // scale
    weights = spatial_weights * wet.to(dtype=values.dtype)
    total_area = spatial_weights.reshape(low_height, scale, low_width, scale).sum(dim=(1, 3))
    wet_area = weights.reshape(low_height, scale, low_width, scale).sum(dim=(1, 3))
    wet_fraction = wet_area / total_area
    denominator = wet_area[None, None]
    time_mean = torch.nan_to_num(values, nan=0.0).mean(dim=0)
    blocks = time_mean.reshape(2, values.shape[2], low_height, scale, low_width, scale)
    weighted = blocks * weights.reshape(1, 1, low_height, scale, low_width, scale)
    field = weighted.sum(dim=(3, 5)) / denominator
    valid = (wet_area > 0).expand(values.shape[2], low_height, low_width)
    return field, valid, wet_fraction


def downsample_masked_uv_batch_torch(values: torch.Tensor, wet: torch.Tensor,
                                     spatial_weights: torch.Tensor,
                                     scale: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch form of :func:`downsample_masked_uv_pair_torch`.

    ``values`` is ``[batch, 2 (time), 2 (channel), depth, height, width]``.  The computation is
    vectorized over records so Torch can use its configured CPU thread pool,
    while the caller controls the batch size to keep mmap and output writes
    bounded.
    """
    if values.ndim != 6 or tuple(values.shape[1:3]) != (2, 2):
        raise ValueError(f"Expected [batch, 2, 2, depth, height, width], got {tuple(values.shape)}")
    height, width = values.shape[-2:]
    if height % scale or width % scale:
        raise ValueError(f"Spatial shape {(height, width)} is not divisible by {scale}")
    if tuple(wet.shape) != (height, width) or tuple(spatial_weights.shape) != (height, width):
        raise ValueError("PRE wet mask and spatial weights must match values' horizontal shape")

    wet = wet.to(device=values.device, dtype=torch.bool)
    spatial_weights = spatial_weights.to(device=values.device, dtype=values.dtype)
    finite = torch.isfinite(values)
    if ((~finite) & wet[None, None, None, None]).any():
        raise ValueError("Fast PRE reduction accepts finite values over wet cells only")

    low_height, low_width = height // scale, width // scale
    weights = spatial_weights * wet.to(dtype=values.dtype)
    weight_blocks = weights.reshape(low_height, scale, low_width, scale)
    total_blocks = spatial_weights.reshape(low_height, scale, low_width, scale)
    wet_area = weight_blocks.sum(dim=(1, 3))
    total_area = total_blocks.sum(dim=(1, 3))
    wet_fraction = wet_area / total_area

    time_mean = torch.nan_to_num(values, nan=0.0).mean(dim=1)
    blocks = time_mean.reshape(
        values.shape[0], 2, values.shape[3], low_height, scale, low_width, scale
    )
    weighted = blocks * weights.reshape(1, 1, 1, low_height, scale, low_width, scale)
    field = weighted.sum(dim=(4, 6)) / wet_area.reshape(1, 1, 1, low_height, low_width)
    valid = (wet_area > 0).reshape(1, 1, low_height, low_width).expand(
        values.shape[0], values.shape[3], low_height, low_width
    )
    return field, valid, wet_fraction


def _create_group(h5: h5py.File, name: str, length: int, channels: int, height: int,
                  width: int, depth: int, wet_fraction: np.ndarray,
                  compression: str | None) -> tuple[h5py.Dataset, h5py.Dataset]:
    group = h5.create_group(name)
    field_kwargs = {"shape": (length, channels, height, width, depth), "dtype": "float32"}
    valid_kwargs = {"shape": (length, 1, height, width, depth), "dtype": "uint8"}
    if length > 0:
        field_kwargs.update({"chunks": (1, channels, height, width, depth), "compression": compression})
        valid_kwargs.update({"chunks": (1, 1, height, width, depth), "compression": compression})
    field = group.create_dataset("field", **field_kwargs)
    valid = group.create_dataset("valid_mask", **valid_kwargs)
    if wet_fraction.shape != (height, width):
        raise ValueError(f"wet_fraction shape {wet_fraction.shape} != {(height, width)}")
    group.create_dataset("wet_fraction", data=wet_fraction[None, :, :, None], dtype="float32")
    return field, valid


def _write_group(h5: h5py.File, name: str, fields: np.ndarray, masks: np.ndarray,
                 start: int, end: int, wet_fraction: np.ndarray,
                 chunk_size: int = 8, compression: str | None = "lzf") -> None:
    n, channels, height, width, depth = fields.shape
    out_field, out_mask = _create_group(
        h5, name, end - start, channels, height, width, depth, wet_fraction, compression
    )
    for pos in range(start, end, chunk_size):
        stop = min(pos + chunk_size, end)
        block = np.asarray(fields[pos:stop], dtype=np.float32)
        block_mask = np.asarray(masks[pos:stop], dtype=np.uint8)
        block[~np.isfinite(block)] = 0.0
        out_field[pos - start:stop - start] = block
        out_mask[pos - start:stop - start] = block_mask


def prepare_pre(source: Path, output: Path, max_records: int | None,
                record_start: int = 0, record_count: int | None = None,
                append: bool = False, compression: str | None = "lzf",
                batch_records: int = 16) -> dict:
    u = np.load(source / "processed/dyn_var/u_eastward.npy", mmap_mode="r")
    v = np.load(source / "processed/dyn_var/v_northward.npy", mmap_mode="r")
    land_mask = np.load(source / "processed/stat_var/mask_rho.npy", mmap_mode="r")
    pm = np.load(source / "processed/stat_var/pm.npy", mmap_mode="r")
    pn = np.load(source / "processed/stat_var/pn.npy", mmap_mode="r")
    if u.shape != v.shape or u.ndim != 4:
        raise ValueError(f"Expected aligned PRE u/v arrays, got {u.shape} and {v.shape}")
    if any(array.shape != u.shape[2:] for array in (land_mask, pm, pn)):
        raise ValueError("PRE mask_rho, pm, and pn must match the u/v horizontal shape")
    # Use only complete 4x4 blocks.  The final source column is explicitly dropped.
    source_height = (u.shape[2] // 4) * 4
    source_width = (u.shape[3] // 4) * 4
    wet = np.asarray(land_mask[:source_height, :source_width] > 0)
    cell_area = np.asarray(1.0 / (pm[:source_height, :source_width] * pn[:source_height, :source_width]), dtype=np.float64)
    all_indices = paired_record_starts(u.shape[0])
    if max_records is not None:
        all_indices = all_indices[:max_records]
    total_records = len(all_indices)
    record_end = total_records if record_count is None else min(record_start + record_count, total_records)
    if record_start < 0 or record_start >= total_records or record_end <= record_start:
        raise ValueError(f"Invalid record range [{record_start}, {record_end}) for {total_records} records")
    indices = all_indices[record_start:record_end]
    height, width = source_height // 4, source_width // 4
    metadata = {"task": "A", "source": str(source),
                "variables": ["u_eastward", "v_northward"],
                "temporal_reduction": "mean of each consecutive two-record pair, excluding non-finite values",
                "spatial_reduction": "4x4 cell-area-weighted mean over wet finite source cells",
                "spatial_scale": 4,
                "source_spatial_crop": [source_height, source_width],
                "dropped_source_time_records": int(u.shape[0] % 2),
                "dropped_source_columns": int(u.shape[3] - source_width),
                "valid_mask_definition": "both reduced u/v components are finite at a wet coarse-grid location",
                "wet_fraction_definition": "wet cell area / total cell area in each coarse 4x4 block",
                "original_shape": list(u.shape),
                "processed_shape": [total_records, 2, height, width, u.shape[1]],
                "split_bounds": split_bounds(total_records), "split_ratio": "8:1:1"}
    _save_pre_streaming(output, u, v, wet, cell_area, indices, metadata,
                        record_start, append, compression, batch_records)
    return metadata


def _save_pre_streaming(output: Path, u: np.ndarray, v: np.ndarray, wet: np.ndarray,
                        cell_area: np.ndarray,
                        indices: np.ndarray, metadata: dict, record_start: int,
                        append: bool, compression: str | None,
                        batch_records: int = 16) -> None:
    if batch_records <= 0:
        raise ValueError(f"batch_records must be positive, got {batch_records}")
    output.parent.mkdir(parents=True, exist_ok=True)
    n, _, height, width, depth = metadata["processed_shape"]
    bounds = metadata["split_bounds"]
    mode = "r+" if append else "w"
    with h5py.File(output, mode) as h5:
        if append:
            if "metadata_json" not in h5.attrs:
                raise ValueError("Cannot append: output has no metadata")
            existing = json.loads(h5.attrs["metadata_json"])
            if existing.get("processed_shape") != metadata["processed_shape"]:
                raise ValueError("Append shape does not match existing output")
            groups = {name: (h5[name]["field"], h5[name]["valid_mask"])
                      for name in bounds}
        else:
            h5.attrs["metadata_json"] = json.dumps(metadata)
            groups = {}
            source_height, source_width = wet.shape
            wet_fraction = (cell_area * wet).reshape(
                source_height // 4, 4, source_width // 4, 4
            ).sum(axis=(1, 3))
            total_area = cell_area.reshape(
                source_height // 4, 4, source_width // 4, 4
            ).sum(axis=(1, 3))
            wet_fraction = np.divide(
                wet_fraction, total_area, out=np.zeros_like(wet_fraction), where=total_area > 0
            ).astype(np.float32)
            for name, (start, end) in bounds.items():
                groups[name] = _create_group(
                    h5, name, end - start, 2, height, width, depth, wet_fraction, compression
                )
        wet_tensor = torch.from_numpy(np.asarray(wet))
        area_tensor = torch.from_numpy(np.asarray(cell_area, dtype=np.float32))
        for pos in range(0, len(indices), batch_records):
            stop = min(pos + batch_records, len(indices))
            batch_starts = indices[pos:stop]
            # Starts are consecutive pairs, so one contiguous mmap slice is
            # substantially faster than NFS advanced indexing per record.
            raw_start = int(batch_starts[0])
            raw_stop = int(batch_starts[-1] + 2)
            expected_starts = np.arange(raw_start, raw_stop, 2, dtype=np.int64)
            if np.array_equal(batch_starts, expected_starts):
                u_block = np.asarray(u[raw_start:raw_stop, :depth, :wet.shape[0], :wet.shape[1]], dtype=np.float32)
                v_block = np.asarray(v[raw_start:raw_stop, :depth, :wet.shape[0], :wet.shape[1]], dtype=np.float32)
                u0, u1 = u_block[0::2], u_block[1::2]
                v0, v1 = v_block[0::2], v_block[1::2]
            else:
                u0 = np.asarray(u[batch_starts, :depth, :wet.shape[0], :wet.shape[1]], dtype=np.float32)
                u1 = np.asarray(u[batch_starts + 1, :depth, :wet.shape[0], :wet.shape[1]], dtype=np.float32)
                v0 = np.asarray(v[batch_starts, :depth, :wet.shape[0], :wet.shape[1]], dtype=np.float32)
                v1 = np.asarray(v[batch_starts + 1, :depth, :wet.shape[0], :wet.shape[1]], dtype=np.float32)
            raw_batch = np.stack((np.stack((u0, v0), axis=1), np.stack((u1, v1), axis=1)), axis=1)
            reduced, valid, _ = downsample_masked_uv_batch_torch(
                torch.from_numpy(raw_batch), wet_tensor, area_tensor
            )
            sample = reduced.permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.float32)
            mask = valid[:, None].permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
            sample[~np.isfinite(sample)] = 0.0
            global_start = record_start + pos
            global_stop = record_start + stop
            for name, (start, end) in bounds.items():
                write_start = max(global_start, start)
                write_stop = min(global_stop, end)
                if write_start >= write_stop:
                    continue
                source_start = write_start - global_start
                source_stop = write_stop - global_start
                target_start = write_start - start
                target_stop = write_stop - start
                groups[name][0][target_start:target_stop] = sample[source_start:source_stop]
                groups[name][1][target_start:target_stop] = mask[source_start:source_stop]
            if pos == 0 or stop == len(indices) or stop % (batch_records * 8) == 0:
                print(f"processed PRE records {stop}/{len(indices)}", flush=True)
        h5.flush()


def prepare_era5(source: Path, output: Path, max_records: int | None,
                 window: int = 8) -> dict:
    tensor_path = source / "era_721_1440.pt"
    tensor = torch.load(tensor_path, map_location="cpu", mmap=True, weights_only=True)
    if tuple(tensor.shape) != (20, 24, 721, 1440, 4):
        raise ValueError(f"Unexpected ERA5 tensor shape: {tuple(tensor.shape)}")
    # Flatten day/hour, reduce two adjacent hours, then conserve horizontal area.
    source_records = tensor.reshape(480, 721, 1440, 4)
    starts = paired_record_starts(source_records.shape[0])
    if max_records is not None:
        starts = starts[:max_records]
    latitude_count = 720  # Drop the south-pole endpoint so horizontal 4x4 blocks are exact.
    latitude = 90.0 - 0.25 * np.arange(latitude_count)
    latitude_weights = np.cos(np.deg2rad(latitude))
    area = np.broadcast_to(latitude_weights[:, None], (latitude_count, 1440)).copy()
    wet = np.ones((latitude_count, 1440), dtype=bool)
    records = np.empty((len(starts), 2, latitude_count // 4, 1440 // 4), dtype=np.float32)
    for record, start in enumerate(starts):
        values = source_records[start:start + 2, :latitude_count, :, 1:3].permute(0, 3, 1, 2)
        if torch.isfinite(values).all():
            records[record] = downsample_regular_global_uv_pair_torch(values, latitude_weights).numpy()
        else:
            reduced, _, _ = conservative_downsample(values.numpy()[:, :, None], wet, area)
            records[record] = reduced[:, 0]
    if records.shape[0] < window:
        raise ValueError(f"Need at least {window} downsampled ERA5 records for windows")
    # Temporal windows become the depth axis expected by the 3D adv-NO model.
    record_bounds = split_bounds(len(records))
    split_fields = {}
    for name, (start, end) in record_bounds.items():
        split_fields[name] = np.stack(
            [records[i:i + window].transpose(1, 2, 3, 0)
             for i in range(start, end - window + 1)], axis=0)
    fields = np.concatenate([split_fields[name] for name in ("train", "val", "test")], axis=0)
    masks = np.ones((len(fields), 1, fields.shape[2], fields.shape[3], fields.shape[4]),
                    dtype=np.uint8)
    window_bounds = {}
    offset = 0
    for name in ("train", "val", "test"):
        size = len(split_fields[name])
        window_bounds[name] = (offset, offset + size)
        offset += size
    return _save(output, fields, masks, {"task": "D", "source": str(tensor_path),
                                         "variables": ["u_component_of_wind", "v_component_of_wind"],
                                         "temporal_reduction": "mean of each consecutive two-hour pair",
                                         "spatial_reduction": "4x4 cos(latitude)-area-weighted mean",
                                         "spatial_scale": 4,
                                         "source_spatial_crop": [latitude_count, 1440],
                                         "dropped_source_latitude_rows": 1,
                                         "valid_mask_definition": "all ones; synthetic sparse mask is sampled during training",
                                         "wet_fraction_definition": "all ones for global ERA5 atmosphere",
                                         "window_records": window,
                                         "source_record_split_bounds": record_bounds,
                                         "split_bounds": window_bounds,
                                         "original_shape": [20, 24, 721, 1440, 4]},
                bounds_override=window_bounds)


def _save(output: Path, fields: np.ndarray, masks: np.ndarray, metadata: dict,
          bounds_override: dict | None = None) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = bounds_override or split_bounds(len(fields))
    metadata.update({"processed_shape": list(fields.shape), "split_bounds": bounds,
                     "split_ratio": "8:1:1"})
    wet_fraction = np.ones(fields.shape[2:4], dtype=np.float32)
    with h5py.File(output, "w") as h5:
        for name, (start, end) in bounds.items():
            _write_group(h5, name, fields, masks, start, end, wet_fraction)
        h5.attrs["metadata_json"] = json.dumps(metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("A", "D"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Optionally limit the number of downsampled records")
    parser.add_argument("--record-start", type=int, default=0,
                        help="Start index in the downsampled time axis (for chunked PRE writes)")
    parser.add_argument("--record-count", type=int, default=None,
                        help="Number of records to write in this invocation")
    parser.add_argument("--append", action="store_true",
                        help="Append a PRE chunk to an initialized HDF5 output")
    parser.add_argument("--no-compression", action="store_true",
                        help="Disable LZF to speed up large PRE writes")
    parser.add_argument("--batch-records", type=int, default=16,
                        help="PRE records held in memory per Torch CPU reduction batch")
    parser.add_argument("--cpu-threads", type=int, default=128,
                        help="Torch CPU threads for vectorized preprocessing")
    parser.add_argument("--era-window", type=int, default=8,
                        help="ERA5 temporal window used as model depth")
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.task == "A":
        metadata = prepare_pre(args.source, args.output, args.max_records,
                               args.record_start, args.record_count, args.append,
                               None if args.no_compression else "lzf", args.batch_records)
    else:
        metadata = prepare_era5(args.source, args.output, args.max_records, args.era_window)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
