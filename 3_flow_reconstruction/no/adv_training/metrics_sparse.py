"""Mask-aware quality metrics for sparse vector-field reconstruction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


METRIC_NAMES = (
    "mae",
    "rmse",
    "bias",
    "relative_l2",
    "pearson_r",
    "nrmse",
    "psnr",
    "ssim",
)


def _broadcast_mask(mask, shape, name):
    if mask is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(mask, dtype=bool)
    if array.shape == shape:
        return array
    if array.ndim == len(shape) - 1 and array.shape == shape[1:]:
        array = array[None, ...]
    try:
        return np.broadcast_to(array, shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} shape {array.shape} is not broadcastable to {shape}") from exc


def _data_ranges(target, valid, data_range):
    channels = target.shape[0]
    if data_range is None:
        ranges = []
        for channel in range(channels):
            values = target[channel][valid[channel] & np.isfinite(target[channel])]
            span = float(values.max() - values.min()) if values.size else 1.0
            ranges.append(span if span > 0.0 else 1.0)
        return np.asarray(ranges, dtype=np.float64)
    ranges = np.asarray(data_range, dtype=np.float64)
    if ranges.ndim == 0:
        ranges = np.repeat(ranges, channels)
    if ranges.shape != (channels,) or np.any(~np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError(f"data_range must be a positive scalar or [{channels}] values")
    return ranges


def _masked_ssim(pred, target, valid, score, data_range, window_size):
    """Compute local SSIM, aggregating only valid window centers."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    height, width, depth = pred.shape
    del height, width
    values = []
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    for index in range(depth):
        p = pred[..., index]
        t = target[..., index]
        v = valid[..., index] & np.isfinite(p) & np.isfinite(t)
        if not v.any():
            continue
        p = np.where(v, p, 0.0).astype(np.float64, copy=False)
        t = np.where(v, t, 0.0).astype(np.float64, copy=False)
        v_float = v.astype(np.float64)
        support = uniform_filter(v_float, size=window_size, mode="nearest")
        denominator = np.maximum(support, np.finfo(np.float64).eps)
        mean_p = uniform_filter(p, size=window_size, mode="nearest") / denominator
        mean_t = uniform_filter(t, size=window_size, mode="nearest") / denominator
        var_p = np.maximum(uniform_filter(p * p, size=window_size, mode="nearest") / denominator - mean_p ** 2, 0.0)
        var_t = np.maximum(uniform_filter(t * t, size=window_size, mode="nearest") / denominator - mean_t ** 2, 0.0)
        cov = uniform_filter(p * t, size=window_size, mode="nearest") / denominator - mean_p * mean_t
        numerator = (2.0 * mean_p * mean_t + c1) * (2.0 * cov + c2)
        denominator_ssim = (mean_p ** 2 + mean_t ** 2 + c1) * (var_p + var_t + c2)
        local = numerator / np.maximum(denominator_ssim, np.finfo(np.float64).eps)
        centers = score[..., index] & v & (support >= 1.0 - 1e-6)
        if centers.any():
            values.append(local[centers])
    if not values:
        return float("nan")
    return float(np.concatenate(values).mean())


def masked_metrics(pred, target, valid=None, score_mask=None, data_range=None, ssim_window=7):
    """Return per-channel metrics for arrays shaped ``[C, H, W, D]``.

    ``valid`` identifies physical cells that may contribute to a metric.  The
    optional ``score_mask`` selects centers/voxels to score, e.g. only missing
    observations.  SSIM uses ``valid`` for its local windows and ``score_mask``
    only for window centers, so a 50% sparse mask does not destroy every SSIM
    window.
    """
    prediction = np.asarray(pred, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.ndim != 4:
        raise ValueError("pred and target must have the same [C, H, W, D] shape")
    physical = _broadcast_mask(valid, prediction.shape, "valid")
    score = _broadcast_mask(score_mask, prediction.shape, "score_mask")
    finite = np.isfinite(prediction) & np.isfinite(reference)
    physical &= finite
    score &= physical
    ranges = _data_ranges(reference, physical, data_range)

    output = {"count": [], "data_range": ranges.tolist()}
    for name in METRIC_NAMES:
        output[name] = []
    for channel in range(prediction.shape[0]):
        point_mask = score[channel]
        p = prediction[channel][point_mask]
        t = reference[channel][point_mask]
        count = int(p.size)
        output["count"].append(count)
        if count == 0:
            for name in METRIC_NAMES:
                output[name].append(float("nan"))
            continue
        error = p - t
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        target_norm = float(np.linalg.norm(t))
        target_std = float(np.std(t))
        if count > 1 and np.std(p) > 0.0 and target_std > 0.0:
            pearson = float(np.corrcoef(p, t)[0, 1])
        else:
            pearson = float("nan")
        output["mae"].append(float(np.mean(np.abs(error))))
        output["rmse"].append(rmse)
        output["bias"].append(float(np.mean(error)))
        output["relative_l2"].append(float(np.linalg.norm(error) / max(target_norm, np.finfo(float).eps)))
        output["pearson_r"].append(pearson)
        output["nrmse"].append(float(rmse / ranges[channel]))
        output["psnr"].append(float("inf") if mse == 0.0 else float(10.0 * np.log10(ranges[channel] ** 2 / mse)))
        output["ssim"].append(_masked_ssim(
            prediction[channel], reference[channel], physical[channel], score[channel],
            ranges[channel], ssim_window,
        ))
    return output


def vector_epe(pred, target, valid=None, score_mask=None):
    """Mean endpoint error for a two-component vector field."""
    prediction = np.asarray(pred, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.ndim != 4 or prediction.shape[0] != 2:
        raise ValueError("pred and target must have the same [2, H, W, D] shape")

    physical = _broadcast_mask(valid, prediction.shape, "valid")
    score = _broadcast_mask(score_mask, prediction.shape, "score_mask")
    point_mask = np.all(physical & score & np.isfinite(prediction) & np.isfinite(reference), axis=0)
    if not point_mask.any():
        return float("nan")
    error = prediction[:, point_mask] - reference[:, point_mask]
    return float(np.linalg.norm(error, axis=0).mean())


def macro_average(metrics):
    """Average per-channel metrics, ignoring NaN values."""
    result = {}
    for name in METRIC_NAMES:
        values = np.asarray(metrics[name], dtype=np.float64)
        result[name] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    result["count"] = int(np.nansum(metrics["count"]))
    return result
