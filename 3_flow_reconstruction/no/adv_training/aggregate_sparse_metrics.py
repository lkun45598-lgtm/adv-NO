"""Aggregate deterministic sparse-reconstruction metric reports by mask seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate_reports(reports, method: str = "model", region: str = "missing"):
    """Return mean and sample standard deviation for matching metric reports."""
    if len(reports) < 2:
        raise ValueError("at least two reports are required to compute a sample standard deviation")

    values_by_metric = {}
    expected_metrics = None
    for index, report in enumerate(reports):
        try:
            section = report["methods"][method][region]
            metrics = dict(section["macro"])
            metrics.pop("count", None)
            metrics["epe"] = section["epe"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"report {index} lacks methods.{method}.{region} metrics") from exc

        metric_names = set(metrics)
        if expected_metrics is None:
            expected_metrics = metric_names
            values_by_metric = {name: [] for name in metric_names}
        elif metric_names != expected_metrics:
            raise ValueError(f"report {index} metrics do not match the first report")

        for name, value in metrics.items():
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"report {index} has non-finite {name}")
            values_by_metric[name].append(scalar)

    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "count": len(values),
        }
        for name, values in sorted(values_by_metric.items())
    }


def _read_reports(paths: list[Path]):
    reports = []
    seeds = []
    mask_ratios = []
    for path in paths:
        report = json.loads(path.read_text())
        metadata = report.get("metadata", {})
        if "seed" not in metadata or "mask_ratio" not in metadata:
            raise ValueError(f"{path} lacks metadata.seed or metadata.mask_ratio")
        seeds.append(int(metadata["seed"]))
        mask_ratios.append(float(metadata["mask_ratio"]))
        reports.append(report)
    if len(seeds) != len(set(seeds)):
        raise ValueError("input reports contain duplicate mask seeds")
    if not np.allclose(mask_ratios, mask_ratios[0], rtol=0.0, atol=1e-12):
        raise ValueError("input reports have different missing rates")
    return reports, seeds, mask_ratios[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="one JSON report per mask seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="model")
    parser.add_argument("--region", default="missing")
    parser.add_argument("--expected-count", type=int, default=3)
    args = parser.parse_args()

    if len(args.input) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} reports, received {len(args.input)}")
    reports, seeds, mask_ratio = _read_reports(args.input)
    summary = {
        "metadata": {
            "dataset": args.dataset,
            "mask_ratio": mask_ratio,
            "seeds": seeds,
            "method": args.method,
            "region": args.region,
            "source_reports": [str(path) for path in args.input],
        },
        "metrics": aggregate_reports(reports, method=args.method, region=args.region),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
