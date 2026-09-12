# Metric Analysis Figures Implementation Plan

> Status: historical plan superseded by the final `0%--100%` PRE experiment and
> full-domain visualization. The path and rate examples below are synchronized
> with the retained final artifacts.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one reproducible plotting command that reads the retained PRE/ERA5 metric JSON files and produces three publication-style analysis figures as PNG and PDF previews.

**Architecture:** A focused plotting module validates two existing JSON schemas (single-run metric reports and multiseed summaries), converts them into ordered metric series, and renders three fixed-purpose figures through shared styling helpers. The command accepts repository and output roots so it remains reproducible without embedding metric values or machine-specific absolute paths.

**Tech Stack:** Python 3, standard-library `argparse/json/pathlib`, NumPy, Matplotlib

---

### Task 1: Metric JSON Contracts

**Files:**
- Create: `3_flow_reconstruction/no/adv_training/plot_metric_analysis.py`

- [ ] **Step 1: Verify the new module does not exist yet**

Run:

```bash
python -c "import sys; sys.path.insert(0, '3_flow_reconstruction/no/adv_training'); import plot_metric_analysis"
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Add constants and strict JSON loading helpers**

Create the module with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATES_STANDARD = (10, 30, 50, 70, 90)
RATES_PRE = (1, *RATES_STANDARD, 99)
METRICS = ("mae", "rmse", "relative_l2", "ssim")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required metric report not found: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Metric report must contain a JSON object: {path}")
    return value


def finite(value: object, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def load_single(path: Path, method: str = "model") -> dict[str, float]:
    report = read_json(path)
    try:
        region = report["methods"][method]["missing"]
        macro = region["macro"]
        values = {name: finite(macro[name], f"{path}:{method}:{name}") for name in METRICS}
    except KeyError as error:
        raise ValueError(f"Missing metric field {error} in {path}") from error
    return values


def load_summary(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    report = read_json(path)
    try:
        means = {name: finite(report["metrics"][name]["mean"], f"{path}:{name}:mean") for name in METRICS}
        stds = {name: finite(report["metrics"][name]["std"], f"{path}:{name}:std") for name in METRICS}
    except KeyError as error:
        raise ValueError(f"Missing summary field {error} in {path}") from error
    return means, stds
```

- [ ] **Step 3: Run inline contract checks**

Run:

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "3_flow_reconstruction/no/adv_training")
from plot_metric_analysis import load_single, load_summary

single = load_single(Path("outputs/pre_ragan_pretrained_0to100_ema_bs64/evaluation_final/metrics_50pct.json"))
mean, std = load_summary(Path("outputs/era_bce_pretrained_ema_bs32/evaluation_multiseed/summary_50pct.json"))
assert set(single) == {"mae", "rmse", "relative_l2", "ssim"}
assert abs(single["mae"] - 0.0026848558) < 1e-8
assert abs(mean["mae"] - 0.4205220302) < 1e-8
assert std["mae"] > 0
PY
```

Expected: PASS with exit code 0.

- [ ] **Step 4: Commit the loaders**

```bash
git add 3_flow_reconstruction/no/adv_training/plot_metric_analysis.py
git commit -m "feat: load sparse metric analysis data"
```

### Task 2: Shared Figure Styling and PRE Robustness Plot

**Files:**
- Modify: `3_flow_reconstruction/no/adv_training/plot_metric_analysis.py`

- [ ] **Step 1: Add an inline failing interface check**

Run:

```bash
python -c "import sys; sys.path.insert(0, '3_flow_reconstruction/no/adv_training'); from plot_metric_analysis import plot_pre_robustness"
```

Expected: FAIL with `ImportError`.

- [ ] **Step 2: Add series, style, and PRE plotting functions**

Add functions with these interfaces and behavior:

```python
def load_single_series(paths: list[Path], method: str = "model") -> dict[str, np.ndarray]:
    rows = [load_single(path, method=method) for path in paths]
    return {name: np.asarray([row[name] for row in rows], dtype=float) for name in METRICS}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
    })


def metric_label(name: str) -> str:
    return {
        "mae": r"MAE (m s$^{-1}$)",
        "rmse": r"RMSE (m s$^{-1}$)",
        "relative_l2": "Relative L2 error (%)",
        "ssim": "SSIM",
    }[name]


def scale_values(name: str, values: np.ndarray) -> np.ndarray:
    return values * 100.0 if name == "relative_l2" else values


def finish_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pre_robustness(repo: Path, output_stem: Path) -> None:
    paths = [
        repo / "outputs/pre_ragan_pretrained_0to100_ema_bs64/evaluation_final" /
        f"metrics_{rate}pct.json"
        for rate in RATES_PRE
    ]
    series = load_single_series(paths)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), sharex=True)
    for axis, name in zip(axes.flat, METRICS):
        values = scale_values(name, series[name])
        axis.plot(RATES_PRE, values, color="#0072B2", marker="o", linewidth=2, markersize=5)
        axis.axvspan(90, 100, color="#D55E00", alpha=0.08)
        axis.scatter([99], [values[-1]], color="#D55E00", marker="D", s=42, zorder=3,
                     label="99% stress test")
        if name in {"mae", "rmse", "relative_l2"}:
            axis.set_yscale("log")
        axis.set_ylabel(metric_label(name))
        axis.set_xticks(RATES_PRE)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    for axis in axes[-1]:
        axis.set_xlabel("Missing rate (%)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("PRE Sparse-Reconstruction Robustness Across Missing Rates", fontweight="bold")
    fig.text(0.5, 0.01, "Final pretrained RaGAN + EMA model; 99% missing is an extreme-sparsity stress test covered by Uniform(0.0, 1.0) training.", ha="center", color="#4B5563")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    finish_figure(fig, output_stem)
```

- [ ] **Step 3: Generate and validate the PRE preview**

Run the final CLI after Task 4 is present; until then call the function inline:

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "3_flow_reconstruction/no/adv_training")
from plot_metric_analysis import configure_style, plot_pre_robustness
configure_style()
plot_pre_robustness(Path.cwd(), Path("outputs/metric_analysis_preview/pre_missing_rate_analysis"))
PY
```

Expected: nonempty PNG and PDF files.

- [ ] **Step 4: Commit the PRE plot**

```bash
git add 3_flow_reconstruction/no/adv_training/plot_metric_analysis.py
git commit -m "feat: plot PRE missing-rate robustness"
```

### Task 3: ERA5 Strategy and Extreme-Sparsity Plots

**Files:**
- Modify: `3_flow_reconstruction/no/adv_training/plot_metric_analysis.py`

- [ ] **Step 1: Verify the ERA plotting interfaces are absent**

Run:

```bash
python -c "import sys; sys.path.insert(0, '3_flow_reconstruction/no/adv_training'); from plot_metric_analysis import plot_era_strategy, plot_era_extreme_tradeoff"
```

Expected: FAIL with `ImportError`.

- [ ] **Step 2: Implement multiseed loading and strategy plotting**

Add `load_summary_series(paths)` returning `means` and `stds` dictionaries of
NumPy arrays. Implement `plot_era_strategy(repo, output_stem)` with a 2 x 2
panel layout, the four run directories and labels below, one-standard-deviation
bands, and the same metric scaling/styling as PRE:

```python
strategies = (
    ("era_adv_no_random_bs32", "Direct BCE", "#999999", "o"),
    ("era_ragan_random_bs32", "Direct RaGAN", "#E69F00", "s"),
    ("era_bce_pretrained_ema_bs32", "Pretrained BCE + EMA", "#0072B2", "o"),
    ("era_ragan_pretrained_ema_bs32", "Pretrained RaGAN + EMA", "#009E73", "s"),
)
```

Use title `ERA5 Reconstruction Accuracy by Training Strategy`, a shared legend
above the panels, and footer `Bands show +/-1 sample SD across mask seeds 25,
42, and 2026.`

- [ ] **Step 3: Implement extreme-sparsity trade-off plotting**

Implement `plot_era_extreme_tradeoff(repo, output_stem)` with:

- standard pretrained BCE mean values through 90%, without a fabricated 99%
  point;
- seed-25 extreme-model, nearest, and Gaussian series through 99%;
- distinct model lines and muted dashed baseline lines;
- 99% vertical stress-test shading;
- title `ERA5 Standard-Range Accuracy vs Extreme-Sparsity Robustness`;
- footer stating `Standard BCE: three-seed mean (10%--90%); extreme model and
  interpolation baselines: seed 25 (10%--99%).`.

Load the three seed-25 curves by calling `load_single_series(extreme_paths,
method=...)` with `model`, `nearest`, and `gaussian`; do not introduce separate
hardcoded baseline values.

- [ ] **Step 4: Run inline numerical checks**

Run:

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "3_flow_reconstruction/no/adv_training")
from plot_metric_analysis import load_summary_series, load_single_series

root = Path.cwd()
summary_paths = [root / f"outputs/era_bce_pretrained_ema_bs32/evaluation_multiseed/summary_{r}pct.json" for r in (10, 30, 50, 70, 90)]
means, stds = load_summary_series(summary_paths)
assert len(means["mae"]) == 5 and len(stds["mae"]) == 5
assert means["mae"][0] < means["mae"][-1]
extreme_paths = [root / "outputs/era_ragan_pretrained_0to100_ema_bs32" / ("evaluation_99pct/metrics_99pct.json" if r == 99 else f"evaluation_all_rates/metrics_{r}pct.json") for r in (10, 30, 50, 70, 90, 99)]
extreme = load_single_series(extreme_paths)
assert abs(extreme["ssim"][-1] - 0.652226) < 1e-5
PY
```

Expected: PASS with exit code 0.

- [ ] **Step 5: Commit both ERA5 plots**

```bash
git add 3_flow_reconstruction/no/adv_training/plot_metric_analysis.py
git commit -m "feat: plot ERA5 metric trade-offs"
```

### Task 4: CLI, Output Verification, and Visual Review

**Files:**
- Modify: `3_flow_reconstruction/no/adv_training/plot_metric_analysis.py`

- [ ] **Step 1: Add the CLI**

Implement:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/metric_analysis_preview"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    configure_style()
    plot_pre_robustness(repo, output / "pre_missing_rate_analysis")
    plot_era_strategy(repo, output / "era5_training_strategy_analysis")
    plot_era_extreme_tradeoff(repo, output / "era5_extreme_sparsity_tradeoff")
    for path in sorted(output.glob("*")):
        if path.suffix in {".png", ".pdf"}:
            if path.stat().st_size == 0:
                raise RuntimeError(f"Empty plot output: {path}")
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile and generate all previews**

Run:

```bash
python -m py_compile 3_flow_reconstruction/no/adv_training/plot_metric_analysis.py
python 3_flow_reconstruction/no/adv_training/plot_metric_analysis.py
```

Expected: six `Wrote ...` lines for three PNG and three PDF files.

- [ ] **Step 3: Check dimensions and nonblank pixels**

Run:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageStat

paths = sorted(Path("outputs/metric_analysis_preview").glob("*.png"))
assert len(paths) == 3
for path in paths:
    image = Image.open(path).convert("RGB")
    assert image.width >= 2500 and image.height >= 1500
    extrema = ImageStat.Stat(image).extrema
    assert any(low < high for low, high in extrema)
    print(path, image.size, path.stat().st_size)
PY
```

Expected: three large, nonblank PNG images with nonzero byte sizes.

- [ ] **Step 4: Visually inspect all PNGs**

Use the local image viewer on each PNG. Confirm titles, axis labels, legends,
line colors, uncertainty bands, 99% annotations, footers, and tick labels do not
overlap or clip. Correct layout defects and regenerate all outputs.

- [ ] **Step 5: Run repository checks and commit the final script**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the plotting script is modified before
the final commit, while preview outputs remain ignored.

```bash
git add 3_flow_reconstruction/no/adv_training/plot_metric_analysis.py
git commit -m "feat: generate sparse metric analysis figures"
```

### Task 5: User Review and Optional Publication

**Files:**
- Preview: `outputs/metric_analysis_preview/*.png`
- Optional modify after approval: `docs/PROJECT_REPORT.md`
- Optional create after approval: `docs/figures/*_analysis.png`
- Optional create after approval: `docs/figures/*_analysis.pdf`

- [ ] **Step 1: Present the three preview images with exact local links**

Report what each figure demonstrates and disclose the seed coverage difference.

- [ ] **Step 2: Wait for visual approval**

Do not move previews into tracked documentation before approval.

- [ ] **Step 3: Publish only after approval**

Copy the approved PNG/PDF artifacts into `docs/figures/`, add an analysis-figure
section to `docs/PROJECT_REPORT.md`, run `git diff --check`, commit, and push only
when the user explicitly asks for publication.
