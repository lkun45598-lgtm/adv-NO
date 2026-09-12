# Full-Domain Sparse Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace center-patch PRE and ERA5 reconstruction figures with correctly labeled, publication-ready full-domain figures generated through tiled inference.

> Status: implemented in `c524aaf`; the documentation and figure checks below
> describe the final physical-coordinate output.

**Architecture:** Keep `evaluate_sparse_metrics.tiled_predict` as the single tiled-inference implementation and call it from the plotting module. Split plotting into testable data preparation, layout, axis-visibility, and rendering helpers; retain the existing CLI while adding explicit tile and figure controls.

**Tech Stack:** Python 3.10, PyTorch, h5py, NumPy, Matplotlib, pytest, Pillow.

---

### Task 1: Specify Full-Domain Prediction Behavior

**Files:**
- Modify: `tests/test_sparse_visualization_layout.py`
- Modify: `3_flow_reconstruction/no/adv_training/plot_sparse_visualization.py`

- [x] **Step 1: Write the failing prediction test**

Add a test that provides a synthetic `[1, 2, 100, 110, 30]` field and checks the plotting prediction helper returns the same full horizontal/depth shape rather than `64 x 64 x 16`.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest -q tests/test_sparse_visualization_layout.py
```

Expected: failure because the current helper crops to the center patch.

- [x] **Step 3: Implement full-domain prediction**

Load the complete selected test record, construct its complete sparse input,
then call:

```python
prediction = tiled_predict(
    model,
    model_input,
    patch=args.patch,
    depth=args.depth,
    stride=args.stride,
    tile_batch=args.tile_batch,
)
```

Return full `target`, `prediction`, `observed`, and `valid` arrays. Do not
compute `y0`, `x0`, or `z0` center-crop offsets.

- [x] **Step 4: Run the prediction tests and verify GREEN**

Run the same pytest command and expect all prediction tests to pass.

### Task 2: Implement Publication Layout

**Files:**
- Modify: `tests/test_sparse_visualization_layout.py`
- Modify: `3_flow_reconstruction/no/adv_training/plot_sparse_visualization.py`

- [x] **Step 1: Write failing label and visibility tests**

Assert the column titles are exactly:

```python
("Ground Truth", "Sparse Observations", "adv-NO Reconstruction", "Absolute Error")
```

Assert only the leftmost column displays y tick labels and only the bottom row
displays x tick labels.

- [x] **Step 2: Run the tests and verify RED**

Expected: failure because current titles use `Reference field` and all panels
repeat both axes.

- [x] **Step 3: Implement the GridSpec layout**

Build a two-row layout with three field panels, one narrow shared field
colorbar, one error panel, and one narrow error colorbar per row. Apply bold
14 pt column titles, bold 12 pt row labels, 11 pt axes, and 10 pt ticks/colorbar
labels. Hide interior axes labels/ticks while preserving panel frames.

- [x] **Step 4: Use full-domain metadata**

Replace `central H x W tile` with `full H x W domain`. Format PRE captions as
`Sigma layer n of 30` and ERA5 captions as `Time step n of 8`.

- [x] **Step 5: Run layout tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/test_sparse_visualization_layout.py
```

Expected: all tests pass.

### Task 3: Render PRE Full-Domain Figures

**Files:**
- Modify: `docs/figures/pre_1pct_visualization.png`
- Modify: `docs/figures/pre_10pct_visualization.png`
- Modify: `docs/figures/pre_30pct_visualization.png`
- Modify: `docs/figures/pre_50pct_visualization.png`
- Modify: `docs/figures/pre_70pct_visualization.png`
- Modify: `docs/figures/pre_90pct_visualization.png`
- Modify: `docs/figures/pre_99pct_visualization.png`

- [x] **Step 1: Run GPU rendering for all PRE rates**

Use the final `pre_ragan_pretrained_0to100_ema_bs64` EMA checkpoint, sample 0,
sigma layer 15 of 30, and rates `0.01 0.10 0.30 0.50 0.70 0.90 0.99`.

- [x] **Step 2: Verify output dimensions and visible domain**

Use Pillow to assert all seven files have identical dimensions. Check the
subtitle states `full 100 x 110 domain` and `Sigma layer 15 of 30`.

- [x] **Step 3: Inspect 1%, 50%, and 99% images**

Check title/axis/colorbar clipping, PRE land masking, sparse observations,
panel order, and error visibility.

### Task 4: Render ERA5 Full-Domain Figures

**Files:**
- Modify: `docs/figures/era5_10pct_visualization.png`
- Modify: `docs/figures/era5_30pct_visualization.png`
- Modify: `docs/figures/era5_50pct_visualization.png`
- Modify: `docs/figures/era5_70pct_visualization.png`
- Modify: `docs/figures/era5_90pct_visualization.png`
- Modify: `docs/figures/era5_99pct_visualization.png`

- [x] **Step 1: Use one checkpoint for the degradation series**

Use `era_ragan_pretrained_0to100_ema_bs32` for every 10%--99% figure so the
series shows one model's degradation curve and remains consistent with the
project report's "ERA5 extreme-sparsity model" visualization section. Do not
mix the standard BCE checkpoint and extreme-sparsity checkpoint under one set
of filenames.

- [x] **Step 2: Run GPU rendering**

Render sample 0, time step 5 of 8, over the full `180 x 360` domain. Preserve
the global `2:1` field aspect ratio.

- [x] **Step 3: Verify dimensions and visual quality**

Assert all ERA5 files share one pixel size and inspect 10%, 50%, and 99% for
layout, readable font size, color scales, and complete-domain coverage.

### Task 5: Documentation And Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_REPORT.md`
- Modify: `docs/results/pre_ragan_pretrained_0to100_ema_bs64/README.md`
- Modify: `3_flow_reconstruction/no/adv_training/README_sparse_tasks.md`

- [x] **Step 1: Update documentation**

Replace references to center `64 x 64` figures with PRE `100 x 110` and ERA5
`180 x 360` full-domain figures. Document the four column titles, physical
longitude/latitude axes, selected layer/time step, colorbar policy, and
checkpoint provenance.

- [x] **Step 2: Run automated verification**

Run:

```bash
git diff --check
python -m pytest -q tests
```

Expected: no whitespace errors and all tests pass.

- [x] **Step 3: Verify generated assets**

Use Pillow/file inspection to verify all expected PNG files are nonempty and
have consistent per-dataset dimensions. Confirm no HDF5, checkpoint, or log is
staged.

- [x] **Step 4: Commit and push**

Commit code/tests/figures/docs together with a message describing full-domain
visualization, then push `main` to the configured `submission` remote and
verify its commit hash.
