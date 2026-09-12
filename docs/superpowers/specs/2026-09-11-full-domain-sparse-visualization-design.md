# Full-Domain Sparse Reconstruction Visualization Design

> Status: implemented in `c524aaf`. The final figures use the complete
> processed domains and physical geographic coordinates described below.
>
> Presentation revision approved on 2026-09-12: remove the auxiliary subtitle
> and footer, make the error definition explicit, label both colorbar types in
> physical units, and reserve enough horizontal space for PRE colorbar ticks.

## Goal

Replace the current center-crop `64 x 64` PRE and ERA5 presentation figures
with full-domain figures that use the same tiled inference path as formal
evaluation and have unambiguous publication-quality labels.

## Scope

- PRE full horizontal domain: `100 x 110`, one selected sigma layer from the
  complete 30-layer processed field.
- ERA5 full horizontal domain: `180 x 360`, one selected time step from the
  complete 8-step window.
- Missing rates: 1%, 10%, 30%, 50%, 70%, 90%, and 99% for PRE; all formally
  reported ERA5 rates, including 99% for the extreme-sparsity model.
- Inference weights remain `generator_ema`; datasets, checkpoints, masks,
  metrics, and training code are unchanged.

## Inference

The plotting script will construct the sparse input on the complete held-out
sample and call the existing overlapping tiled predictor. Tiles use the model
patch dimensions stored in the checkpoint, overlap at the configured stride,
and are averaged in overlap regions. Plotting must not run the model directly
on a single center crop.

PRE land cells remain `NaN` in every displayed panel. They do not become
observations, predictions, or error values. ERA5 uses the supplied all-valid
mask.

## Figure Content

Each figure uses two rows and four data columns:

| Column | Title |
| --- | --- |
| 1 | **Ground Truth** |
| 2 | **Sparse Observations** |
| 3 | **adv-NO Reconstruction** |
| 4 | **Absolute Error (|Reconstruction - Ground Truth|)** |

Row labels are:

- PRE: **Eastward Velocity (u)** and **Northward Velocity (v)**.
- ERA5: **Zonal Wind Velocity (u)** and **Meridional Wind Velocity (v)**.

`Input`, `U/V Input`, and `U/V Recon` will not be used because they are
ambiguous or unnecessarily abbreviated.

## Titles And Metadata

- PRE title: **Full-Domain PRE Ocean Velocity Reconstruction from Sparse
  Observations**.
- ERA5 title: **Full-Domain ERA5 Wind Velocity Reconstruction from Sparse
  Observations**.
- No held-out-sample subtitle is displayed below the main title.
- No domain, geographic-extent, selected-dimension, missing-rate, observed-rate,
  checkpoint-weight, or scale-policy sentence is displayed above or below the
  panels. Those facts remain reproducible from the filename, command, and
  project documentation without competing with the data panels.

## Axes And Typography

- PRE axes: physical longitude/latitude from the source RHO curvilinear grid,
  aggregated with the same conservative `4 x 4` weights as the field. Both
  major-axis intervals are fixed at `0.5°` for readable, consistent spacing.
- ERA5 axes: physical longitude `0--360°E` and latitude `90°S--90°N`; the
  source descending array order `90°N -> 90°S` is preserved.
- Y-axis labels and ticks appear only in the leftmost column.
- X-axis labels and ticks appear only in the bottom row.
- Main title: bold, 18 pt.
- Column titles: bold, 14 pt.
- Row labels: bold, 12 pt.
- Axis labels: 11 pt; tick labels: 10 pt; colorbar labels: 10 pt.
- PRE panels preserve the grid aspect ratio. ERA5 panels preserve the global
  `2:1` horizontal aspect ratio rather than stretching each field to square.
- Colorbar tick labels face away from the adjacent error panel, and the
  GridSpec reserves explicit space around each colorbar so tick labels cannot
  overlap a neighboring panel.

## Colorbars

- Ground truth, sparse observations, and reconstruction share one symmetric
  velocity scale per component row.
- The field scale is derived from the selected ground truth, making it fixed
  across missing-rate figures for a given sample and slice.
- Absolute-error panels use a common non-negative scale for u and v within a
  dataset, with a fixed value supplied for all missing rates.
- Field colorbars use `Velocity (m s^-1)` and error colorbars use
  `Absolute error (m s^-1)`. The error-column heading defines the plotted
  quantity as `|Reconstruction - Ground Truth|`; row labels identify whether
  the values belong to the u or v component. White/blank PRE cells denote
  invalid land, not zero error.

## Repository Documentation

The preprocessing and training scripts remain tracked in Git. The main README
must not say that data files are generated locally or not committed, because
that wording can be mistaken for a statement about the scripts. It continues
to link the tracked preprocessing/training code and provide reproducible
commands; large generated artifacts remain governed by `.gitignore` without a
separate explanatory sentence in that data-shape paragraph.

## Outputs

The script continues to produce one PNG per missing rate in `docs/figures/`.
All figures for a dataset use identical pixel dimensions so they do not shift
when presented sequentially. Old center-patch figures are replaced, not kept
alongside the full-domain versions.

## Verification

1. Unit tests verify full-domain tiled inference, the explicit error title,
   colorbar unit labels, axis visibility, and fixed color scales.
2. Generated-image checks verify PRE output covers `100 x 110` and ERA5 covers
   `180 x 360` from plotting inputs; figures contain no auxiliary subtitle or
   footer text.
3. All output images for each dataset have identical dimensions.
4. Visual inspection is performed at 1%, 50%, and 99% missing rates for title
   clipping, text overlap, aspect ratio, land masking, colorbar labels, and
   colorbar tick clearance.
5. Existing sparse-training and metric tests remain green.
