# Full-Domain Sparse Reconstruction Visualization Design

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
| 4 | **Absolute Error** |

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
- Subtitle records the held-out sample, missing rate, observed fraction, and
  selected dimension position.
- PRE uses `Sigma layer <n> of 30`; ERA5 uses `Time step <n> of 8`.

## Axes And Typography

- PRE axes: `xi-grid index` and `eta-grid index`, because the processed HDF5
  does not store geographic coordinates.
- ERA5 axes: `Longitude index` and `Latitude index` unless geographic
  coordinates are explicitly added later.
- Y-axis labels and ticks appear only in the leftmost column.
- X-axis labels and ticks appear only in the bottom row.
- Main title: bold, 18 pt.
- Column titles: bold, 14 pt.
- Row labels: bold, 12 pt.
- Axis labels: 11 pt; tick labels: 10 pt; colorbar labels: 10 pt.
- PRE panels preserve the grid aspect ratio. ERA5 panels preserve the global
  `2:1` horizontal aspect ratio rather than stretching each field to square.

## Colorbars

- Ground truth, sparse observations, and reconstruction share one symmetric
  velocity scale per component row.
- The field scale is derived from the selected ground truth, making it fixed
  across missing-rate figures for a given sample and slice.
- Absolute-error panels use a common non-negative scale for u and v within a
  dataset, with a fixed value supplied for all missing rates.
- Colorbars state physical units (`m s^-1`). White/blank PRE cells denote
  invalid land, not zero error.

## Outputs

The script continues to produce one PNG per missing rate in `docs/figures/`.
All figures for a dataset use identical pixel dimensions so they do not shift
when presented sequentially. Old center-patch figures are replaced, not kept
alongside the full-domain versions.

## Verification

1. Unit tests verify the script selects full-domain tiled inference, dimension
   captions, column/row labels, axis visibility, and fixed color scales.
2. Generated-image checks verify PRE output covers `100 x 110` and ERA5 covers
   `180 x 360`, based on subtitle metadata and plotting inputs.
3. All output images for each dataset have identical dimensions.
4. Visual inspection is performed at 1%, 50%, and 99% missing rates for title
   clipping, text overlap, aspect ratio, land masking, and colorbar readability.
5. Existing sparse-training and metric tests remain green.
