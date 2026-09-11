# PRE/ERA5 sparse reconstruction

This directory contains the dataset adapter and runner used for the two
smallest requested tasks:

- **Task A (PRE):** `u_eastward` and `v_northward`, which are the eastward and
  northward `u/v` components already interpolated to the common RHO grid.
- **Task D (ERA5):** channels 1 and 2 (`u_component_of_wind` and
  `v_component_of_wind`).

`prepare_sparse_data.py` applies temporal factor 2 and horizontal factor 4
downsampling using paired-time averaging and conservative 4x4 area-weighted
block averaging, then makes chronological train/validation/test splits of
8:1:1. This is the preprocessing used by the formal `*_conservative.h5`
results; it is not bicubic interpolation or point decimation. PRE's `mask_rho`
is combined with finite-value checks; land points are stored as zero but have
`valid_mask=0`.
ERA5 is represented as 8 consecutive downsampled hours on the depth axis so it
can use the repository's 3D adv-NO U-Net. ERA5 windows are created separately
inside each chronological split, so no window contains records from another
split.

The formal conservative files currently contain:

| task | field shape | train / val / test | valid fraction |
| --- | --- | --- | --- |
| A | `[5295, 2, 100, 110, 30]` | `4236 / 529 / 530` | `0.73373` |
| D | `[219, 2, 180, 360, 8]` | `185 / 17 / 17` | `1.0` |

The formal training-split statistics are recorded in each runner's
`config.json`; the ERA5 run uses `mean=(2.98951411, -0.01058338)` and
`std=(11.58474255, 4.26915359)` for `(u, v)` respectively.

## Prepare data

```bash
python prepare_sparse_data.py --task A \
  --source /data/PRE_ocean_data --output ../../../data/pre_uv_2t4x_conservative.h5
python prepare_sparse_data.py --task D \
  --source /data/era_data/raw_data --output ../../../data/era_uv_2t4x_conservative.h5
```

## Two-stage training

The paper-sized defaults are `dim=16`, `lr=2e-4`, `batch-size=8`, and
`epochs=500`. The formal PRE run uses batch size 64; ERA5 uses batch size 32.
Both tasks first train an L1-only NO and then initialize GAN fine-tuning from
the best EMA generator.

### PRE final model (0%--100% random missing-rate training)

```bash
# Stage 1: L1-only NO pretraining + EMA.
python train_sparse_adv_no.py --data ../../../data/pre_uv_2t4x_conservative.h5 \
  --output-dir ../../../outputs/pre_no_pretrain_0to100_ema_bs64 --patch 64 --depth 16 \
  --batch-size 64 --epochs 500 --cpu-threads 1 \
  --train-mask-min 0.0 --train-mask-max 1.0 \
  --adversarial-weight 0 --ema-decay 0.999

# Stage 2: RaGAN fine-tuning from the Stage-1 EMA generator.
python train_sparse_adv_no.py --data ../../../data/pre_uv_2t4x_conservative.h5 \
  --output-dir ../../../outputs/pre_ragan_pretrained_0to100_ema_bs64 --patch 64 --depth 16 \
  --batch-size 64 --epochs 500 --cpu-threads 1 \
  --train-mask-min 0.0 --train-mask-max 1.0 \
  --init-generator ../../../outputs/pre_no_pretrain_0to100_ema_bs64/best_model.pt \
  --adversarial-loss ragan --adversarial-weight 0.1 --ema-decay 0.999
```

The completed PRE checkpoints reached best validation/test normalized L1 of
`0.01642772`/`0.01605174` after RaGAN fine-tuning.  The final multi-rate
JSON reports and paper-ready figures are published in
[`docs/results/pre_ragan_pretrained_0to100_ema_bs64/`](../../../docs/results/pre_ragan_pretrained_0to100_ema_bs64)
and [`docs/figures/`](../../../docs/figures).  The large HDF5 files,
checkpoints and logs remain local and are intentionally excluded from Git.

### ERA5 standard-range comparison

The repository GAN script first initializes its generator from a separately
trained NO.  Reproduce that sequence for ERA5 with a dataset-matched
pretraining checkpoint; do **not** use the repository's turbulent-flow
`best_model.pt`, whose channel layout differs from ERA5 `u/v`.

```bash
# Stage 1: L1-only sparse reconstruction pretraining.
python train_sparse_adv_no.py --data ../../../data/era_uv_2t4x_conservative.h5 \
  --output-dir ../../../outputs/era_no_pretrain_bs32 --patch 64 --depth 8 \
  --batch-size 32 --epochs 500 --train-mask-min 0.1 --train-mask-max 0.9 \
  --adversarial-weight 0 --ema-decay 0.999

# Stage 2a: BCE-GAN fine-tuning from the same pretraining checkpoint.
python train_sparse_adv_no.py --data ../../../data/era_uv_2t4x_conservative.h5 \
  --output-dir ../../../outputs/era_bce_pretrained_ema_bs32 --patch 64 --depth 8 \
  --batch-size 32 --epochs 500 --train-mask-min 0.1 --train-mask-max 0.9 \
  --init-generator ../../../outputs/era_no_pretrain_bs32/best_model.pt \
  --adversarial-loss bce --adversarial-weight 0.1 --ema-decay 0.999

# Stage 2b: RaGAN fine-tuning from that identical checkpoint.
python train_sparse_adv_no.py --data ../../../data/era_uv_2t4x_conservative.h5 \
  --output-dir ../../../outputs/era_ragan_pretrained_ema_bs32 --patch 64 --depth 8 \
  --batch-size 32 --epochs 500 --train-mask-min 0.1 --train-mask-max 0.9 \
  --init-generator ../../../outputs/era_no_pretrain_bs32/best_model.pt \
  --adversarial-loss ragan --adversarial-weight 0.1 --ema-decay 0.999
```

### ERA5 extreme-sparsity model

Use the same two stages with a missing-rate distribution spanning `[0, 1]`:

```bash
# Stage 1: L1-only NO pretraining.
python train_sparse_adv_no.py --data ../../../data/era_uv_2t4x_conservative.h5 \
  --output-dir ../../../outputs/era_no_pretrain_0to100_ema_bs32 --patch 64 --depth 8 \
  --batch-size 32 --epochs 500 --train-mask-min 0.0 --train-mask-max 1.0 \
  --adversarial-weight 0 --ema-decay 0.999

# Stage 2: RaGAN fine-tuning.
python train_sparse_adv_no.py --data ../../../data/era_uv_2t4x_conservative.h5 \
  --output-dir ../../../outputs/era_ragan_pretrained_0to100_ema_bs32 --patch 64 --depth 8 \
  --batch-size 32 --epochs 500 --train-mask-min 0.0 --train-mask-max 1.0 \
  --init-generator ../../../outputs/era_no_pretrain_0to100_ema_bs32/best_model.pt \
  --adversarial-loss ragan --adversarial-weight 0.1 --ema-decay 0.999
```

With the default nonzero EMA decay, every checkpoint retains both `generator`
and `generator_ema`.  If EMA is disabled with `--ema-decay 0`, checkpoints
contain only `generator`. Metric and visualization scripts select
`generator_ema` by default when it is available and record that choice in
metric metadata. Pass `--raw-generator` only for an explicit raw-versus-EMA
ablation.

The generator input has four channels: mean-filled sparse `u/v` values followed
by the shared observation mask repeated for `u` and `v`.  Missing and PRE-land
values are filled with the corresponding training mean, which becomes zero
after normalization.  Each output component is z-scored independently using
training-split statistics only; by default statistics use all training records.
Pixel loss is averaged
over valid voxels and both components.  The discriminator has no spatial
normalization and its GAN BCE uses only logits whose complete convolutional
receptive field is valid, so PRE land points and their neighboring invalid
logits do not contribute.  ERA5 has no
land mask in the supplied tensor, so its global land values are intentionally
treated as valid; an ocean-only ERA5 experiment requires an external mask.

The original repository's optional MedicalNet perceptual feature extractor is
present in the checkout, but it is not enabled here: its convolution/BatchNorm
feature statistics are not mask-aware and would let PRE land values influence a
content loss.  This runner therefore uses the same Unet3D generator with a
compact 3D patch discriminator and normalized pixel + GAN objectives, both of
which have explicit valid-region handling.

## Full-domain visualization

The `64 x 64` dimensions above are training and model-forward patch sizes, not
the published field extent. `plot_sparse_visualization.py` applies the sparse
mask to a complete held-out sample and uses overlapping tiled inference to
reconstruct the full processed domain. The paper-ready figures show PRE at
`100 x 110` on sigma layer 15 of 30 and ERA5 at `180 x 360` on time step 5 of
8. PRE land remains blank. The four columns are `Ground Truth`, `Sparse
Observations`, `adv-NO Reconstruction`, and `Absolute Error`; field scales are
shared by the first three columns within each row. PRE and ERA5 use fixed
cross-rate absolute-error limits of `0.12 m s^-1` and `10 m s^-1`,
respectively. ERA5's visualization series uses only the extreme-sparsity
`era_ragan_pretrained_0to100_ema_bs32` EMA checkpoint.

## ERA5 result summary

The completed formal runs are:

| run | best val normalized L1 | test normalized L1 at 50% missing | evaluation weights |
| --- | ---: | ---: | --- |
| `era_no_pretrain_bs32` (pure NO) | 0.07577464 | 0.07910468 | `generator_ema` |
| `era_bce_pretrained_ema_bs32` | **0.04897056** | **0.04833671** | `generator_ema` |
| `era_ragan_pretrained_ema_bs32` | 0.04950763 | 0.04885580 | `generator_ema` |

Compared with the existing random-initialization direct runs, the staged
pretraining + EMA workflow lowers the 50% test normalized L1 by about 14.81%
for BCE and 22.19% for RaGAN. After pretraining, BCE and RaGAN are close; the
five-rate missing-region evaluation reports an average MAE difference of about
1.46%.

The additional `0%--100%` missing-rate RaGAN reaches `0.05674879` best
validation L1 and `0.06660492` test L1 at 50% missing. It improves robustness
at 90%--99% missing while trading away some accuracy at lower missing rates.
For exact multi-rate metrics and PRE results, see
[`docs/PROJECT_REPORT.md`](../../../docs/PROJECT_REPORT.md).
