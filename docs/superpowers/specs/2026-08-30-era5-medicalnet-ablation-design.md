# ERA5 MedicalNet Perceptual-Loss Ablation

## Goal

Measure whether the original repository's frozen MedicalNet perceptual loss
improves sparse reconstruction of ERA5 `u/v` wind fields relative to the
current mask-aware adv-NO baseline.

## Controlled Experiment

The baseline and ablation use the same prepared input file, chronological
split, random seed, model size, two-GPU data parallelism, and evaluation
procedure.

| setting | baseline | MedicalNet ablation |
| --- | --- | --- |
| data | `data/era_uv_2t4x_conservative.h5` | same |
| patch / depth | `64 x 64 x 8` | same |
| batch size / epochs | `32 / 500` | same |
| optimizer | Adam, generator LR `2e-4`, discriminator LR `1e-4` | same |
| train missing rate | uniformly sampled per item in `[0.1, 0.9]` | same |
| generator objective | normalized masked L1 + `0.1 * GAN BCE` | baseline objective + `1.0 * perceptual loss` |

The MedicalNet loss uses the original repository's frozen 3D ResNet-10
weights (`pretrain/resnet_10_23dataset.pth`) and intermediate-stage weights
`[0.1, 0.1, 1.0]`. It receives individually normalized `u` and `v` volumes,
as in the original implementation. The feature extractor is put in `eval()`
mode so that its medical-domain BatchNorm running statistics remain frozen.
This is a controlled, stable variant rather than a bit-for-bit replay of the
original script, which leaves the frozen extractor in training mode.

## Scope

Modify only the sparse training runner and its focused unit tests. Add an
optional, default-disabled perceptual-loss flag; a zero weight must preserve
the current baseline behavior. Keep the existing data adapter, `Unet3D`,
patch discriminator, metric evaluator, visualization scripts, baseline
checkpoint, and existing output directory unchanged.

The new model and logs are written only below:

`outputs/era_adv_no_medicalnet_bs32/`

## Evaluation

Run the existing GPU evaluator at fixed missing rates of 10%, 30%, 50%, 70%,
and 90%. Report missing-point MAE, RMSE, Relative L2, PSNR, and SSIM for the
baseline and ablation. Use the same seeded masks for both runs. Include a
50%-missing qualitative figure using the existing publication plotting
convention.

The ablation answers whether this cross-domain perceptual term helps ERA5;
it does not establish that MedicalNet is generally appropriate for PRE, where
land masking requires separate mask-aware feature-loss design.

## Validation and Rollback

Before the full run, unit tests must show that the optional loss is zero when
disabled, produces gradients for the prediction when enabled, keeps MedicalNet
parameters frozen, and preserves the expected 3D feature shapes for an ERA5
patch. A one-step GPU smoke run verifies the complete loss path and checkpoint
output before the 500-epoch run.

All code changes are optional and default to baseline behavior. Rollback is
removing the new optional loss path and the independent output directory;
neither action affects the existing ERA5 checkpoint or metrics.

## Risks

- MedicalNet is pretrained on medical images, so the added feature loss may
  degrade wind-field pointwise metrics or visual realism.
- Its additional 3D forward/backward path raises GPU memory use and step time.
  The smoke run determines whether batch size 32 remains viable; if it does
  not, the ablation's batch size change will be recorded explicitly and the
  comparison will be qualified.
- Relative loss scales can differ across domains. Perceptual, pixel, and GAN
  terms will be logged separately to make an unstable weighting visible.
