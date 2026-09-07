# Metric Analysis Figures Design

## Goal

Generate three standalone, publication-style analysis figures from the existing
PRE and ERA5 metric JSON files. The figures explain robustness and training
strategy trade-offs; they do not repeat the existing field-reconstruction
visualizations and do not require new training or inference.

## Outputs

Initial review files are written under the ignored directory
`outputs/metric_analysis_preview/` as 300 DPI PNG and vector PDF files:

1. `pre_missing_rate_analysis`: PRE final-model degradation across 10%, 30%,
   50%, 70%, 90%, and 99% missing rates.
2. `era5_training_strategy_analysis`: Direct BCE, Direct RaGAN, pretrained BCE
   + EMA, and pretrained RaGAN + EMA across 10%--90% missing rates.
3. `era5_extreme_sparsity_tradeoff`: standard-range pretrained BCE + EMA versus
   0%--100% pretrained RaGAN + EMA, with nearest and Gaussian baselines included
   where they clarify the 99% stress test.

After visual approval, final files will be copied into `docs/figures/` and linked
from `docs/PROJECT_REPORT.md` in a separate change. Before approval, only the
ignored preview files are created.

## Data Sources

- PRE: `outputs/pre_ragan_pretrained_ema_bs64/evaluation_final/metrics_*.json`.
  These are complete-test, seed-25 results.
- ERA5 standard strategy comparison: `evaluation_multiseed/summary_*.json` from
  the four direct/pretrained BCE/RaGAN run directories. Curves use the reported
  three-seed mean and show one sample standard deviation.
- ERA5 extreme model: `evaluation_all_rates/metrics_*.json` and
  `evaluation_99pct/metrics_99pct.json` from
  `era_ragan_pretrained_0to100_ema_bs32`. These are complete-test, seed-25
  results.
- ERA5 standard reference in the trade-off plot: the three-seed summary files
  from `era_bce_pretrained_ema_bs32`.

The plotting code reads values from JSON and must not duplicate metric numbers
as hardcoded arrays.

## Figure Content

Each figure uses four panels: MAE, RMSE, Relative L2, and SSIM. MAE/RMSE units
are `m s^-1`; Relative L2 is displayed as percent; SSIM is dimensionless.
Missing rate is the x-axis in percent.

### PRE Robustness

- Plot only the final pretrained RaGAN + EMA model.
- Use logarithmic y-scales for error panels when necessary so the valid
  10%--90% trend remains readable despite the 99% outlier.
- Mark 99% as an out-of-distribution stress test because PRE was trained on
  10%--90% missing rates.

### ERA5 Training Strategy

- Plot four consistently colored and marked strategy curves.
- Show mean values with one-standard-deviation uncertainty bands for all four
  strategies.
- Keep the title focused on initialization/adversarial-strategy comparison;
  do not imply that pretraining and EMA effects have been separately isolated.

### ERA5 Extreme-Sparsity Trade-off

- Plot the standard-range pretrained BCE + EMA curve through 90%.
- Plot the 0%--100% pretrained RaGAN + EMA curve through 99%.
- Include nearest and Gaussian interpolation baselines using the same seed-25
  files as the extreme model.
- Visually identify 99% as a stress-test point and state that the standard and
  extreme curves have different seed coverage in a compact footer.

## Visual Rules

- White background, restrained color-blind-safe palette, solid gridlines only
  on the y-axis, and no gradients or decorative elements.
- English professional titles and axis labels, consistent with existing final
  figures.
- Shared legend above or below panels; labels must not overlap plots.
- Stable figure dimensions, 300 DPI PNG output, and vector PDF output.
- No PRE/ERA5 physical-error values on a shared axis because their physical
  scales differ.

## Validation

- Fail clearly when a required file, method, region, metric, mean, or standard
  deviation is missing.
- Assert that expected missing-rate sequences are complete and monotonic.
- Confirm all plotted values are finite and all output files are nonempty.
- Run the plot script from the repository root and visually inspect all three
  PNG files for clipping, overlap, readable legends, and correct stress-test
  annotations.

## Non-Goals

- No retraining, new inference, or metric-formula changes.
- No training-loss curves because complete comparable epoch histories are not
  part of the retained final artifacts.
- No modification of the existing reconstruction visualizations during the
  preview stage.
