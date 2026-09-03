# Current Task README Design

## Goal

Replace the upstream-first root README with a Chinese-first entry point for the
completed PRE and ERA5 sparse-reconstruction work, while preserving upstream
paper attribution, citation, and licensing.

## Audience And Scope

The primary audience is a reviewer reproducing Task A or Task D. The README
must describe only implemented behavior and checked-in artifacts. Detailed
metric definitions and full tables remain in `docs/PROJECT_REPORT.md`.

## Information Architecture

1. Project title and one-paragraph summary.
2. Task A/D status table and explicit statement that B/C are out of scope.
3. Data-processing contract: 2x temporal averaging, conservative 4x4 spatial
   averaging, chronological 8:1:1 split, and PRE land masking.
4. Model contract: four input channels, two output channels, two-stage NO/GAN
   training, random training missing rates, and EMA evaluation.
5. Repository map for preprocessing, training, evaluation, metrics, plotting,
   the full report, and final figures.
6. Environment, data preparation, training, evaluation, and visualization
   commands using repository-relative paths.
7. Compact final-result tables that distinguish the standard ERA5 model from
   the 0%-100% extreme-sparsity experiment and do not overstate 99% results.
8. Data/checkpoint publication policy.
9. Upstream Gen4Turbulence attribution, BibTeX citation, and MIT license.

## Accuracy Rules

- Use “missing rate” and “observation rate” consistently.
- State that metrics are computed on valid missing locations.
- State that PRE land is excluded from loss, metrics, and visualization.
- Identify `generator_ema` as the evaluation weights.
- Do not describe 99% missing reconstruction as high accuracy.
- Do not claim that datasets or new checkpoints are stored in GitHub.

## Verification

- Every local Markdown link must resolve in the repository.
- Every documented CLI flag must exist in the corresponding `--help` output.
- Reported headline values must match `docs/PROJECT_REPORT.md`.
- The README must contain no user-specific server path, credential, test, or
  smoke-run instructions.
