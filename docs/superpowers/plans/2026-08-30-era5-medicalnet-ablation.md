# ERA5 MedicalNet Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional frozen MedicalNet perceptual loss to the sparse adv-NO runner and train a 500-epoch ERA5 ablation against the existing baseline.

**Architecture:** Keep `Unet3D`, sparse masking, masked L1, and patch GAN unchanged. Add a small loss helper that reshapes the two output channels into independent single-channel 3D volumes, extracts frozen MedicalNet intermediate features, and returns the weighted L1 feature distance. The CLI weight defaults to zero, preserving baseline behavior; the ablation passes weight one and writes to a separate output directory.

**Tech Stack:** Python 3, PyTorch, MedicalNet ResNet-10, pytest, HDF5, tmux, CUDA DataParallel.

---

### Task 1: Add failing tests for the optional perceptual path

**Files:**
- Modify: `Gen4Turbulence/3_flow_reconstruction/no/adv_training/test_train_sparse_adv_no.py`
- Test target: `Gen4Turbulence/3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py`

- [ ] **Step 1: Write tests for disabled/enabled behavior and frozen extractor.**

Add imports for `perceptual_l1` and `build_perceptual_extractor`, then add:

```python
def test_perceptual_l1_is_zero_when_weight_is_zero():
    pred = torch.randn(1, 2, 16, 16, 8, requires_grad=True)
    target = torch.randn_like(pred)
    assert perceptual_l1(pred, target, extractor=None, weight=0.0).item() == 0.0


def test_medicalnet_extractor_is_frozen_and_eval():
    extractor = build_perceptual_extractor(
        mean=torch.tensor([0.0, 0.0]), std=torch.tensor([1.0, 1.0]),
        weight_path=Path(__file__).parent / "pretrain" / "resnet_10_23dataset.pth",
    )
    assert not extractor.training
    assert all(not parameter.requires_grad for parameter in extractor.parameters())


def test_perceptual_l1_backpropagates_to_prediction():
    extractor = build_perceptual_extractor(
        mean=torch.tensor([0.0, 0.0]), std=torch.tensor([1.0, 1.0]),
        weight_path=Path(__file__).parent / "pretrain" / "resnet_10_23dataset.pth",
    )
    pred = torch.randn(1, 2, 16, 16, 8, requires_grad=True)
    target = torch.randn_like(pred)
    loss = perceptual_l1(pred, target, extractor=extractor, weight=1.0)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
```

- [ ] **Step 2: Run the focused tests and verify the expected red failure.**

Run from `Gen4Turbulence/3_flow_reconstruction/no/adv_training`:

```bash
pytest -q test_train_sparse_adv_no.py -k perceptual
```

Expected: collection fails because the new helper names are not yet defined in `train_sparse_adv_no.py`.

### Task 2: Implement the frozen MedicalNet loss and CLI wiring

**Files:**
- Modify: `Gen4Turbulence/3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py`
- Test: `Gen4Turbulence/3_flow_reconstruction/no/adv_training/test_train_sparse_adv_no.py`

- [ ] **Step 1: Add the feature extractor factory and loss helper.**

Import `FeatureExtractor` from the local `model.py`, then add:

```python
def build_perceptual_extractor(mean, std, weight_path):
    par = {
        "out_shift": mean.detach().cpu().reshape(-1, 1, 1, 1, 1),
        "out_scale": std.detach().cpu().reshape(-1, 1, 1, 1, 1),
    }
    extractor = FeatureExtractor(par, weight_path=str(weight_path), layers=[1, 2, 3])
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor


def perceptual_l1(pred, target, extractor, weight=0.0):
    if weight == 0.0:
        return pred.new_zeros(())
    if extractor is None:
        raise ValueError("extractor is required when perceptual weight is nonzero")
    pred_features = extractor(pred)
    with torch.no_grad():
        target_features = extractor(target)
    stage_weights = (0.1, 0.1, 1.0)
    return float(weight) * sum(
        F.l1_loss(pred_feature, target_feature)
        * stage_weight
        for pred_feature, target_feature, stage_weight
        in zip(pred_features, target_features, stage_weights)
    )
```

`FeatureExtractor` already normalizes each channel and reshapes channels to independent one-channel 3D volumes. The helper preserves that original behavior and allows gradients only through the predicted features.

- [ ] **Step 2: Add CLI options and initialize the extractor only when enabled.**

Add these parser arguments beside the existing loss/model options:

```python
parser.add_argument("--perceptual-weight", type=float, default=0.0)
parser.add_argument(
    "--perceptual-weight-path",
    type=Path,
    default=HERE / "pretrain" / "resnet_10_23dataset.pth",
)
```

Validate `perceptual_weight >= 0`. After `mean_t` and `std_t` are created, set:

```python
perceptual_extractor = None
if args.perceptual_weight > 0.0:
    perceptual_extractor = build_perceptual_extractor(
        mean_t, std_t, args.perceptual_weight_path
    ).to(device)
```

Store both values in `config.json` so the checkpoint directory records the exact experiment.

- [ ] **Step 3: Add the perceptual term to the generator objective and log it.**

Immediately after `reconstruction_l1` is computed, add:

```python
perceptual = perceptual_l1(
    pred, field, perceptual_extractor, args.perceptual_weight
)
loss_g = reconstruction_l1 + perceptual + 0.1 * adversarial_bce
```

Replace the existing generator assignment only; discriminator loss and all masks remain unchanged. Include `perceptual={perceptual.item():.6f}` in the periodic log.

- [ ] **Step 4: Run the focused tests and the full existing unit suite.**

Run:

```bash
pytest -q test_train_sparse_adv_no.py -k perceptual
pytest -q
```

Expected: all new perceptual tests and all pre-existing tests pass.

### Task 3: Smoke-test the complete ablation path

**Files:**
- Create: `Gen4Turbulence/outputs/era_adv_no_medicalnet_smoke/` (generated output only)

- [ ] **Step 1: Run a one-step CPU/GPU-compatible smoke training command.**

```bash
CUDA_VISIBLE_DEVICES=3,4 python -u Gen4Turbulence/3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py \
  --data Gen4Turbulence/data/era_uv_2t4x_conservative.h5 \
  --output-dir Gen4Turbulence/outputs/era_adv_no_medicalnet_smoke \
  --patch 16 --depth 8 --batch-size 1 --epochs 1 --max-steps 1 \
  --perceptual-weight 1.0
```

Expected: process exits zero, logs finite reconstruction/GAN/perceptual values, and writes `best_model.pt`, `config.json`, and `metrics.json`.

- [ ] **Step 2: Verify the smoke artifacts and configuration.**

```bash
test -s Gen4Turbulence/outputs/era_adv_no_medicalnet_smoke/best_model.pt
python -c 'import json; from pathlib import Path; c=json.loads(Path("Gen4Turbulence/outputs/era_adv_no_medicalnet_smoke/config.json").read_text()); assert c["perceptual_weight"] == 1.0; assert c["perceptual_weight_path"].endswith("resnet_10_23dataset.pth"); print("smoke config verified")'
```

### Task 4: Train and evaluate the full ERA5 ablation

**Files:**
- Create: `Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/` (generated output only)

- [ ] **Step 1: Start persistent two-GPU training.**

```bash
tmux new-session -d -s era_medicalnet \
  "CUDA_VISIBLE_DEVICES=3,4 python -u Gen4Turbulence/3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py \
   --data Gen4Turbulence/data/era_uv_2t4x_conservative.h5 \
   --output-dir Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32 \
   --patch 64 --depth 8 --batch-size 32 --epochs 500 \
   --perceptual-weight 1.0 \
   > Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/train.log 2>&1"
```

- [ ] **Step 2: Confirm the session and monitor for finite losses/checkpoint progress.**

```bash
tmux has-session -t era_medicalnet
tail -n 40 Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/train.log
```

Stop and report the actual error if the MedicalNet forward exceeds available memory; do not silently change batch size.

- [ ] **Step 3: Evaluate fixed missing rates with the existing evaluator.**

Run these five commands from the repository root, changing only the fixed mask rate and output filename:

```bash
for rate in 0.1 0.3 0.5 0.7 0.9; do
  python -u Gen4Turbulence/3_flow_reconstruction/no/adv_training/evaluate_sparse_metrics.py \
    --data Gen4Turbulence/data/era_uv_2t4x_conservative.h5 \
    --checkpoint Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/best_model.pt \
    --config Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/config.json \
    --output Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/evaluation_${rate}.json \
    --mask-ratio "$rate" --patch 64 --depth 8 --stride 32 --tile-batch 8 --seed 25
done
```

Generate the 50%-missing visualization with:

```bash
python -u Gen4Turbulence/3_flow_reconstruction/no/adv_training/plot_sparse_visualization.py \
  --data Gen4Turbulence/data/era_uv_2t4x_conservative.h5 \
  --checkpoint Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/best_model.pt \
  --config Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/config.json \
  --output Gen4Turbulence/outputs/era_adv_no_medicalnet_bs32/era5_medicalnet_50pct.png \
  --mask-ratio 0.5 --sample 0 --time-index 4 --seed 25
```

- [ ] **Step 4: Compare against baseline and report the result.**

Compare each rate's MAE, RMSE, Relative L2, PSNR, and SSIM, and state whether MedicalNet improves or degrades the ERA5 reconstruction. Include training/runtime or memory caveats if observed.
