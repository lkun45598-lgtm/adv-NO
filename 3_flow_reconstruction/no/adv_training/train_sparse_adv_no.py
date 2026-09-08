"""Train the repository's 3D adv-NO generator on prepared sparse fields.

The runner keeps the original ``Unet3D`` generator and adds a compact
patch discriminator.  Data are read lazily from the HDF5 produced by
``prepare_sparse_data.py``; ``--max-steps`` can bound short diagnostic runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tcunet import Unet3D  # noqa: E402
from model import FeatureExtractor  # noqa: E402


class SparseH5Dataset(Dataset):
    def __init__(self, path: Path, split: str, patch: int, depth: int,
                 max_samples: int | None = None, training: bool = False):
        self.path, self.split = str(path), split
        self.patch, self.depth = patch, depth
        self.training = training
        with h5py.File(self.path, "r") as h5:
            shape = h5[split]["field"].shape
            mask_shape = h5[split]["valid_mask"].shape
            wet_shape = h5[split]["wet_fraction"].shape if "wet_fraction" in h5[split] else None
        if len(shape) != 5 or shape[1] != 2 or mask_shape != (shape[0], 1, *shape[2:]):
            raise ValueError(
                f"Expected field [N, 2, H, W, D] and mask [N, 1, H, W, D], "
                f"got {shape} and {mask_shape}"
            )
        if wet_shape is not None and wet_shape != (1, shape[2], shape[3], 1):
            raise ValueError(f"Expected wet_fraction [1, H, W, 1], got {wet_shape}")
        self.has_wet_fraction = wet_shape is not None
        self.length = min(shape[0], max_samples) if max_samples else shape[0]
        _, _, self.height, self.width, self.full_depth = shape
        if patch > min(self.height, self.width) or depth > self.full_depth:
            raise ValueError(f"Patch {(patch, patch, depth)} exceeds data shape {shape}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.path, "r") as h5:
            field = np.array(h5[self.split]["field"][index], dtype=np.float32, copy=True)
            valid = np.array(h5[self.split]["valid_mask"][index], dtype=np.float32, copy=True)
            if self.has_wet_fraction:
                wet_fraction = np.array(h5[self.split]["wet_fraction"], dtype=np.float32, copy=True)
            else:
                wet_fraction = np.ones((1, self.height, self.width, 1), dtype=np.float32)
        if self.training:
            y0 = random.randint(0, self.height - self.patch)
            x0 = random.randint(0, self.width - self.patch)
            z0 = random.randint(0, self.full_depth - self.depth)
        else:
            y0 = (self.height - self.patch) // 2
            x0 = (self.width - self.patch) // 2
            z0 = (self.full_depth - self.depth) // 2
        return (
            torch.from_numpy(field[:, y0:y0 + self.patch, x0:x0 + self.patch, z0:z0 + self.depth]),
            torch.from_numpy(valid[:, y0:y0 + self.patch, x0:x0 + self.patch, z0:z0 + self.depth]),
            torch.from_numpy(wet_fraction[:, y0:y0 + self.patch, x0:x0 + self.patch, :]
                             .repeat(self.depth, axis=-1)),
        )


class PatchDiscriminator3D(nn.Module):
    def __init__(self, channels: int = 2, features: int = 32):
        super().__init__()
        # Avoid InstanceNorm: its spatial statistics would mix PRE land into ocean logits.
        self.net = nn.Sequential(
            nn.Conv3d(channels, features, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features, features * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features * 2, features * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True), nn.Conv3d(features * 4, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def validity_mask(valid: torch.Tensor, output_shape) -> torch.Tensor:
        """Return logits whose full convolutional receptive field is valid.

        Average pooling is normalized by the corresponding padded all-ones
        pool, so an all-valid ERA5 patch remains valid at convolution borders.
        """
        support = valid
        for kernel, stride, padding in ((4, 2, 1), (4, 2, 1), (4, 2, 1), (3, 1, 1)):
            pad = (padding,) * 6
            padded_support = F.pad(support, pad, value=0.0)
            padded_ones = F.pad(torch.ones_like(support), pad, value=0.0)
            numerator = F.avg_pool3d(padded_support, kernel, stride, padding=0)
            denominator = F.avg_pool3d(padded_ones, kernel, stride, padding=0)
            support = numerator / denominator.clamp_min(torch.finfo(support.dtype).eps)
        if tuple(support.shape) != tuple(output_shape):
            raise ValueError(f"Validity mask shape {tuple(support.shape)} != logits {tuple(output_shape)}")
        # A nearly-one floating value can still include an invalid input voxel.
        # All-valid receptive fields divide identical pooled tensors and are exact.
        return (support == 1.0).to(dtype=valid.dtype)


def stats(dataset: SparseH5Dataset, max_items: int | None = None):
    count = np.zeros(2, dtype=np.float64)
    total = np.zeros(2, dtype=np.float64)
    total2 = np.zeros(2, dtype=np.float64)
    n = len(dataset) if max_items is None else min(len(dataset), max_items)
    with h5py.File(dataset.path, "r") as h5:
        field_ds, mask_ds = h5[dataset.split]["field"], h5[dataset.split]["valid_mask"]
        wet_ds = h5[dataset.split].get("wet_fraction")
        for i in range(n):
            field = np.asarray(field_ds[i], dtype=np.float64)
            mask = np.asarray(mask_ds[i, 0], dtype=bool)
            if wet_ds is None:
                wet_fraction = np.ones(mask.shape, dtype=np.float64)
            else:
                wet_fraction = np.broadcast_to(np.asarray(wet_ds[0], dtype=np.float64), mask.shape)
            for c in range(2):
                valid = mask & np.isfinite(field[c])
                weights = wet_fraction[valid]
                values = field[c][valid]
                count[c] += weights.sum()
                total[c] += (weights * values).sum()
                total2[c] += (weights * np.square(values)).sum()
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(total2 / np.maximum(count, 1) - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def sample_training_ratios(batch_size: int, minimum: float, maximum: float,
                           device: torch.device,
                           generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one missing-rate uniformly for each item in a training batch."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        raise ValueError("training mask-rate bounds must be finite")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError(
            f"training mask-rate bounds must satisfy 0 <= min <= max <= 1, got {minimum}, {maximum}"
        )
    return torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(
        minimum, maximum, generator=generator
    )


def make_mask(field: torch.Tensor, valid: torch.Tensor, ratio: float | torch.Tensor,
              mean_t: torch.Tensor | None = None,
              generator: torch.Generator | None = None):
    # One shared sparse-observation mask preserves the vector nature of uv.
    batch_size = field.shape[0]
    if torch.is_tensor(ratio):
        ratios = ratio.to(device=valid.device, dtype=torch.float32)
        if ratios.ndim == 0:
            ratios = ratios.expand(batch_size)
        elif tuple(ratios.shape) != (batch_size,):
            raise ValueError(
                f"per-sample mask ratios must have shape [{batch_size}], got {tuple(ratios.shape)}"
            )
    else:
        ratios = torch.full((batch_size,), float(ratio), device=valid.device, dtype=torch.float32)
    if torch.any((ratios < 0.0) | (ratios > 1.0)):
        raise ValueError(f"mask ratios must be in [0, 1], got {ratios.detach().cpu().tolist()}")
    random_values = torch.rand(valid.shape, device=valid.device, dtype=valid.dtype,
                               generator=generator)
    observed = (random_values >= ratios.reshape(batch_size, 1, 1, 1, 1)).to(valid.dtype) * valid
    if mean_t is None:
        mean_t = torch.zeros((1, field.shape[1], 1, 1, 1), dtype=field.dtype,
                             device=field.device)
    fill = mean_t.expand_as(field)
    masked = torch.where(observed.expand_as(field).bool(), field, fill)
    return torch.cat((masked, observed.repeat(1, field.shape[1], 1, 1, 1)), dim=1), observed


def masked_l1(pred, target, valid, wet_fraction=None):
    weight = valid.expand_as(pred)
    if wet_fraction is not None:
        weight = weight * wet_fraction.expand_as(pred)
    error = torch.where(
        weight.bool(), torch.abs(pred - target) * weight, torch.zeros_like(pred)
    )
    return error.sum() / weight.sum().clamp_min(1.0)


def masked_bce_with_logits(logits, target, valid):
    """BCE averaged only over valid discriminator logits."""
    weight = valid.expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = torch.where(weight.bool(), loss, torch.zeros_like(loss))
    denominator = weight.sum()
    if denominator.item() == 0:
        return (loss * weight).sum()
    return (loss * weight).sum() / denominator


def masked_logit_mean(logits, valid):
    """Average discriminator logits over valid receptive fields only."""
    weight = valid.expand_as(logits)
    denominator = weight.sum()
    if denominator.item() == 0:
        return (logits * weight).sum()
    return (logits * weight).sum() / denominator


def masked_ragan_discriminator_loss(real_logits, fake_logits, valid):
    """RaGAN discriminator loss with all reductions restricted to valid logits."""
    mean_real = masked_logit_mean(real_logits, valid)
    mean_fake = masked_logit_mean(fake_logits, valid)
    loss_real = masked_bce_with_logits(
        real_logits - mean_fake, torch.ones_like(real_logits), valid
    )
    loss_fake = masked_bce_with_logits(
        fake_logits - mean_real, torch.zeros_like(fake_logits), valid
    )
    return 0.5 * (loss_real + loss_fake)


def masked_ragan_generator_loss(real_logits, fake_logits, valid):
    """RaGAN generator loss with all reductions restricted to valid logits."""
    mean_real = masked_logit_mean(real_logits, valid)
    mean_fake = masked_logit_mean(fake_logits, valid)
    loss_fake = masked_bce_with_logits(
        fake_logits - mean_real, torch.ones_like(fake_logits), valid
    )
    loss_real = masked_bce_with_logits(
        real_logits - mean_fake, torch.zeros_like(real_logits), valid
    )
    return 0.5 * (loss_fake + loss_real)


def masked_bce_discriminator_loss(real_logits, fake_logits, valid):
    """Standard GAN discriminator BCE restricted to valid logits."""
    return 0.5 * (
        masked_bce_with_logits(real_logits, torch.ones_like(real_logits), valid)
        + masked_bce_with_logits(fake_logits, torch.zeros_like(fake_logits), valid)
    )


def masked_bce_generator_loss(fake_logits, valid):
    """Standard GAN generator BCE restricted to valid logits."""
    return masked_bce_with_logits(fake_logits, torch.ones_like(fake_logits), valid)


def masked_normalized_l1(pred, target, valid, mean_t, std_t, wet_fraction=None):
    pred = (pred - mean_t) / std_t
    target = (target - mean_t) / std_t
    return masked_l1(pred, target, valid, wet_fraction)


def build_perceptual_extractor(mean, std, weight_path):
    """Build the frozen MedicalNet feature extractor used by the paper loss."""
    par = {
        "out_shift": mean.detach().cpu().reshape(-1, 1, 1, 1, 1),
        "out_scale": std.detach().cpu().reshape(-1, 1, 1, 1, 1),
    }
    extractor = FeatureExtractor(
        par, weight_path=str(weight_path), layers=[1, 2, 3]
    )
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor


def perceptual_l1(pred, target, extractor, weight=0.0):
    """Compute the weighted frozen-MedicalNet feature distance."""
    if weight == 0.0:
        return pred.new_zeros(())
    if extractor is None:
        raise ValueError("extractor is required when perceptual weight is nonzero")
    pred_features = extractor(pred)
    with torch.no_grad():
        target_features = extractor(target)
    stage_weights = (0.1, 0.1, 1.0)
    return float(weight) * sum(
        F.l1_loss(pred_feature, target_feature) * stage_weight
        for pred_feature, target_feature, stage_weight in zip(
            pred_features, target_features, stage_weights
        )
    )


def unwrap_parallel(module: nn.Module) -> nn.Module:
    """Return the underlying module for stable, portable checkpoints."""
    return module.module if isinstance(module, nn.DataParallel) else module


def select_generator_state(checkpoint: dict, prefer_ema: bool = True):
    """Return the requested generator state and its checkpoint key."""
    if prefer_ema and "generator_ema" in checkpoint:
        return checkpoint["generator_ema"], "generator_ema"
    if "generator" in checkpoint:
        return checkpoint["generator"], "generator"
    raise KeyError("Checkpoint must contain 'generator' or 'generator_ema'")


class ModelEMA:
    """Maintain an exponential moving average of a generator's state."""
    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.model = copy.deepcopy(unwrap_parallel(model)).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        source_state = unwrap_parallel(model).state_dict()
        for key, averaged_value in self.model.state_dict().items():
            source_value = source_state[key].detach().to(device=averaged_value.device)
            if torch.is_floating_point(averaged_value):
                averaged_value.lerp_(source_value.to(dtype=averaged_value.dtype), 1.0 - self.decay)
            else:
                averaged_value.copy_(source_value)

    def state_dict(self):
        return self.model.state_dict()


def evaluate(generator, loader, device, mean_t, std_t, mask_ratio, seed: int):
    was_training = generator.training
    generator.eval()
    losses = []
    mask_generator = torch.Generator(device=device.type).manual_seed(seed)
    with torch.no_grad():
        for field, valid, wet_fraction in loader:
            field, valid, wet_fraction = field.to(device), valid.to(device), wet_fraction.to(device)
            model_in, _ = make_mask(field, valid, mask_ratio, mean_t=mean_t,
                                    generator=mask_generator)
            pred = generator(model_in, torch.ones(field.shape[0], device=device))
            losses.append(masked_normalized_l1(pred, field, valid, mean_t, std_t, wet_fraction).item())
    generator.train(was_training)
    return float(np.mean(losses)) if losses else float("nan")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stats-max-items", type=int, default=None,
                        help="Limit training records used for normalization statistics")
    parser.add_argument("--mask-ratio", type=float, default=0.5,
                        help="Fixed missing rate used by validation and final test loss")
    parser.add_argument("--train-mask-min", type=float, default=0.1,
                        help="Minimum random missing rate sampled during training")
    parser.add_argument("--train-mask-max", type=float, default=0.9,
                        help="Maximum random missing rate sampled during training")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--adversarial-loss", choices=("bce", "ragan"), default="ragan",
                        help="Adversarial formulation used when --adversarial-weight is positive")
    parser.add_argument("--adversarial-weight", type=float, default=0.1,
                        help="Set to zero to train the L1-only NO pretraining stage")
    parser.add_argument("--init-generator", type=Path,
                        help="Checkpoint from which to initialize the generator")
    parser.add_argument("--init-use-raw-generator", action="store_true",
                        help="Use 'generator' instead of 'generator_ema' from --init-generator")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="Generator EMA decay; set to zero to disable EMA")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="PyTorch intra-op CPU threads used for batch collation and host-side work",
    )
    parser.add_argument("--perceptual-weight", type=float, default=0.0)
    parser.add_argument(
        "--perceptual-weight-path",
        type=Path,
        default=HERE / "pretrain" / "resnet_10_23dataset.pth",
    )
    return parser


def validate_args(args):
    if args.patch % 8 or args.depth % 8:
        raise ValueError("--patch and --depth must be divisible by 8 for the 3D U-Net")
    if args.perceptual_weight != 0.0:
        raise ValueError(
            "--perceptual-weight must be zero: MedicalNet is not mask-aware for sparse reconstruction"
        )
    if args.adversarial_weight < 0.0:
        raise ValueError("--adversarial-weight must be non-negative")
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    return args


def configure_cpu_threads(cpu_threads: int) -> None:
    torch.set_num_threads(cpu_threads)


def main():
    args = validate_args(build_arg_parser().parse_args())
    configure_cpu_threads(args.cpu_threads)
    print(f"Using {torch.get_num_threads()} PyTorch CPU thread(s)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask_generator = torch.Generator(device=device.type).manual_seed(args.seed + 10)
    train_ds = SparseH5Dataset(args.data, "train", args.patch, args.depth, args.max_samples, True)
    val_ds = SparseH5Dataset(args.data, "val", args.patch, args.depth, args.max_samples, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    mean, std = stats(train_ds, max_items=args.stats_max_items)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 2, 1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 2, 1, 1, 1)
    perceptual_extractor = None
    if args.perceptual_weight > 0.0:
        perceptual_extractor = build_perceptual_extractor(
            mean_t, std_t, args.perceptual_weight_path
        ).to(device)
    par = {
        "nf": 2, "nx": args.patch, "ny": args.patch, "nz": args.depth,
        "inp_shift": torch.tensor(np.r_[mean, [0.0, 0.0]], dtype=torch.float32, device=device).reshape(1, 4, 1, 1, 1),
        "inp_scale": torch.tensor(np.r_[std, [1.0, 1.0]], dtype=torch.float32, device=device).reshape(1, 4, 1, 1, 1),
        "out_shift": torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 2, 1, 1, 1),
        "out_scale": torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 2, 1, 1, 1),
        "t_shift": torch.tensor(0.0, device=device), "t_scale": torch.tensor(1.0, device=device),
    }
    generator = Unet3D(dim=args.dim, Par=par, dim_mults=(1, 2, 4, 8), channels=4).to(device)
    init_source = None
    if args.init_generator:
        checkpoint = torch.load(args.init_generator, map_location=device, weights_only=False)
        initial_state, init_source = select_generator_state(
            checkpoint, prefer_ema=not args.init_use_raw_generator
        )
        generator.load_state_dict(initial_state)
        print(f"Initialized generator from {args.init_generator} ({init_source})")
    discriminator = PatchDiscriminator3D().to(device) if args.adversarial_weight > 0.0 else None
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        generator = nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = nn.DataParallel(discriminator)
        print(f"Using DataParallel across {gpu_count} visible GPUs")
    else:
        print(f"Using device: {device}")
    generator_ema = ModelEMA(generator, args.ema_decay) if args.ema_decay > 0.0 else None
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4) if discriminator is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(device), "gpu_count": gpu_count,
                   "data_parallel": gpu_count > 1,
                   "mean": mean.tolist(), "std": std.tolist(),
                   "init_generator_source": init_source,
                   "validation_weights": "generator_ema" if generator_ema else "generator"})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    step = 0
    best = float("inf")
    for epoch in range(args.epochs):
        for field, valid, wet_fraction in train_loader:
            field, valid, wet_fraction = field.to(device), valid.to(device), wet_fraction.to(device)
            train_ratios = sample_training_ratios(
                field.shape[0], args.train_mask_min, args.train_mask_max,
                device=device, generator=mask_generator,
            )
            model_in, _ = make_mask(field, valid, train_ratios, mean_t=mean_t,
                                    generator=mask_generator)
            if discriminator is not None:
                # Discriminator update.
                with torch.no_grad():
                    fake_detached = generator(model_in, torch.ones(field.shape[0], device=device))
                opt_d.zero_grad(set_to_none=True)
                real_logits = discriminator(((field - mean_t) / std_t) * valid)
                fake_logits = discriminator(((fake_detached - mean_t) / std_t) * valid)
                valid_logits = PatchDiscriminator3D.validity_mask(valid, real_logits.shape)
                if args.adversarial_loss == "ragan":
                    loss_d = masked_ragan_discriminator_loss(real_logits, fake_logits, valid_logits)
                else:
                    loss_d = masked_bce_discriminator_loss(real_logits, fake_logits, valid_logits)
                loss_d.backward()
                opt_d.step()
            else:
                loss_d = field.new_zeros(())
            # Generator update, using the paper's 0.1 adversarial weighting by default.
            opt_g.zero_grad(set_to_none=True)
            pred = generator(model_in, torch.ones(field.shape[0], device=device))
            reconstruction_l1 = masked_normalized_l1(
                pred, field, valid, mean_t, std_t, wet_fraction
            )
            if discriminator is not None:
                real_logits = discriminator(((field - mean_t) / std_t) * valid).detach()
                fake_logits = discriminator(((pred - mean_t) / std_t) * valid)
                if args.adversarial_loss == "ragan":
                    adversarial_loss = masked_ragan_generator_loss(
                        real_logits, fake_logits, valid_logits
                    )
                else:
                    adversarial_loss = masked_bce_generator_loss(fake_logits, valid_logits)
            else:
                adversarial_loss = field.new_zeros(())
            perceptual = perceptual_l1(
                pred, field, perceptual_extractor, args.perceptual_weight
            )
            loss_g = reconstruction_l1 + perceptual + args.adversarial_weight * adversarial_loss
            loss_g.backward()
            opt_g.step()
            if generator_ema is not None:
                generator_ema.update(generator)
            step += 1
            if step % 10 == 0 or step == 1:
                print(
                    f"epoch={epoch + 1} step={step} "
                    f"reconstruction_l1={reconstruction_l1.item():.6f} "
                    f"perceptual={perceptual.item():.6f} "
                    f"adversarial={adversarial_loss.item():.6f} "
                    f"discriminator_adversarial={loss_d.item():.6f}"
                )
            if args.max_steps is not None and step >= args.max_steps:
                break
        validation_generator = generator_ema.model if generator_ema is not None else generator
        val_loss = evaluate(validation_generator, val_loader, device, mean_t, std_t, args.mask_ratio,
                            seed=args.seed + 1)
        print(f"epoch={epoch + 1} step={step} val_l1={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            checkpoint = {
                "generator": unwrap_parallel(generator).state_dict(),
                "par": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in par.items()},
                "val_l1": best,
                "adversarial_loss": args.adversarial_loss if discriminator is not None else None,
                "adversarial_weight": args.adversarial_weight,
            }
            if generator_ema is not None:
                checkpoint["generator_ema"] = generator_ema.state_dict()
            if discriminator is not None:
                checkpoint["discriminator"] = unwrap_parallel(discriminator).state_dict()
            torch.save(checkpoint, args.output_dir / "best_model.pt")
        if args.max_steps is not None and step >= args.max_steps:
            break
    test_ds = SparseH5Dataset(args.data, "test", args.patch, args.depth, args.max_samples, False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_generator = generator_ema.model if generator_ema is not None else generator
    test_loss = evaluate(test_generator, test_loader, device, mean_t, std_t, args.mask_ratio,
                         seed=args.seed + 2)
    metrics = {"best_val_l1": best, "test_l1": test_loss, "steps": step,
               "evaluation_weights": "generator_ema" if generator_ema else "generator"}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
