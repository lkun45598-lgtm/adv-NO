# adv-NO 海洋流场与全球风场稀疏重建项目汇报

> 项目状态：已完成 Task A（PRE 海洋三维 `u/v`）和 Task D（ERA5 全球风场 `u/v`）的保守下采样、稀疏掩码训练、纯 NO 预训练、GAN 微调、EMA 推理和多缺失率评价。
>
> PRE 最终模型采用“0%--100% 随机缺失率纯 NO 预训练 + RaGAN 微调 + EMA”；ERA5 同时报告标准缺失率训练模型和 `0%--100%` 随机缺失率的极端稀疏实验。

## 1. 项目概述

### 1.1 背景

本项目基于 Gen4Turbulence 仓库的 `3_flow_reconstruction/no/adv_training` 实现，将 3D U-Net 神经算子生成器与 3D PatchGAN 判别器用于规则网格上的稀疏场重建。仓库 README 将该分支列为 Flow Reconstruction 的 GAN 方法，并另列独立扩散模型分支 `3_flow_reconstruction/dm`。本考核明确要求使用 adv-NO，因此当前实现选择前者，而不是扩散模型。目标是在只给出部分观测值的情况下恢复完整场：PRE 恢复二维水平空间上的 30 个 sigma 垂向层，ERA5 恢复全球二维空间上的连续时间窗口，并保持速度场的幅值和结构。

原论文及仓库覆盖超分辨率、预测和稀疏重建。本项目保留仓库稀疏重建 GAN 分支的生成器和对抗训练思路，针对 PRE 海洋数据和 ERA5 风场补充了：

- 2 倍时间、4 倍空间的物理量下采样；
- 时间顺序的 `train/val/test = 8:1:1` 切分；
- 训练阶段逐样本随机缺失率；
- 显式有效区域和陆地掩码；
- 只在真正缺失且有效的位置计算指标；
- 多缺失率和多随机种子评价；
- 数据匹配的纯 NO 预训练、GAN 微调和 EMA 权重评估。

### 1.2 任务选择

考核任务包括：

| 任务 | 数据 | 当前状态 |
| --- | --- | --- |
| A | PRE 海洋三维流场 `u/v` 稀疏重建 | 已完成 |
| B | OSTIA SST 稀疏重建 | 当前项目未训练 |
| C | Copernicus 全球流速场稀疏重建 | 当前项目未训练 |
| D | ERA5 全球风速场 `u/v` 稀疏重建 | 已完成 |

因此当前汇报覆盖任务 A 和任务 D，分别代表海洋流速场和大气风场。

## 2. 代码与目录

项目根目录为 `Gen4Turbulence/`。原始仓库说明见 [`README.md`](../README.md)，稀疏任务说明见 [`README_sparse_tasks.md`](../3_flow_reconstruction/no/adv_training/README_sparse_tasks.md)。本项目的关键文件如下：

| 功能 | 文件 |
| --- | --- |
| 数据预处理 | [`prepare_sparse_data.py`](../3_flow_reconstruction/no/adv_training/prepare_sparse_data.py) |
| 训练 runner | [`train_sparse_adv_no.py`](../3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py) |
| 3D U-Net 生成器 | [`tcunet.py`](../3_flow_reconstruction/no/tcunet.py) |
| 稀疏掩码、训练损失 | [`train_sparse_adv_no.py`](../3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py) |
| 指标计算 | [`metrics_sparse.py`](../3_flow_reconstruction/no/adv_training/metrics_sparse.py) |
| 推理和单次评价 | [`evaluate_sparse_metrics.py`](../3_flow_reconstruction/no/adv_training/evaluate_sparse_metrics.py) |
| 多种子汇总 | [`aggregate_sparse_metrics.py`](../3_flow_reconstruction/no/adv_training/aggregate_sparse_metrics.py) |
| 可视化 | [`plot_sparse_visualization.py`](../3_flow_reconstruction/no/adv_training/plot_sparse_visualization.py) |

## 3. 数据与预处理

### 3.1 数据来源和变量

| 数据集 | 源数据地址 | 变量含义 | 原始/中间形状 | 正式 HDF5 |
| --- | --- | --- | --- | --- |
| PRE（Task A） | `/data/PRE_ocean_data` | `u_eastward`、`v_northward` | 原始 `[10591, 30, 400, 441]`；去除不可整除边界后使用 `[10590, 30, 400, 440]` | `data/pre_uv_2t4x_conservative.h5` |
| ERA5（Task D） | `/data/era_data/raw_data/era_721_1440.pt` | `u_component_of_wind`、`v_component_of_wind` | 原始 `[20, 24, 721, 1440, 4]`；展平为 480 个时次 | `data/era_uv_2t4x_conservative.h5` |

正式结果使用 `conservative` 文件。它们位于：

- [`pre_uv_2t4x_conservative.h5`](../data/pre_uv_2t4x_conservative.h5)
- [`era_uv_2t4x_conservative.h5`](../data/era_uv_2t4x_conservative.h5)

### 3.2 2 倍时间下采样

相邻两个时间记录做平均：

$$
X_{2t}[k] = \frac{X[2k] + X[2k+1]}{2}.
$$

PRE 共得到 5295 个时间记录，最后剩余的单条记录不参与配对。ERA5 先将日期和小时展平，再对相邻两个小时求平均，得到 240 个 2 小时记录。

### 3.3 4 倍空间下采样

本项目没有采用简单点抽取，也没有使用 bicubic 插值作为正式方案，而是对每个 `4 x 4` 网格块做面积加权平均：

$$
\bar{x} = \frac{\sum_{i \in B} w_i m_i x_i}{\sum_{i \in B} w_i m_i}.
$$

其中 `B` 是一个 `4 x 4` 网格块，`w_i` 是网格面积权重，`m_i` 是有效/湿润掩码。

- PRE：`w_i` 使用 `pm/pn` 推导的网格面积，`m_i` 使用 `mask_rho` 和有限值检查；陆地点不进入平均。
- ERA5：规则经纬网采用 `cos(latitude)` 作为面积权重。为保证整除，去掉南端一个纬度端点，使用 `720 x 1440` 网格，输出 `180 x 360`。

这种处理保留了空间平均意义和物理量尺度，尤其适合存在海陆边界的 PRE 数据。

### 3.4 有效掩码和湿润比例

PRE 每个粗网格位置保存：

- `valid_mask`：粗网格内 `u` 和 `v` 均有有效湿润源单元时为 1，否则为 0；
- `wet_fraction`：粗网格中湿润网格面积占总网格面积的比例。

ERA5 当前输入没有附带海陆掩码，`valid_mask` 全为 1，表示全球大气场的所有网格均作为有效样本。若后续要求只评估海洋区域，需要额外提供 ERA5 的外部陆地掩码。

### 3.5 数据切分

数据按时间顺序切分，避免相邻时间样本跨集合泄漏。切分比例为 8:1:1：

| 数据集 | 处理后字段形状 `[N, 2, H, W, D]` | train | val | test | 有效比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRE | `[5295, 2, 100, 110, 30]` | 4236 | 529 | 530 | 约 73.37% |
| ERA5 | `[219, 2, 180, 360, 8]` | 185 | 17 | 17 | 100% |

ERA5 的深度轴不是物理深度，而是连续的 8 个 2 小时时间片。窗口在每个时间切分内部单独生成，因此任何窗口都不会跨越 train、val、test 边界。

## 4. 稀疏观测构造

### 4.1 输入掩码

给定完整字段 `field` 和物理有效掩码 `valid_mask`，每个样本随机生成一个共享的空间-时间观测掩码 `M`：

$$
M \sim \mathrm{Bernoulli}(1-r) \odot valid\_mask,
$$

其中 `r` 是缺失率。`u` 和 `v` 使用同一个 `M`，以保持速度矢量的对应关系。

缺失值用训练集对应分量均值填充，均值填充值在模型归一化后约为 0。模型实际输入的四个通道为：

| 通道 | 内容 |
| ---: | --- |
| 0 | 掩码后的 `u`，缺失处为训练集 `u` 均值 |
| 1 | 掩码后的 `v`，缺失处为训练集 `v` 均值 |
| 2 | `u` 的观测掩码 |
| 3 | `v` 的观测掩码 |

### 4.2 训练和测试缺失率

- PRE 最终模型：每个 batch 内逐样本采样 `r ~ Uniform(0.0, 1.0)`。
- ERA5 标准模型：同样使用 `r ~ Uniform(0.1, 0.9)`。
- ERA5 极端稀疏模型：使用 `r ~ Uniform(0.0, 1.0)`，用于覆盖接近无观测的情况。
- 训练脚本的验证和最终 `test_l1`：固定采用 50% 缺失率，便于 checkpoint 选择。
- 正式推理：固定测试 10%、30%、50%、70%、90%；PRE 还测试 1% 缺失率，PRE 最终模型和 ERA5 极端稀疏模型均额外测试 99% 缺失率，即仅 1% 观测。

因此测试集不存在唯一的固定缺失率；报告中的每一行结果都对应一个明确的测试缺失率。随机缺失率仅用于训练增强，正式测试使用固定缺失率和固定 seed，保证结果可复现。

## 5. adv-NO 模型

### 5.1 输入输出

训练时从完整样本随机裁剪 patch：

```text
PRE:   model_input [B, 4, 64, 64, 16] -> prediction [B, 2, 64, 64, 16]
ERA5:  model_input [B, 4, 64, 64, 8]  -> prediction [B, 2, 64, 64, 8]
```

输出通道含义与输入字段一致：

- 通道 0：重建后的 eastward 分量 `u`；
- 通道 1：重建后的 northward 分量 `v`。

`Unet3D.forward()` 内部使用保存于 checkpoint 的 `out_shift/out_scale` 进行反归一化，因此输出已经回到原始物理量尺度。

### 5.2 生成器

生成器是仓库原有的 3D U-Net，配置为：

```text
dim = 16
dim_mults = (1, 2, 4, 8)
input channels = 4
output channels = 2
```

网络沿空间和深度三个轴进行 3D 卷积、下采样和上采样，并通过跳跃连接保留局部结构。

### 5.3 判别器

判别器是新增的紧凑 3D PatchGAN，由多层 `Conv3d` 和 `LeakyReLU` 组成，输出局部真假 logits。判别器不使用 `InstanceNorm`，避免空间统计将 PRE 陆地值传播到海洋区域。

只有完整卷积感受野都位于有效区域的 logits 才参与对抗损失。这样可以排除 PRE 陆地和邻近无效感受野对 GAN loss 的影响。

### 5.4 两阶段训练与 EMA

PRE 与 ERA5 最终实验都采用两阶段流程：

1. 先使用归一化、有效区域约束的 L1 重建损失训练纯 NO 生成器；
2. 再从第一阶段最佳 checkpoint 的 `generator_ema` 权重出发进行 GAN 微调。

PRE 最终阶段使用 RaGAN。ERA5 标准范围实验分别做了 BCE 和 RaGAN 对照；极端稀疏实验使用 RaGAN。EMA 衰减率均为 `0.999`，验证、测试和正式推理默认使用 EMA 生成器权重；原始生成器权重仍保存在 checkpoint 中，可通过 `--raw-generator` 做消融。

## 6. 损失函数和归一化

### 6.1 训练集归一化

对 `u`、`v` 分别计算训练集统计量：

$$
x'_{c} = \frac{x_c - \mu_c}{\sigma_c}.
$$

统计量只来自 train split 的有效区域，val/test 不参与均值和标准差计算。

| 数据集 | `u` mean | `v` mean | `u` std | `v` std |
| --- | ---: | ---: | ---: | ---: |
| PRE | -0.04966159 | -0.02448541 | 0.19870540 | 0.11505016 |
| ERA5 | 2.98951411 | -0.01058338 | 11.58474255 | 4.26915359 |

PRE 的陆地点和无效值不参与统计；ERA5 当前全局 `valid_mask=1`。

### 6.2 生成器损失

重建项是归一化、掩码感知的 L1：

$$
L_{recon} = \frac{\sum |x'_{pred}-x'_{target}| \cdot valid \cdot wet\_fraction}
{\sum valid \cdot wet\_fraction}.
$$

当前 GAN 阶段的总生成器损失为：

$$
L_G = L_{recon} + 0.1 L_{GAN}.
$$

纯 NO 预训练阶段设置 `L_{GAN}=0`，因此只优化 `L_{recon}`。BCE 和 RaGAN 微调阶段的重建项和权重完全一致，仅替换 `L_{GAN}` 的具体定义。

原仓库中存在 MedicalNet 感知损失实现，但本项目设置 `perceptual_weight=0`，因此没有进入正式结果。原因是该特征提取器的卷积和 BatchNorm 统计不是陆地掩码感知的，直接用于 PRE 可能让陆地填充值影响特征距离。MedicalNet 可作为后续单独消融实验，而不是当前 baseline 的已启用创新模块。

### 6.3 判别器损失

$$
L_D = \frac{1}{2}\left[
\mathrm{BCE}(D(x),1)+\mathrm{BCE}(D(G(x)),0)
\right].
$$

RaGAN 版本先在有效 logits 上计算相对平均值：

$$
\begin{aligned}
L_D^{Ra} &= \frac{1}{2}\left[
\mathrm{BCE}(d_r-\overline{d_f},1)+
\mathrm{BCE}(d_f-\overline{d_r},0)\right],\\
L_{GAN}^{Ra} &= \frac{1}{2}\left[
\mathrm{BCE}(d_f-\overline{d_r},1)+
\mathrm{BCE}(d_r-\overline{d_f},0)\right].
\end{aligned}
$$

其中 `\overline{d}` 只对完整有效感受野的 logits 求平均。

判别器输入为归一化后的完整有效区域；无效 logits 被 `validity_mask` 排除。BCE 版本使用普通真假分类，RaGAN 版本使用有效 logits 上的相对平均判别结果。两者都不让无效或陆地感受野参与数值或梯度计算。

### 6.4 优化配置

```text
optimizer: Adam
generator learning rate: 2e-4
discriminator learning rate: 1e-4
epochs: 500
generator dim: 16
```

正式训练使用多 GPU `DataParallel`（运行时可通过 `CUDA_VISIBLE_DEVICES` 指定可见卡）。

## 7. 实验配置与训练结果

所有正式训练均为 500 epochs，生成器维度 `dim=16`，学习率 `2e-4`，EMA 衰减率 `0.999`。PRE 使用 batch size 64、33500 steps；ERA5 使用 batch size 32、3000 steps。

### 7.1 PRE 两阶段最终结果

| 阶段 | 训练缺失率 | 对抗项 | 初始化 | best val normalized L1 | test normalized L1（50% 缺失） | checkpoint |
| --- | --- | --- | --- | ---: | ---: | --- |
| 纯 NO 预训练 | 0%--100% | 无 | 随机 | 0.01813946 | 0.01721473 | `outputs/pre_no_pretrain_0to100_ema_bs64/best_model.pt` |
| RaGAN 微调 | 0%--100% | RaGAN，权重 0.1 | NO EMA | **0.01642772** | **0.01605174** | `outputs/pre_ragan_pretrained_0to100_ema_bs64/best_model.pt` |

RaGAN 微调相对纯 NO 阶段使 best validation L1 降低约 9.44%，test L1 降低约 6.76%。PRE 最终汇报和可视化均使用第二阶段的 `generator_ema`。

### 7.2 ERA5 标准范围对照

以下实验均使用 `Uniform(0.1, 0.9)` 训练缺失率：

| 实验 | 初始化 | 对抗项 | EMA | best val normalized L1 | test normalized L1（50% 缺失） | checkpoint |
| --- | --- | --- | --- | ---: | ---: | --- |
| Direct BCE | 随机 | BCE | 否 | 0.05484968 | 0.05674314 | `outputs/era_adv_no_random_bs32/best_model.pt` |
| Direct RaGAN | 随机 | RaGAN | 否 | 0.05698321 | 0.06278615 | `outputs/era_ragan_random_bs32/best_model.pt` |
| 纯 NO 预训练 | 随机 | 无 | 是 | 0.07577464 | 0.07910468 | `outputs/era_no_pretrain_bs32/best_model.pt` |
| 预训练 + BCE | NO EMA | BCE | 是 | **0.04897056** | **0.04833671** | `outputs/era_bce_pretrained_ema_bs32/best_model.pt` |
| 预训练 + RaGAN | NO EMA | RaGAN | 是 | 0.04950763 | 0.04885580 | `outputs/era_ragan_pretrained_ema_bs32/best_model.pt` |

在标准 10%--90% 缺失率范围内，预训练 + EMA + BCE 的综合误差最低。与随机初始化直接训练相比，BCE 的 50% 缺失率 test L1 降低约 14.81%，RaGAN 降低约 22.19%。

### 7.3 ERA5 极端稀疏训练

为测试仅 1% 观测时的稳定性，额外把训练缺失率扩展为 `Uniform(0.0, 1.0)`：

| 阶段 | 训练缺失率 | 对抗项 | 初始化 | best val normalized L1 | test normalized L1（50% 缺失） | checkpoint |
| --- | --- | --- | --- | ---: | ---: | --- |
| 纯 NO 预训练 | 0%--100% | 无 | 随机 | 0.07878698 | 0.08250125 | `outputs/era_no_pretrain_0to100_ema_bs32/best_model.pt` |
| RaGAN 微调 | 0%--100% | RaGAN，权重 0.1 | NO EMA | **0.05674879** | **0.06660492** | `outputs/era_ragan_pretrained_0to100_ema_bs32/best_model.pt` |

第二阶段使 best validation L1 降低约 27.97%，50% 缺失率 test L1 降低约 19.27%。更宽的训练分布在后续物理指标中表现出明确权衡：10%--70% 缺失时不如标准 BCE，90% 缺失时则更好。因此该模型适合作为高缺失率鲁棒性方案，而不是无条件替代标准范围的最佳 BCE 模型。

PRE 与 ERA5 的 normalized L1 不可直接横向比较，因为两套数据的物理尺度、分布和归一化统计量不同。

## 8. 多缺失率评价结果

评价区域严格为：

```text
valid_mask AND missing_mask
```

即只评价有效且真正需要重建的点，不包含陆地、无效点和已观测点。

### 8.1 PRE 最终模型：预训练 + RaGAN + EMA

下表使用完整 530 条 test 样本、seed 25，所有数值仅统计有效海洋缺失点。该最终模型训练时的缺失率覆盖 `0%--100%`，因此 99% 缺失率也属于训练分布覆盖范围：

| 缺失率 | MAE | RMSE | Bias | NRMSE | EPE | Relative L2 | PSNR (dB) | SSIM | Pearson r |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.001696 | 0.003072 | 0.000040 | 0.000926 | 0.002687 | 0.019592 | 60.695 | 0.999720 | 0.999798 |
| 10% | 0.001785 | 0.003269 | -0.000023 | 0.000991 | 0.002826 | 0.020968 | 60.121 | 0.999661 | 0.999773 |
| 30% | 0.002135 | 0.003979 | -0.000037 | 0.001217 | 0.003373 | 0.025778 | 58.366 | 0.999442 | 0.999651 |
| 50% | 0.002685 | 0.005111 | -0.000026 | 0.001574 | 0.004239 | 0.033346 | 56.164 | 0.999055 | 0.999407 |
| 70% | 0.003694 | 0.007161 | -0.000044 | 0.002214 | 0.005831 | 0.046924 | 53.220 | 0.998099 | 0.998816 |
| 90% | 0.006484 | 0.012037 | 0.000128 | 0.003730 | 0.010235 | 0.079064 | 48.702 | 0.993863 | 0.996625 |
| 99% | 0.014234 | 0.023330 | 0.000157 | 0.007242 | 0.022417 | 0.153520 | 42.949 | 0.967172 | 0.987237 |

PRE 在整个训练覆盖范围内性能随缺失率平稳下降。即使 90% 缺失，Relative L2 仍约为 7.91%，Pearson `r` 约为 0.997；99% 缺失时 Relative L2 为 15.35%，SSIM 为 0.9672。99% 工况的模型 MAE（0.01423）仍低于最近邻（0.02818）和高斯插值（0.02596），说明扩展训练缺失率后，PRE 的极端稀疏重建相比旧版 10%--90% 训练模型有明显改善。

### 8.2 ERA5 风场：随机初始化直接 BCE

本节与 8.3、8.4 使用三个随机种子（25、42、2026）的均值 ± 样本标准差，用于标准训练策略对照。

| 缺失率 | MAE | RMSE | Bias | NRMSE | EPE | Relative L2 (%) | PSNR (dB) | SSIM | Pearson r |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.404314 ± 0.000154 | 0.561099 ± 0.000263 | 0.050479 ± 0.000358 | 0.008725 ± 0.00000400 | 0.648774 ± 0.000173 | 6.903 ± 0.005 | 41.212 ± 0.004 | 0.975089 | 0.997532 |
| 30% | 0.415606 ± 0.000048 | 0.565370 ± 0.000098 | -0.007442 ± 0.000464 | 0.008933 ± 0.00000164 | 0.658811 ± 0.000100 | 7.284 ± 0.003 | 40.980 ± 0.002 | 0.968730 | 0.997192 |
| 50% | 0.453657 ± 0.000111 | 0.615764 ± 0.000181 | -0.027495 ± 0.000491 | 0.009772 ± 0.00000224 | 0.717552 ± 0.000205 | 8.035 ± 0.003 | 40.205 ± 0.002 | 0.960460 | 0.996375 |
| 70% | 0.539421 ± 0.000211 | 0.731199 ± 0.000327 | -0.067721 ± 0.000983 | 0.011663 ± 0.00000398 | 0.851386 ± 0.000417 | 9.677 ± 0.002 | 38.680 ± 0.003 | 0.943171 | 0.994620 |
| 90% | 0.915462 ± 0.001244 | 1.258780 ± 0.001434 | -0.198319 ± 0.002239 | 0.020097 ± 0.00002186 | 1.442148 ± 0.001939 | 16.703 ± 0.019 | 33.956 ± 0.010 | 0.854987 | 0.984678 |

ERA5 在 10% 到 70% 缺失率下仍保持较高相关性；90% 缺失时误差和结构退化明显，Relative L2 约为 16.70%。PRE 与 ERA5 的 MAE、RMSE、PSNR 不能直接横向比较，应分别观察各自随缺失率变化的趋势。

### 8.3 推荐模型：ERA5 预训练 + EMA + BCE

下表为当前推荐模型 `era_bce_pretrained_ema_bs32` 的缺失区域结果，数值为三个随机种子（25、42、2026）的均值 ± 样本标准差。

| 缺失率 | MAE | RMSE | EPE | Relative L2 | PSNR (dB) | SSIM | Pearson r |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 0.329908 ± 0.000146 | 0.448523 ± 0.000209 | 0.522895 ± 0.000098 | 0.058386 ± 0.000023 | 42.963 ± 0.002 | 0.981506 ± 0.000018 | 0.998225 ± 0.000001 |
| 30% | 0.380248 ± 0.000317 | 0.517963 ± 0.000251 | 0.601923 ± 0.000514 | 0.067916 ± 0.000028 | 41.695 ± 0.004 | 0.975452 ± 0.000041 | 0.997730 ± 0.000000 |
| 50% | 0.420522 ± 0.000086 | 0.569730 ± 0.000208 | 0.665808 ± 0.000131 | 0.074370 ± 0.000033 | 40.879 ± 0.003 | 0.968073 ± 0.000006 | 0.997068 ± 0.000004 |
| 70% | 0.504320 ± 0.000516 | 0.683830 ± 0.000788 | 0.799554 ± 0.000880 | 0.088678 ± 0.000091 | 39.310 ± 0.010 | 0.953131 ± 0.000060 | 0.995736 ± 0.000007 |
| 90% | 0.835234 ± 0.001261 | 1.157107 ± 0.001482 | 1.323784 ± 0.002108 | 0.148747 ± 0.000204 | 34.766 ± 0.011 | 0.879982 ± 0.000232 | 0.988290 ± 0.000021 |

### 8.4 ERA5 训练策略与对抗损失对照

下表给出 ERA5 缺失区域、三个随机种子均值。`direct` 表示随机初始化后直接进行 GAN 训练，`pretraining + EMA` 表示先进行纯 NO 预训练，再从最佳 `generator_ema` 权重进行 GAN 微调，并在评价时使用 EMA 权重。

| 缺失率 | Direct BCE MAE | Pretrained BCE MAE | Direct RaGAN MAE | Pretrained RaGAN MAE |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 0.4043 | **0.3299** | 0.4111 | 0.3306 |
| 30% | 0.4156 | **0.3802** | 0.4907 | 0.3827 |
| 50% | 0.4537 | **0.4205** | 0.5351 | 0.4274 |
| 70% | 0.5394 | **0.5043** | 0.6173 | 0.5146 |
| 90% | 0.9155 | **0.8352** | 0.9731 | 0.8585 |

| 缺失率 | Direct BCE RMSE | Pretrained BCE RMSE | Direct RaGAN RMSE | Pretrained RaGAN RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 0.5611 | **0.4485** | 0.5551 | 0.4495 |
| 30% | 0.5654 | **0.5180** | 0.6748 | 0.5211 |
| 50% | 0.6158 | **0.5697** | 0.7372 | 0.5785 |
| 70% | 0.7312 | **0.6838** | 0.8428 | 0.6969 |
| 90% | 1.2588 | **1.1571** | 1.3450 | 1.1866 |

| 缺失率 | Direct BCE SSIM | Pretrained BCE SSIM | Direct RaGAN SSIM | Pretrained RaGAN SSIM |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 0.9751 | **0.9815** | 0.9726 | 0.9816 |
| 30% | 0.9687 | **0.9755** | 0.9643 | 0.9752 |
| 50% | 0.9605 | **0.9681** | 0.9554 | 0.9673 |
| 70% | 0.9432 | **0.9531** | 0.9372 | 0.9517 |
| 90% | 0.8550 | **0.8800** | 0.8489 | 0.8745 |

预训练 + EMA 在所有缺失率下都降低 MAE 和 RMSE，并提高 SSIM。相对随机初始化直接训练，BCE 的 MAE 降幅约为 6.51%--18.40%，RaGAN 的 MAE 降幅约为 11.78%--22.01%。预训练之后 BCE 与 RaGAN 的差距很小：五个缺失率上的平均 MAE 差异约为 1.46%，因此主要收益来自 NO 预训练和 EMA，而不是 BCE/RaGAN 形式本身。

### 8.5 ERA5 极端稀疏模型：0%--100% 随机缺失率训练

下表来自最终 `generator_ema`、完整 17 条 test 样本和 seed 25。前五档用于观察同一模型的完整退化曲线，99% 缺失率对应仅 1% 观测：

| 缺失率 | 观测率 | MAE | RMSE | EPE | Relative L2 | PSNR (dB) | SSIM | Pearson r |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 90% | 0.4182 | 0.5713 | 0.6594 | 0.0754 | 40.83 | 0.9755 | 0.9978 |
| 30% | 70% | 0.4575 | 0.6326 | 0.7223 | 0.0834 | 39.94 | 0.9658 | 0.9970 |
| 50% | 50% | 0.4929 | 0.6739 | 0.7793 | 0.0884 | 39.41 | 0.9564 | 0.9961 |
| 70% | 30% | 0.5649 | 0.7656 | 0.8942 | 0.1002 | 38.31 | 0.9409 | 0.9946 |
| 90% | 10% | 0.8003 | 1.0820 | 1.2668 | 0.1417 | 35.30 | 0.8847 | 0.9892 |
| 99% | 1% | 1.6529 | 2.2499 | 2.6139 | 0.2936 | 28.95 | 0.6522 | 0.9531 |

10%--90% 缺失率下，误差随缺失率单调上升，但相关系数始终高于 0.989。到 99% 缺失率时，Relative L2 上升到 29.36%，SSIM 降至 0.6522，说明模型仍能恢复大尺度相关结构，但细节和幅值误差已经明显，不能将该工况表述为高精度重建。

在 99% 缺失率下，最近邻基线的 MAE、RMSE 和 EPE 分别为 1.6432、2.1924 和 2.5948，略低于模型；模型的 SSIM 为 0.6522，高于最近邻的 0.5992，Pearson `r` 也从 0.9488 提高到 0.9531。也就是说，adv-NO 在极端稀疏条件下更擅长保持结构，但点值误差尚未全面超过简单插值基线。

## 9. 指标定义

所有指标都在有效缺失区域计算，`u/v` 的逐分量指标再做宏平均；EPE 直接对速度矢量计算。

$$
\begin{aligned}
MAE &= \mathrm{mean}(|\hat{x}-x|),\\
RMSE &= \sqrt{\mathrm{mean}((\hat{x}-x)^2)},\\
Relative\ L2 &= \frac{||\hat{x}-x||_2}{||x||_2},\\
Bias &= \mathrm{mean}(\hat{x}-x),\\
NRMSE &= RMSE / data\_range,\\
PSNR &= 10\log_{10}(data\_range^2/MSE),\\
EPE &= \mathrm{mean}\left(\sqrt{(\hat{u}-u)^2+(\hat{v}-v)^2}\right).
\end{aligned}
$$

SSIM 使用 7 x 7 局部窗口。窗口中心必须有效且属于缺失评价区域，窗口上下文只使用有效区域。SSIM 用于补充结构相似性，不能替代 MAE、RMSE 或 Relative L2。

推荐答辩正文重点展示：`MAE`、`RMSE`、`Relative L2`、`EPE`、`PSNR`、`SSIM`；`Bias`、`NRMSE`、`Pearson r` 可放在补充表。

## 10. 可视化结果

正式可视化不是训练时的中心 `64 × 64` patch。绘图脚本先在完整测试样本上生成稀疏掩码，再用与正式评价相同的重叠 tiled inference 覆盖整个下采样域：PRE 为 `100 × 110 × 30`，ERA5 为 `180 × 360 × 8`。图中选取 PRE 第 15/30 个 sigma 层和 ERA5 第 5/8 个时间片展示二维完整域。

每张图的四列依次为 **Ground Truth**、**Sparse Observations**、**adv-NO Reconstruction** 和 **Absolute Error (|Reconstruction - Ground Truth|)**。只有最左列保留纵坐标，只有最底行保留横坐标；场值前三列在每个分量内共享由真值确定的对称色标。PRE 陆地和其他无效区域统一显示为空白，不把无物理约束的陆地输出误认为有效重建。

PRE 图为 `18 × 8.8 in`、`300 dpi`（`5400 × 2640 px`），空间轴使用源
RHO 曲线网格经 `4×4` 聚合后的真实经纬度（约 `112.32--115.67°E`、
`20.90--23.12°N`），第五维使用 `Sigma layer`，绝对误差固定为
`0--0.12 m s^-1`。ERA5 图为 `18 × 6.8 in`、`300 dpi`（`5400 × 2040 px`），
空间轴使用真实全球经纬度 `0--360°E`、`90°S--90°N`；底层数组仍按
`90°N -> 90°S` 降序排列，保持全球场 `2:1` 比例，绝对误差固定为
`0--10 m s^-1`。场值色标标注 `Velocity (m s^-1)`，误差色标标注
`Absolute error (m s^-1)`；列标题放在图像轴上方的独立留白区域，避免遮挡数据。
同一数据集所有缺失率使用相同画布和误差色标，便于逐图对比。图中不添加测试
样本、维度、缺失率或色标策略的辅助副标题/页脚。

### PRE 最终模型

| 缺失率 | 图像 |
| ---: | --- |
| 1% | [`pre_1pct_visualization.png`](figures/pre_1pct_visualization.png) |
| 10% | [`pre_10pct_visualization.png`](figures/pre_10pct_visualization.png) |
| 30% | [`pre_30pct_visualization.png`](figures/pre_30pct_visualization.png) |
| 50% | [`pre_50pct_visualization.png`](figures/pre_50pct_visualization.png) |
| 70% | [`pre_70pct_visualization.png`](figures/pre_70pct_visualization.png) |
| 90% | [`pre_90pct_visualization.png`](figures/pre_90pct_visualization.png) |
| 99%（1% 观测） | [`pre_99pct_visualization.png`](figures/pre_99pct_visualization.png) |

### ERA5 极端稀疏模型

以下退化序列统一使用 `era_ragan_pretrained_0to100_ema_bs32` 的
`generator_ema`，不混用标准范围 BCE checkpoint。

| 缺失率 | 图像 |
| ---: | --- |
| 10% | [`era5_10pct_visualization.png`](figures/era5_10pct_visualization.png) |
| 30% | [`era5_30pct_visualization.png`](figures/era5_30pct_visualization.png) |
| 50% | [`era5_50pct_visualization.png`](figures/era5_50pct_visualization.png) |
| 70% | [`era5_70pct_visualization.png`](figures/era5_70pct_visualization.png) |
| 90% | [`era5_90pct_visualization.png`](figures/era5_90pct_visualization.png) |
| 99%（1% 观测） | [`era5_99pct_visualization.png`](figures/era5_99pct_visualization.png) |

对应 JSON 指标分别位于 PRE 的 `evaluation_final/`，以及 ERA5 的 `evaluation_all_rates/` 和 `evaluation_99pct/`。评估记录中的 `generator_weights` 均为 `generator_ema`。

## 11. 复现方法

以下命令从仓库根目录执行。预处理脚本默认使用保守下采样实现，CPU 线程数可按服务器资源调整，当前正式处理使用过 128 个 CPU 线程。

### 11.1 数据准备

```bash
cd 3_flow_reconstruction/no/adv_training

python -u prepare_sparse_data.py \
  --task A \
  --source /data/PRE_ocean_data \
  --output ../../../data/pre_uv_2t4x_conservative.h5 \
  --cpu-threads 128 \
  --batch-records 16

python -u prepare_sparse_data.py \
  --task D \
  --source /data/era_data/raw_data \
  --output ../../../data/era_uv_2t4x_conservative.h5 \
  --cpu-threads 128
```

### 11.2 持久化训练

训练 runner 会自动使用可见 GPU；指定多张卡时脚本内部使用 `DataParallel`。正式训练按“纯 NO 预训练 -> 从 NO EMA 初始化 GAN 微调”执行，每条长任务均应放在独立 tmux 会话中。

PRE 最终训练命令的核心参数为（训练缺失率 `0%--100%`）：

```bash
# Run from the repository root.

# Stage 1: pure NO + EMA
CUDA_VISIBLE_DEVICES=3,4 python -u 3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py \
  --data data/pre_uv_2t4x_conservative.h5 \
  --output-dir outputs/pre_no_pretrain_0to100_ema_bs64 \
  --patch 64 --depth 16 --batch-size 64 --epochs 500 \
  --cpu-threads 1 --train-mask-min 0.0 --train-mask-max 1.0 \
  --adversarial-weight 0 --ema-decay 0.999

# Stage 2: RaGAN fine-tuning from Stage-1 EMA
CUDA_VISIBLE_DEVICES=3,4 python -u 3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py \
  --data data/pre_uv_2t4x_conservative.h5 \
  --output-dir outputs/pre_ragan_pretrained_0to100_ema_bs64 \
  --patch 64 --depth 16 --batch-size 64 --epochs 500 \
  --cpu-threads 1 --train-mask-min 0.0 --train-mask-max 1.0 \
  --init-generator outputs/pre_no_pretrain_0to100_ema_bs64/best_model.pt \
  --adversarial-loss ragan --adversarial-weight 0.1 --ema-decay 0.999
```

ERA5 极端稀疏实验的两阶段命令为：

```bash
# Run from the repository root.

# Stage 1: pure NO + EMA, random missing rate in [0, 1]
CUDA_VISIBLE_DEVICES=1,4 python -u 3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py \
  --data data/era_uv_2t4x_conservative.h5 \
  --output-dir outputs/era_no_pretrain_0to100_ema_bs32 \
  --patch 64 --depth 8 --batch-size 32 --epochs 500 \
  --train-mask-min 0.0 --train-mask-max 1.0 \
  --adversarial-weight 0 --ema-decay 0.999

# Stage 2: RaGAN fine-tuning from Stage-1 EMA
CUDA_VISIBLE_DEVICES=1,4 python -u 3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py \
  --data data/era_uv_2t4x_conservative.h5 \
  --output-dir outputs/era_ragan_pretrained_0to100_ema_bs32 \
  --patch 64 --depth 8 --batch-size 32 --epochs 500 \
  --train-mask-min 0.0 --train-mask-max 1.0 \
  --init-generator outputs/era_no_pretrain_0to100_ema_bs32/best_model.pt \
  --adversarial-loss ragan --adversarial-weight 0.1 --ema-decay 0.999
```

启动时可先执行 `tmux new -s <session_name>`，再运行对应命令；用 `tmux attach -t <session_name>` 查看进度。预训练 checkpoint 的 `generator_ema` 是第二阶段初始化，不能用原仓库其他数据的 checkpoint 代替。

### 11.3 单个缺失率评价

```bash
python -u 3_flow_reconstruction/no/adv_training/evaluate_sparse_metrics.py \
  --data data/era_uv_2t4x_conservative.h5 \
  --checkpoint outputs/era_ragan_pretrained_0to100_ema_bs32/best_model.pt \
  --config outputs/era_ragan_pretrained_0to100_ema_bs32/config.json \
  --output /tmp/era5_metrics_50pct.json \
  --mask-ratio 0.5 --patch 64 --depth 8 --stride 32 \
  --tile-batch 8 --seed 25 --device cuda:0
```

PRE 评价将数据和 checkpoint/config 替换为 `pre_uv_2t4x_conservative.h5` 与 `pre_ragan_pretrained_0to100_ema_bs64`，并把 `--depth` 改为 16。评估脚本默认选择 `generator_ema`；只有显式传入 `--raw-generator` 才会改用非 EMA 权重。

### 11.4 批量评价注意事项

最终 PRE 和 ERA5 极端稀疏表使用 seed 25，并分别对 `--mask-ratio 0.01 0.1 0.3 0.5 0.7 0.9 0.99` 重复运行 11.3 的命令。每次评价都覆盖完整 test split，不能用训练日志中的固定 50% `test_l1` 代替多缺失率物理指标。

## 12. 结果解释与结论

1. 当前正式结果已经满足“2 倍时间、4 倍空间下采样”和 `8:1:1` 时间切分要求。
2. 保守面积加权平均比直接点抽取更适合 PRE，能够在海陆边界处排除陆地并保留粗网格平均意义。
3. PRE 最终采用“0%--100% 随机缺失率纯 NO 预训练 + RaGAN 微调 + EMA”，test normalized L1 相对纯 NO 下降约 6.76%。
4. ERA5 的纯 NO 预训练为后续 GAN 微调提供了更稳定的初始化。标准范围内，预训练 + EMA 的 BCE 模型综合误差最低；预训练后 BCE 与 RaGAN 的差距很小，主要收益来自预训练和 EMA。
5. 把 ERA5 训练缺失率扩展到 0%--100% 可以覆盖 99% 缺失测试。代价是 10%--70% 缺失时精度下降，收益是 90% 缺失时 MAE 从标准 BCE 的 0.8352 降到 0.8003。标准范围平均表现优先选 10%--90% BCE，极端稀疏优先选 0%--100% RaGAN。
6. ERA5 在 99% 缺失时仍有 Pearson `r=0.9531`，但 Relative L2 已达到 29.36%，且 MAE/RMSE 略差于最近邻。该结果说明大尺度结构尚可恢复，不代表点值重建已经足够准确。
7. PRE 的 99% 缺失率已被 `0%--100%` 训练分布覆盖，模型仍保持 SSIM 0.9672 和 Pearson `r` 0.9872；这验证了将目标极端缺失率纳入训练分布能够显著提升鲁棒性。
8. 所有训练损失、评价指标和图像都采用有效区域约束，PRE 陆地不参与模型优化、指标统计或可视化填充。

## 13. 局限性与后续工作

- 当前只完成考核任务 A 和 D，任务 B（OSTIA SST）与任务 C（Copernicus 全球流速场）尚未纳入本报告的训练结果。
- ERA5 当前没有外部海陆掩码，因此报告的是全球大气网格结果；不能把它解释为纯海洋风场结果。
- 正式 baseline 未启用 MedicalNet 感知损失。若需要验证其收益，应先实现掩码感知特征提取或对比“启用/不启用”的独立消融实验。
- ERA5 样本量比 PRE 小，虽然使用了 8 个连续时间片组成窗口，但仍应在后续工作中增加时间范围或采用更严格的跨年份测试。
- 当前评价的 SSIM 是带有效窗口约束的实现，与直接在整幅含填充值图像上计算的 SSIM 不同；不同项目之间比较时必须统一公式和评价掩码。
- 当前“预训练 + EMA”与“随机初始化直接训练”的比较同时改变了初始化策略和推理权重，不能把全部增益单独归因于预训练。若要严格分离两者，应增加“随机初始化 + EMA”或对新模型使用 `--raw-generator` 的独立消融。
- 最终 PRE 多缺失率结果目前是固定 seed 25；正式论文若需要不确定性，应再补 seed 42、2026，并报告均值与样本标准差。
- 99% 缺失率不是标准考核的主指标档位，且模型点值误差未超过最近邻基线，应将其作为压力测试而不是主结果。
- 当前输出只在缺失位置被评价；若业务场景要求同时检查已观测点的一致性，可额外报告 `all`、`observed` 和 `missing` 三种区域。

## 14. 参考文献与仓库

- Oommen et al., *Learning Turbulent Flows with Generative Models for Super Resolution and Sparse Flow Reconstruction*, arXiv:2509.08752, 2025。
- Gen4Turbulence repository: <https://github.com/vivekoommen/Gen4Turbulence>
- MedicalNet repository（原仓库可选感知特征提取器）：<https://github.com/Tencent/MedicalNet>
