# PRE 0%--100% 缺失率最终结果

这是 Task A（PRE 海洋 `u/v`）最终模型的正式多缺失率评价结果。

- 模型：纯 NO 预训练 → RaGAN 微调 → `generator_ema`
- 训练缺失率：`Uniform(0.0, 1.0)`
- 测试集：PRE `test` split，530 条样本
- 随机种子：25
- 评价区域：`valid_mask AND missing_mask`，只统计有效海洋缺失位置
- 输入 patch：`64 × 64 × 16`

每个 JSON 同时保存 `model`、最近邻 `nearest` 和高斯插值 `gaussian` 三种方法，包含 MAE、RMSE、Bias、Relative L2、NRMSE、PSNR、SSIM、Pearson `r` 和 EPE。报告中的主结果使用 `methods.model.missing.macro`。

配套图片展示完整 `100 × 110` 下采样域，而不是训练时的中心 `64 × 64`
patch；统一使用 `18 × 8.8 in`、`300 dpi`（`5400 × 2640 px`）画布。PRE
第五维是垂向 sigma 层，图中展示第 15/30 层并标为 `Sigma layer`；空间轴标为
`xi-grid index` 和 `eta-grid index`。四列依次为 `Ground Truth`、
`Sparse Observations`、`adv-NO Reconstruction` 和 `Absolute Error`。前三列
在每个分量内共享场值色标，u/v 绝对误差固定使用 `0--0.12 m s^-1` 色标。

| 缺失率 | MAE | RMSE | PSNR (dB) | SSIM | Pearson r |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 0.001696 | 0.003072 | 60.695 | 0.999720 | 0.999798 |
| 10% | 0.001785 | 0.003269 | 60.121 | 0.999661 | 0.999773 |
| 30% | 0.002135 | 0.003979 | 58.366 | 0.999442 | 0.999651 |
| 50% | 0.002685 | 0.005111 | 56.164 | 0.999055 | 0.999407 |
| 70% | 0.003694 | 0.007161 | 53.220 | 0.998099 | 0.998816 |
| 90% | 0.006484 | 0.012037 | 48.702 | 0.993863 | 0.996625 |
| 99% | 0.014234 | 0.023330 | 42.949 | 0.967172 | 0.987237 |

原始 JSON 文件按固定缺失率命名：`metrics_1pct.json`、`metrics_10pct.json`、`metrics_30pct.json`、`metrics_50pct.json`、`metrics_70pct.json`、`metrics_90pct.json`、`metrics_99pct.json`。

大数据集 HDF5、checkpoint 和训练日志不提交到 Git；按根目录 README 的复现命令在本地生成。
