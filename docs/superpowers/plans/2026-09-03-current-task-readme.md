# Current Task README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository root README a Chinese-first, reproducible entry point for the completed PRE and ERA5 sparse adv-NO tasks.

**Architecture:** Keep the root README concise enough for first contact while linking to the task-level operating guide and full report for detail. Source every command and result from the checked-in implementation/report, preserve upstream attribution, and publish no local data or checkpoints.

**Tech Stack:** Markdown, Python/PyTorch CLI documentation, Git link validation

---

### Task 1: Replace The Upstream-First README

**Files:**
- Modify: `README.md`
- Reference: `3_flow_reconstruction/no/adv_training/README_sparse_tasks.md`
- Reference: `docs/PROJECT_REPORT.md`

- [ ] **Step 1: Record the pre-change acceptance failure**

Run:

```bash
ruby -e 's=File.read("README.md"); required=["PRE", "ERA5", "2 倍时间", "4 倍空间", "8:1:1", "generator_ema", "99%", "数据与权重"]; missing=required.reject{|x| s.include?(x)}; abort("missing: #{missing.join(", ")}") unless missing.empty?'
```

Expected: nonzero exit with multiple missing Chinese task-entry terms.

- [ ] **Step 2: Rewrite the root README**

Replace `README.md` with these sections and facts:

```text
# adv-NO 稀疏流场重建：PRE 与 ERA5
项目定位 and upstream attribution
## 已完成任务
Task A and Task D status; B/C explicitly out of scope
## 方法概览
2x temporal averaging; conservative 4x4 spatial averaging; 8:1:1 chronological split;
four-channel sparse input; two-channel u/v output; NO pretraining -> GAN fine-tuning -> EMA inference;
PRE valid-mask handling; fixed missing-region metrics
## 项目结构
links to all public task scripts, task README, report, and figures
## 环境安装
Python version and pip install command
## 数据准备
repository-relative PRE and ERA5 commands from README_sparse_tasks.md
## 两阶段训练
final PRE commands and both ERA5 standard/extreme configuration summaries, linking detailed commands
## 推理、指标与可视化
one evaluator command and one visualization command using documented flags
## 核心结果
PRE final 10%-90% table; ERA5 standard BCE summary; ERA5 0%-100% six-rate table;
honest warnings for both 99% cases
## 数据与权重
explicit statement that data/output/checkpoints are gitignored and must be generated/provided locally
## 上游项目与引用
official repository, paper, BibTeX, license, acknowledgments
```

Use `docs/PROJECT_REPORT.md` values verbatim. Do not add unsupported claims or server-specific personal paths.

- [ ] **Step 3: Run README acceptance checks**

Run:

```bash
ruby -e 's=File.read("README.md"); required=["PRE", "ERA5", "2 倍时间", "4 倍空间", "8:1:1", "generator_ema", "99%", "数据与权重"]; missing=required.reject{|x| s.include?(x)}; abort("missing: #{missing.join(", ")}") unless missing.empty?; puts "README content checks passed"'
```

Expected: `README content checks passed`.

Run:

```bash
ruby -e 'doc="README.md"; base=File.dirname(doc); missing=File.read(doc).scan(/\]\(([^)]+)\)/).flatten.reject{|p| p.start_with?("http") || p.start_with?("<http")}.reject{|p| File.exist?(File.expand_path(p,base))}; abort("missing links: #{missing.join(", ")}") unless missing.empty?; puts "README links valid"'
```

Expected: `README links valid`.

Run:

```bash
rg -n '/data1/user|/home/lz|smoke|test_' README.md
```

Expected: no output and exit code 1.

- [ ] **Step 4: Verify documented CLIs**

Run:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python 3_flow_reconstruction/no/adv_training/prepare_sparse_data.py --help
/home/lz/miniconda3/envs/pytorch/bin/python 3_flow_reconstruction/no/adv_training/train_sparse_adv_no.py --help
/home/lz/miniconda3/envs/pytorch/bin/python 3_flow_reconstruction/no/adv_training/evaluate_sparse_metrics.py --help
/home/lz/miniconda3/envs/pytorch/bin/python 3_flow_reconstruction/no/adv_training/plot_sparse_visualization.py --help
```

Expected: all four commands exit zero and list every flag used by the README examples.

- [ ] **Step 5: Commit the README**

```bash
git add README.md
git commit -m "docs: make sparse reconstruction the project entry point"
```

Expected: one documentation commit containing only `README.md`.

