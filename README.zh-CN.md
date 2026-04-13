# DPCG

<p align="center">
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  DPCG 正式开源仓库
</p>

<p align="center">
  <a href="#快速概览">快速概览</a> •
  <a href="#仓库结构">仓库结构</a> •
  <a href="#安装">安装</a> •
  <a href="#1-构建数据集">数据集</a> •
  <a href="#2-训练默认模型配置">训练</a> •
  <a href="#4-运行-gpu-benchmark">Benchmark</a>
</p>

这个仓库包含当前代码库采用的标准流程和默认配置：

1. 从带标记的 Abaqus 主 `.inp` 构建 case library
2. 将通过校验的 Abaqus 输出转换为 NPZ 样本
3. 使用当前条件数损失训练仓库中的标准模型配置
4. 将学习得到的预条件器与 PETSc GPU AMG 基线进行对比


## 快速概览

| 项目 | 说明 |
| --- | --- |
| 仓库定位 | DPCG 正式开源仓库 |
| 工作流 | Abaqus case 生成 -> NPZ 转换 -> 模型训练 -> PETSc GPU benchmark |
| 训练入口 | `configs/train/train_sunet0_tail_default.yaml` |
| Benchmark 入口 | `configs/benchmark/benchmark_petsc_gpu_sunet_vs_amg.yaml` |
| 语言切换 | 切换到 [English](./README.md) |

## 仓库结构

```text
abaqus_model/                Abaqus 主模型示例
configs/dataset/             数据集构建 / 校验 / 转换配置
configs/train/               训练配置
configs/benchmark/           Benchmark 配置
scripts/                     本地编译与 PETSc/hypre 运行辅助脚本
src/dpcg/abaqus/             对外的数据集构建入口
src/dpcg/_native/            PETSc learning operator 的本地源码与产物
src/dpcg/models.py           本仓库使用的模型定义
src/dpcg/train.py            训练 CLI
src/dpcg/benchmark.py        Benchmark CLI
tests/                       冒烟测试与形状测试
```

## 安装

核心包安装：

```bash
python -m pip install -e .
```

较重的运行时依赖请按本机环境安装：

- 支持 CUDA 的 PyTorch
- 与当前 PyTorch/CUDA 对应的 `spconv`
- 用于 GPU benchmark 路径的 `petsc4py`、PETSc、hypre
- 与当前 CUDA toolkit 对应的 `cupy`，用于 PETSc GPU learning 模式

## 1. 构建数据集

由于 Abaqus 求解发生在中间阶段，整个流程按显式阶段拆分。

先从带标记的主输入文件构建正式 case library：

```bash
python -m dpcg.abaqus.cli build \
  --dataset-id frame99-7500 \
  --baseline-type pure_frame \
  --master-inp abaqus_model/frame99-7500/master/Job-frame99-7500.inp \
  --output-root ./data \
  --num-cases 500
```

等待 Abaqus 完成生成后，再执行校验：

```bash
python -m dpcg.abaqus.cli validate \
  --dataset-root ./data/frame99-7500
```

将通过校验的结果转换为 NPZ 样本：

```bash
python -m dpcg.abaqus.cli convert \
  --dataset-root ./data/frame99-7500 \
  --output-dir ./data/frame99-7500/npz_release
```

预期输出：

- `./data/<dataset-id>/npz_release` 下的 `.npz` 样本
- 每个样本对应一个包含 split/family 元数据的 `.meta.json`
- 转换器写出的数据集级 manifest

## 2. 训练默认模型配置

使用仓库的标准训练配置：

```bash
python -m dpcg.train \
  --config configs/train/train_sunet0_tail_default.yaml \
  --model sunet0 \
  --loss cond_loss
```

预期输出：

- `best.ckpt`
- `last.ckpt`
- `split_manifest.json`
- `training_summary.json`
- `epoch_metrics.csv`

训练说明：

- 当前主目标函数是缓存谱状态的条件数损失
- `torch.linalg.cond` 保留为参考指标辅助与测试目标，不作为训练目标
- `SUNet0` 接收稀疏下三角输入，并在激活坐标上预测稀疏因子

## 3. 构建 PETSc Learning 本地库

PETSc GPU benchmark 需要 `src/dpcg/_native/libdpcg_petsc_learning.so`。

先启用 PETSc/hypre 运行包装脚本，再编译本地库：

```bash
scripts/use_petsc_hypre.sh bash scripts/build_petsc_learning_native.sh
```

预期输出：

- `src/dpcg/_native/libdpcg_petsc_learning.so`

## 4. 运行 GPU Benchmark

执行仓库中的标准 PETSc GPU 对比：

```bash
scripts/use_petsc_hypre.sh python -m dpcg.benchmark \
  --config configs/benchmark/benchmark_petsc_gpu_sunet_vs_amg.yaml \
  --model sunet0
```

标准 PETSc GPU benchmark 配置会评测 `none`、`amg`、`jacobi`、`sgs`、`ssor`、`ic0` 和 `learning`。
它使用 `PETSC_AMG_BACKEND: gamg`，数据划分为 `350 / 50 / 50`，family 配额为 `A:10, B:10, BH:30`，并将 `sunet0` 的 benchmark 模型配置设置为 `mask_percentile: 0`。

预期输出：

- `results/` 下的 benchmark CSV
- 按 split 保存到配置目录中的残差历史

## 可复现性说明

- 安装、训练和 benchmark 运行尽量使用同一个 Python 环境
- 运行 GPU benchmark 前，请按本机环境配置 PETSc、hypre、CUDA 及相关运行时变量
- PETSc GPU benchmark 调用可使用 `scripts/use_petsc_hypre.sh`，或使用你本机等价的环境包装方式

## 验证

Lint：

```bash
python -m ruff check .
```

测试：

```bash
python -m pytest -q
```
