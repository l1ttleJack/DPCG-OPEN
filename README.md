# DPCG

<p align="center">
  <strong>English</strong> | <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  Official open-source repository for DPCG
</p>

<p align="center">
  <a href="#at-a-glance">At a Glance</a> •
  <a href="#repository-layout">Layout</a> •
  <a href="#installation">Installation</a> •
  <a href="#1-build-the-dataset">Dataset</a> •
  <a href="#2-train-the-default-model-configuration">Training</a> •
  <a href="#4-run-the-gpu-benchmark">Benchmark</a>
</p>

This repository contains the standard workflow and the default configuration used in this codebase:

1. build an Abaqus case library from a marked master `.inp`
2. convert validated Abaqus outputs to NPZ samples
3. train the repository's standard model configuration with the current condition-number loss
4. compare the learned preconditioner against the PETSc GPU AMG baseline


## At a Glance

| Item | Description |
| --- | --- |
| Positioning | Official DPCG open-source repository |
| Workflow | Abaqus case generation -> NPZ conversion -> training -> PETSc GPU benchmark |
| Training entry | `configs/train/train_sunet0_tail_default.yaml` |
| Benchmark entry | `configs/benchmark/benchmark_petsc_gpu_sunet_vs_amg.yaml` |
| Language | Switch to [简体中文](./README.zh-CN.md) |

## Repository Layout

```text
abaqus_model/                Example master Abaqus models
configs/dataset/             Dataset build / validate / convert configs
configs/train/               Training configs
configs/benchmark/           Benchmark configs
scripts/                     Runtime helpers for native build and PETSc/hypre env
src/dpcg/abaqus/             Public dataset-building entrypoints
src/dpcg/_native/            Native PETSc learning operator source / output
src/dpcg/models.py           Model definitions used by this repository
src/dpcg/train.py            Training CLI
src/dpcg/benchmark.py        Benchmark CLI
tests/                       Smoke and shape tests
```

## Installation

Core package install:

```bash
python -m pip install -e .
```

Install the heavy runtime dependencies to match your machine:

- CUDA-enabled PyTorch
- `spconv` matched to the installed PyTorch/CUDA stack
- `petsc4py`, PETSc, and hypre for the GPU benchmark path
- `cupy` matched to the installed CUDA toolkit for PETSc GPU learning mode

## 1. Build the Dataset

The Abaqus workflow is split into explicit phases because Abaqus solves happen between them.

Build the formal case library from a marked master input:

```bash
python -m dpcg.abaqus.cli build \
  --dataset-id frame99-7500 \
  --baseline-type pure_frame \
  --master-inp abaqus_model/frame99-7500/master/Job-frame99-7500.inp \
  --output-root ./data \
  --num-cases 500
```

After Abaqus finishes the generated cases, validate them:

```bash
python -m dpcg.abaqus.cli validate \
  --dataset-root ./data/frame99-7500
```

Convert the validated cases to NPZ samples:

```bash
python -m dpcg.abaqus.cli convert \
  --dataset-root ./data/frame99-7500 \
  --output-dir ./data/frame99-7500/npz_release
```

Expected outputs:

- `.npz` samples under `./data/<dataset-id>/npz_release`
- one `.meta.json` per sample with split/family metadata
- a dataset-level manifest written by the converter

## 2. Train the Default Model Configuration

Use the repository's standard training config:

```bash
python -m dpcg.train \
  --config configs/train/train_sunet0_tail_default.yaml \
  --model sunet0 \
  --loss cond_loss
```

Expected outputs:

- `best.ckpt`
- `last.ckpt`
- `split_manifest.json`
- `training_summary.json`
- `epoch_metrics.csv`

Training notes:

- the primary objective is the currently used cached spectral condition-number loss
- `torch.linalg.cond` is kept as a reference metric helper and test target, not as the training objective
- `SUNet0` consumes sparse lower-triangular inputs and predicts a sparse factor on active coordinates

## 3. Build the Native PETSc Learning Library

The PETSc GPU benchmark requires `src/dpcg/_native/libdpcg_petsc_learning.so`.

Use the PETSc/hypre runtime wrapper, then build the sidecar library:

```bash
scripts/use_petsc_hypre.sh bash scripts/build_petsc_learning_native.sh
```

Expected output:

- `src/dpcg/_native/libdpcg_petsc_learning.so`

## 4. Run the GPU Benchmark

Run the repository's standard PETSc GPU comparison:

```bash
scripts/use_petsc_hypre.sh python -m dpcg.benchmark \
  --config configs/benchmark/benchmark_petsc_gpu_sunet_vs_amg.yaml \
  --model sunet0
```

The standard PETSc GPU benchmark config evaluates `none`, `amg`, `jacobi`, `sgs`, `ssor`, `ic0`, and `learning`.
It uses `PETSC_AMG_BACKEND: gamg`, the dataset split counts `350 / 50 / 50`, family quotas `A:10, B:10, BH:30`, and sets the `sunet0` benchmark model to `mask_percentile: 0`.

Expected outputs:

- benchmark CSV under `results/`
- per-split residual histories under the configured residual directory

## Reproducibility Notes

- use the same Python environment for install, training, and benchmark runs
- configure PETSc, hypre, CUDA, and related runtime variables for your local machine before running the GPU benchmark
- use `scripts/use_petsc_hypre.sh` or an equivalent local environment wrapper for PETSc GPU benchmark invocations

## Verification

Lint:

```bash
python -m ruff check .
```

Tests:

```bash
python -m pytest -q
```
