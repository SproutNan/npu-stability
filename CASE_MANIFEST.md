# v4 Fault Trial Case Manifest

This is a stripped local snapshot of the remote workspace:

```text
/data02/npu_stablity/LLM
```

It keeps the code needed to reproduce the v4 training flow, plus only two top-level launch scripts: the latest baseline script and the latest fault-injection script. Logs, metrics, plots, caches, git metadata, archives, data, old launch scripts, sweeps, and preprocessing helpers are removed.

## Source Log

```text
/data02/npu_stablity/LLM/logs/v4_fault/train_v4_bsO7_bsdO7_lr5.0e-5_iters30000_20260609_233819.log
```

The log maps to this single-trial launch:

```bash
bash train_qwen3_moe_v4_fault.sh \
  --lr 5.0e-5 \
  --train-iters 30000 \
  --bits-o 7 \
  --bits-do 7
```

The original run was part of a sweep whose grid used:

```text
TRAIN_ITERS=30000
LRS=(5.0e-5 2.0e-4 5.0e-4 2.0e-3 5.0e-3)
BITS=(0 7)
```

## Dataset Identified

The trial used a preprocessed Megatron mmap dataset prefix:

```text
/data07/megatron_data/fineweb-edu_100BT
```

At runtime, the target machine must have:

```text
/data07/megatron_data/fineweb-edu_100BT.bin
/data07/megatron_data/fineweb-edu_100BT.idx
```

The log confirms:

```text
data_path=['/data07/megatron_data/fineweb-edu_100BT']
dataset_impl=mmap
split=100,0,0
```

Data movement is intentionally not included in this snapshot.

## Included

```text
MindSpeed-LLM/
MindSpeed/
Megatron-LM/
MindSpeedRun-llm0121_gitcode/
stability_monitor/
docs/
*.md
.gitignore
CASE_MANIFEST.md
```

Top-level launch scripts kept:

```text
train_qwen3_moe_v4.sh
train_qwen3_moe_v4_fault.sh
```

Key code paths for this trial:

```text
MindSpeed-LLM/pretrain_gpt.py
stability_monitor/
```

## Removed

```text
logs/
metrics/
monitor_*_plots/
.git/
__pycache__/
.pytest_cache/
.mypy_cache/
.cache/
*.log
*.tar.gz
*.zip
.DS_Store
old top-level train_*.sh scripts
run_sweep*.sh scripts
top-level preprocessing / download / parser helpers
```

## Current Script Defaults

The launch scripts are currently configured for the NPU training machine layout:

```text
/opt/tiger/npu-stability/MindSpeed-LLM/pretrain_gpt.py
/mnt/hdfs/training_data/Qwen3-14B
/mnt/hdfs/training_data/fineweb-edu_100BT
```

Distributed launch uses:

```text
MASTER_PORT=1234
```

If you place the repository or data elsewhere, update `train_qwen3_moe_v4.sh` and `train_qwen3_moe_v4_fault.sh`.

## Source Version Notes

Observed on `250-yipei`:

```text
MindSpeed-LLM: f50744c80
MindSpeed:     3f559970
Megatron-LM:  1620401
```

The copied snapshot excludes `.git`, so these are reference hashes only.
