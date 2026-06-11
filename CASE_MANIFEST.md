# NPU Stability v4 Case

This repository is a minimal runnable snapshot for the Qwen3-MoE v4 NPU
stability experiment. It keeps the open-source MindSpeed training path, the
local stability monitor, and the launch scripts needed for single-node and
8-node runs.

## Kept

```text
MindSpeed/
MindSpeed-LLM/
stability_monitor/
setup.sh
train_qwen3_moe_v4.sh
train_qwen3_moe_v4_multinode.sh
debug_multinode_probe.sh
run_v4_fault_50000_o8_do8.sh
```

The old standalone `Megatron-LM/` copy, `MindSpeedRun-llm0121_gitcode/`,
worklog/design docs, generated logs, metrics, caches, and obsolete launch
wrappers were removed. `MindSpeed-LLM/` already carries the Megatron sources
used by `pretrain_gpt.py`.

## Setup

```bash
bash setup.sh
```

The setup script sources the Ascend toolkit and installs MindSpeed:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r MindSpeed/requirements.txt
pip install -e MindSpeed
```

## Single-Node Entry

Use one script for both baseline and fault-injection runs:

```bash
bash train_qwen3_moe_v4.sh
```

Common overrides:

```bash
bash train_qwen3_moe_v4.sh --lr 3.0e-4 --aux-loss 0.002
bash train_qwen3_moe_v4.sh --bits-o 7 --bits-do 7
bash train_qwen3_moe_v4.sh --train-iters 50000 --bits-o 8 --bits-do 8
```

The script defaults to:

```text
DATA_PATH=/mnt/hdfs/training_data/fineweb-edu_100BT
TOKENIZER_PATH=/mnt/hdfs/training_data/Qwen3-14B
TRAIN_ITERS=30000
MASTER_PORT=1234
```

Logs and stability-monitor metrics for both baseline and fault runs are grouped
together:

```text
logs/v4/
metrics/v4/
```

The filename tag distinguishes the experiment:

```text
v4_baseline_...
v4_bsO7_bsdO7_...
```

## Original Trial Mapping

The source trial log was:

```text
/data02/npu_stablity/LLM/logs/v4_fault/train_v4_bsO7_bsdO7_lr5.0e-5_iters30000_20260609_233819.log
```

It maps to the current unified command:

```bash
bash train_qwen3_moe_v4.sh \
  --lr 5.0e-5 \
  --train-iters 30000 \
  --bits-o 7 \
  --bits-do 7
```

## 8-Node Entry

The multi-node baseline Arnold entry is:

```bash
bash train_qwen3_moe_v4_multinode.sh
```

It expects Arnold to provide:

```text
ARNOLD_EXECUTOR_NUM
ARNOLD_ID
```

The script resolves node0 IPv4 through a shared marker under
`/mnt/hdfs/__INFRA_OUTPUT__/npu_debug/` when platform master variables resolve
to IPv6.

For multi-node diagnosis, run:

```bash
bash debug_multinode_probe.sh
```

The probe records per-node environment, resolves node0 IPv4 through the shared
HDFS output directory, then runs HCCL barrier, all-reduce, all-to-all, and
short training stages with timeout-protected logs.

## Data

Data is intentionally not part of this repository. The expected HDFS-mounted
dataset prefix is:

```text
/mnt/hdfs/training_data/fineweb-edu_100BT
```

The tokenizer path is:

```text
/mnt/hdfs/training_data/Qwen3-14B
```

The observed Megatron dataset build for `fineweb-edu_100BT` reported:

```text
total samples: 24,329,872
sequence length: 4096
```

With the single-node defaults:

```text
GBS=384
TRAIN_ITERS=30000
required samples = 30,000 x 384 = 11,520,000
required tokens  = 11,520,000 x 4096 = 47,185,920,000
```

So the dataset supports the default run without wrapping:

```text
max full-epoch iterations = floor(24,329,872 / 384) = 63,359
default usage = 47.35% of available samples
```

For the 50,000-step `O8/dO8` trial:

```text
required samples = 50,000 x 384 = 19,200,000
required tokens  = 19,200,000 x 4096 = 78,643,200,000
usage = 78.92% of available samples
```

This is still below the observed maximum of 63,359 single-node iterations.
