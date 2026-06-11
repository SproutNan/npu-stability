# Auto Pipeline

## Purpose

This folder provides a simple batch workflow for:

1. Running selected short-run scene directories
2. Running profiling analysis on the generated outputs
3. Running selected long-run loss scene directories

The intended order is:

- short-run first
- profiling analysis second
- loss runs last

## Main Entry Scripts

### `prepare_pipeline_workspace.sh`

This is the workspace bootstrap entrypoint.

It is responsible for:

- cleaning old workspace directories
- downloading code
- installing dependencies
- calling `start_pipeline_tasks.sh`

### `start_pipeline_tasks.sh`

This is the orchestration entrypoint.

It is responsible for:

- selecting which short-run scene directories to execute
- running profiling analysis after short-run profiling outputs exist
- selecting which loss scene directories to execute

### `run_training_pipeline.sh`

This is the per-scene executor.

It is responsible for:

- copying `MindSpeedRun/scripts` into `MindSpeed-LLM/scripts`
- detecting target parallelism from device type and IP count
- running one scene directory at a time
- writing logs and profile outputs into the requested log root

## Simple Usage

### 1. Bootstrap the full workflow

```bash
bash MindSpeedRun/scripts/auto_pipeline/prepare_pipeline_workspace.sh \
  --ips "141.62.24.228" \
  --inet-prefix 141 \
  --confirm-cleanup true
```

### 2. Run the scheduler directly

```bash
bash MindSpeedRun/scripts/auto_pipeline/start_pipeline_tasks.sh \
  --ips "141.62.24.228" \
  --inet-prefix 141
```

## Required Parameters

Both `prepare_pipeline_workspace.sh` and `start_pipeline_tasks.sh` require:

- `--ips "ip1 ip2 ..."`
- `--inet-prefix 141`

## Optional Parameters

### `--world-size`

```text
auto | 8p | 32p | 64p
```

Default is `auto`.

This is used as an explicit expected world size. It is forwarded into the scene executor and checked against the current device and IP-count constraints.

### `--short-scenes`

Comma-separated short-run scene directory list.

Example:

```bash
--short-scenes "deepseek3_swap,longcat_swap,qwen3_235b_swap"
```

Default:

```text
deepseek3_swap,longcat_swap,qwen3_235b_swap
```

### `--loss-scenes`

Comma-separated long-run loss scene directory list.

Example:

```bash
--loss-scenes "deepseek3_noswap_8p_loss,longcat_loss,qwen3_235b_loss"
```

Default:

```text
deepseek3_noswap_8p_loss,longcat_loss,qwen3_235b_loss
```

### `--workspace-dir`

Only used by `prepare_pipeline_workspace.sh`.

Default: current directory.

### `--log-dir`

Only used by `start_pipeline_tasks.sh`.

Default:

```text
<pipeline_dir>/logs/<DEVICE>/<MM-DD-HH-MM>
```

### `--skip-analysis`

Only used by `start_pipeline_tasks.sh`.

Default: `false`.

If set to `true`, the profiling analysis phase is skipped.

## Cleanup Safety

`prepare_pipeline_workspace.sh` will delete:

- `MindSpeed`
- `Megatron-LM`
- `MindSpeed-LLM`
- `/root/.gitconfig`

It will not continue unless you pass:

```bash
--confirm-cleanup true
```

This is required to make destructive behavior explicit.

## Production Notes

Current production-oriented design choices:

- explicit CLI instead of hardcoded `ips="XXX"`
- explicit short-run and loss scene lists instead of hidden defaults in the middle of the script
- explicit cleanup confirmation for destructive setup
- phase-oriented scheduler: short-run -> analysis -> loss
- log directory and workspace directory can be overridden
- scene execution is directory-driven, which makes future scene additions easier to maintain

## Current Limitations

- `run_training_pipeline.sh` still keeps legacy compatibility logic for non-qwen scene directories that use old standalone scripts
- `prepare_pipeline_workspace.sh` still performs environment installation inline; that is practical, but it is not yet split into smaller reusable functions
- some scene scripts outside `auto_pipeline` are still legacy-style wrappers rather than centralized launchers