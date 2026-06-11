# NPU Stability LLM

大模型训练稳定性监控框架的开源复现项目，基于 MindSpeed / MindSpeed-LLM 在昇腾（Ascend）NPU 上运行。

## 快速索引

- **设计文档**：`docs/plans/`
- **工作记录**：`docs/YYYY-MM-DD-worklog.md`
- **数据预处理**：`standalone_preprocess.py`
- **数据根目录**：`/data07/megatron_data/`
- **训练架构总览**：`docs/qwen3-moe-training-architecture.md` — model init, training loop, `train_step`, operator file paths, spec construction, data flow. 修改内部算子前先读这个。

## 训练脚本

两代架构，四个变体 × v1/v2 = 7 个脚本（v1 无 combined 版本）：

### v1（hidden=1024, ffn=1024, moe_ffn=1024, 128 experts, topk=8, 8 layers）
| 脚本 | 用途 |
|---|---|
| `train_shrunk_qwen3_moe.sh` | 基础训练 |
| `train_shrunk_qwen3_moe_monitor.sh` | 基础 + 稳定性监控 |
| `train_fault_injection_qwen3_moe.sh` | 基础 + 故障注入 |

### v2（hidden=1536, ffn=1536, moe_ffn=2048, 64 experts, topk=6, 16 layers）
| 脚本 | 用途 |
|---|---|
| `train_shrunk_qwen3_moe_v2.sh` | 基础训练 |
| `train_shrunk_qwen3_moe_v2_monitor.sh` | 基础 + 稳定性监控 |
| `train_fault_injection_qwen3_moe_v2.sh` | 基础 + 故障注入 |
| `train_fault_injection_qwen3_moe_v2_monitor.sh` | 基础 + 故障注入 + 监控 |

### 故障注入机制
通过环境变量控制（`bitshift` 和 `round2nearest` 互斥，每对只能启用一个）：
- `bitshift_fa_backward_O` / `round2nearest_fa_backward_O` — 注入到 FlashAttention backward 的 O（attn_out）
- `bitshift_fa_backward_dO` / `round2nearest_fa_backward_dO` — 注入到 FlashAttention backward 的 dO（grad_output）
- v1 脚本默认 bitshift=6，v2 脚本默认 bitshift=7

### 稳定性监控
- 代码：`stability_monitor/`（`monitor.py`, `integration.py`, `config.py`, `router.py`, `qk_product.py`, `spectral.py`, `tracker.py`, `jsonl_logger.py`, `plot_metrics.py`）
- 启用：`export STABILITY_MONITOR_ENABLED=1` 或 `--enable-stability-monitor`
- 输出：JSONL 文件，路径由 `STABILITY_MONITOR_OUTPUT_PATH` 环境变量或 `--monitor-output-path` 指定
- 关键 cadence：`--monitor-router-cadence 10`（默认 10），`--monitor-delta-windows 5 10 20 50 100 500 1000`（多窗口 dW，每个窗口的 cadence 等于其大小），`--monitor-w-cadence 500`（默认 500）
- 注册逻辑在 `integration.py:register_all_layers`，须注意 wrapper 展开（DDP → Float16Module → GPTModel）和属性名（`layer.mlp.router` 非 `layer.router`）
- `tracker.py` 使用 refcounting 按 step 存储权重快照，所有窗口共享同一快照存储，refcount 归零时自动释放
- `plot_metrics.py` — JSONL 可视化，每 metric 一张 per-layer 图，用法：`python stability_monitor/plot_metrics.py metrics/monitor.jsonl [--out figs/] [--metrics "router/*"] [--step-min 100] [--step-max 1000]`

### 日志与产物目录
- 基础脚本日志：`logs/train_*.log`
- 故障注入日志：`logs/fault_injection/train_v2*.log`
- 监控指标：`metrics/monitor_*.jsonl` 或 `metrics/fault_injection/monitor_*.jsonl`

## 工作惯例

- **工作记录**：每天工作结束（或完成重要里程碑）后，在 `docs/YYYY-MM-DD-worklog.md` 中创建一份带日期的工作记录，内容需包括：
  - 当日完成的工作
  - 关键代码/数据改动
  - 输出产物及其存放位置
  - 遇到的阻塞或下一步计划
