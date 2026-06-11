# Baseline → Current 变更总结

Baseline commit: `bd931b3` (2026-04-21) — 实验脚本与配置的初始导入，共 9 个文件 / 1223 行。
对比范围: `bd931b3..HEAD` (7 个 commits, 新增 3449 行, 删除 16 行, 新增 20 个文件).

---

## 1. 数据预处理 — Parquet 输入支持

**文件**: `standalone_preprocess.py` (+100 行)

- 新增 `pyarrow.parquet` 依赖，支持三种输入格式（自动检测）:
  - **JSONL** (原有): `_iter_texts_jsonl()` — 逐行解析
  - **单个 Parquet 文件**: `_iter_texts_parquet_file()` — 按 batch 读取指定列
  - **Parquet 目录**: `_iter_texts_parquet_dir()` — 遍历 `*.parquet`，合并所有文件
- Tokenizer 调用增加容错: 单个文档编码失败时 fallback 到逐条编码，不再丢弃整批
- CLI `--input` 参数 help text 更新 (支持 parquet 目录)
- 变量重命名: `JSONL_PATH` → `INPUT_PATH`

---

## 2. v1 训练脚本配置变更

**文件**: `train_shrunk_qwen3_moe.sh` (更新)

| 配置项 | Baseline | 当前 |
|---|---|---|
| 可见 NPU | 8-15 | 0-7 |
| 数据集 | `qwen3_train_text_document` | `fineweb-edu_100BT` |
| 专家数 | 160 | **128** |
| TopK | 2 | **8** |
| FFN hidden | 1024 (不变) | 1024 |
| MBS | 1 | **8** |
| GBS | 16 | **64** |
| Train iters | 1000 | **10000** |
| `--fix-router` | 启用 | 注释掉 (允许路由学习) |
| `--qk-layernorm` | 无 | **新增** |

---

## 3. 新增 v2 训练脚本

v2 模型扩大了 hidden 和 FFN 维度，专家数减半、TopK 减小:

| 配置项 | v1 | v2 |
|---|---|---|
| HIDDEN_SIZE | 1024 | **1536** |
| FFN_HIDDEN_SIZE | 1024 | **1536** |
| MOE_FFN_HIDDEN_SIZE | 1024 | **2048** |
| NUM_EXPERTS | 128 | **64** |
| MOE_ROUTER_TOPK | 8 | **6** |
| NUM_LAYERS | 16 (不变) | 16 |
| MBS | 8 | **6** |

新增脚本:
- `train_shrunk_qwen3_moe_v2.sh` — v2 基础训练
- `train_shrunk_qwen3_moe_v2_monitor.sh` — v2 + 稳定性监控

---

## 4. 故障注入系统

**机制**: 通过环境变量在 FlashAttention backward 的 O (attn_out) 和 dO (grad_output) 注入精度故障。
- `bitshift`: 清零低 N 位尾数 (模拟 stuck-at-0)
- `round2nearest`: 舍入到 N 位尾数可表示的最接近值
- 每对互斥 (bitshift 和 round2nearest 不能同时为非零值)
- 支持通过命令行参数选择 NPU 范围 (`first_8_rank` / `last_8_rank`)

最新提交 (37540af) 将默认 bitshift 提高到 15，并将 echo 语句改为直接变量引用以修复位运算识别问题.

新增脚本:
- `train_fault_injection_qwen3_moe.sh` — v1 + 故障注入
- `train_fault_injection_qwen3_moe_v2.sh` — v2 + 故障注入
- `train_fault_injection_qwen3_moe_v2_monitor.sh` — v2 + 故障注入 + 监控 (组合)

---

## 5. 稳定性监控模块

**目录**: `stability_monitor/` (10 个文件, ~1350 行)

### 模块结构

| 文件 | 职责 |
|---|---|
| `monitor.py` | 主监控器，管理注册层、snapshot 定时采集、指标计算调度 |
| `config.py` | 监控配置 (cadence、窗口大小、输出路径、启用标志) |
| `integration.py` | 与 MindSpeed-LLM 集成, `register_all_layers()` 遍历模型层注册 hook |
| `tracker.py` | 权重快照存储，基于 refcounting 按 step 管理，所有窗口共享存储 (refcount 归零自动释放) |
| `router.py` | Router weight 指标: `router_weight_frob_norm`, `router_weight_max_abs`, `router_weight_centered_frob_sq` |
| `qk_product.py` | QK product 监控: `qk_product_frob_norm`, `qk_product_max_abs` |
| `spectral.py` | 谱分析: `weight_spectral_norm`, `weight_stable_rank` |
| `jsonl_logger.py` | JSONL 格式输出 (step, metric_name, layer, value, timestamp) |
| `plot_metrics.py` | 指标可视化 |

### 关键指标

- **Delta-W (多窗口)**: 窗口大小 [5, 10, 20, 50, 100, 500, 1000]，每个窗口 cadence = 窗口大小
- **W_c Frobenius 能量** (`router_weight_centered_frob_sq`): 最新提交 (53eca4e)，减去均值后的 Frobenius 平方
- **QK product**: Frobenius 范数和最大绝对值
- **Spectral**: 谱范数 + Stable Rank

### 激活方式
```bash
export STABILITY_MONITOR_ENABLED=1
# 或
--enable-stability-monitor --monitor-output-path /path/to/output.jsonl
```

### 关键实现细节
- Hook 注册须穿透 wrapper 层 (DDP → Float16Module → GPTModel)
- Router 属性路径: `layer.mlp.router` (非 `layer.router`)
- Tracker 使用 refcounting，同一 step 快照被多个窗口共享

---

## 6. 新增 v1 监控脚本

- `train_shrunk_qwen3_moe_monitor.sh` — v1 基础训练 + 稳定性监控

---

## 7. 设计文档

- `docs/plans/2026-04-28-stability-monitor-design.md` — 稳定性监控详细设计文档
- `docs/qwen3-moe-training-architecture.md` / `docs/qwen3-moe-training-architecture-v2.md` — 训练架构总览

---

## 提交时间线

```
37540af feat: bump fault injection bitshift defaults and fix echo to use direct variable references
53eca4e feat: add router_weight_centered_frob_sq metric for W_c Frobenius energy tracking
c209d49 feat: replace single delta-W window with multi-window collection
2a2f713 feat: add stability monitor module
7dbde46 feat: add v2 training scripts (basic, monitor, fault injection, combined)
91d3fdf feat: add v1 monitor and fault injection training scripts
e8dd6dc feat: add parquet input support to preprocess, update v1 training config
bd931b3 baseline: experiment scripts and configs   ← 起点
```

---

## 总计

| 维度 | 数量 |
|---|---|
| 新增文件 | 20 |
| 修改文件 | 2 |
| 新增代码行 | 3449 |
| 追踪提交 | 7 |
