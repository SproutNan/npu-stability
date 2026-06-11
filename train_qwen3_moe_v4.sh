#!/bin/bash
set -euo pipefail

# ============================================================================
# v4 baseline training, Qwen3-MoE
# ----------------------------------------------------------------------------
# Hardware: 16 × Ascend 910B (64 GB HBM each), single node
# Model:    ~9.6B total / ~1.2B active (20L × hidden 1536, 64 experts top-4)
# Stability monitor: enabled by default
# Fault injection:   NOT supported in this script (clean baseline)
#
# Usage:
#   bash train_qwen3_moe_v4.sh                       # defaults: lr=5e-4, 10k iters
#   bash train_qwen3_moe_v4.sh --lr 3.0e-4 --train-iters 20000
# ============================================================================

# ============================================================================
# Default values
# ============================================================================
LR=5.0e-4
TRAIN_ITERS=10000

# ============================================================================
# Argument parsing (MUST run before any derived values)
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr)           LR="$2"; shift 2 ;;
        --train-iters)  TRAIN_ITERS="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $0 [--lr RATE] [--train-iters N]"
            exit 1
            ;;
    esac
done

# Derived values — placed AFTER argparse so --lr propagates to MIN_LR.
# (Fixes the v3/alignedv1 bug where MIN_LR was frozen at the default LR.)
MIN_LR=$(awk -v l="$LR" 'BEGIN {printf "%.10f", l/10}')

# ============================================================================
# Hardware: use all 16 NPUs (Path B — single experiment per launch)
# ============================================================================
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
NPUS_PER_NODE=16

# ============================================================================
# Standard Ascend / runtime environment
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export STREAMS_PER_DEVICE=32

# Stability monitor toggle (consumed by stability_monitor/integration.py)
export STABILITY_MONITOR_ENABLED=1

# ============================================================================
# Distributed launch
# ============================================================================
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# ============================================================================
# Data & tokenizer
# ============================================================================
DATA_PATH="/data07/megatron_data/fineweb-edu_100BT"
TOKENIZER_PATH="/data02/npu_stablity/mojo_opset/ckpts/Qwen3-14B"

# ============================================================================
# Parallelism
#   TP=1, PP=1, CP=1, EP=8 over 16 NPUs gives:
#     - shared-DP = world / TP / PP / CP = 16
#     - expert-DP = world / EP / TP / PP / CP = 2  (each expert replicated 2×)
#   All-to-all width stays at EP=8 (doesn't blow up with the extra ranks).
# ============================================================================
TP=1
PP=1
EP=8
CP=1

# ============================================================================
# Model architecture (~9.6B total / ~1.2B active)
#   20 layers × hidden 1536, head_dim=128 (12 heads), GQA 3:1
#   64 experts, top-4, MoE FFN=1536 (1.0× hidden per expert)
# ============================================================================
NUM_LAYERS=20
HIDDEN_SIZE=1536
NUM_ATTENTION_HEADS=12          # head_dim = 1536 / 12 = 128
NUM_QUERY_GROUPS=4              # GQA 3:1
FFN_HIDDEN_SIZE=1536            # unused (all-MoE), set for clarity

NUM_EXPERTS=64
MOE_ROUTER_TOPK=4
MOE_FFN_HIDDEN_SIZE=1536

# ============================================================================
# Batch
#   GBS=384, MBS=12, shared-DP=16  →  num_micro_batches = 384/(12×16) = 2
#   tokens/step = 384 × 4096 = 1,572,864  (~1.57M)
#   per-expert throughput (ideal) = 1.57M × topk / num_experts ≈ 98k tokens/step
# ============================================================================
SEQ_LENGTH=4096
MBS=6
GBS=384

# ============================================================================
# Argument groups
# ============================================================================
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts $NUM_EXPERTS \
    --moe-router-topk $MOE_ROUTER_TOPK \
    --moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.001 \
    --moe-z-loss-coeff 1e-3 \
    --norm-topk-prob \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-layer-freq -1 \
    --first-k-dense-replace 0 \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --transformer-impl local \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --qk-layernorm \
    --attention-softmax-in-fp32 \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights
"
# NOTE: --qk-layernorm is REQUIRED for the qwen3_spec to wire PTNorm on q/k;
#       without it, the spec defaults to IdentityOp and qk-norm is silently OFF.
#       See MindSpeed-LLM/mindspeed_llm/tasks/models/spec/qwen3_spec.py:25,48-49.

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --expert-model-parallel-size $EP \
    --context-parallel-size $CP \
    --expert-tensor-parallel-size 1 \
    --attention-mask-type causal
"

TRAIN_ARGS="
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 10000 \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.03 \
    --accumulate-allreduce-grads-in-fp32 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16 \
    --seed 42
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage
"

OPTIMIZE_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --manual-gc \
    --manual-gc-interval 50
"

# ============================================================================
# Logging
# ============================================================================
LOG_DIR="logs/v4"
METRICS_DIR="metrics/v4"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="v4_lr${LR}_iters${TRAIN_ITERS}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/train_${TAG}.log"
METRICS_FILE="${METRICS_DIR}/monitor_${TAG}.jsonl"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"
export STABILITY_MONITOR_OUTPUT_PATH="${METRICS_FILE}"

# Derived display values
HEAD_DIM=$((HIDDEN_SIZE / NUM_ATTENTION_HEADS))
TOKENS_PER_STEP=$((GBS * SEQ_LENGTH))
SHARED_DP=$((NPUS_PER_NODE / TP / PP / CP))
EXPERT_DP=$((NPUS_PER_NODE / EP / TP / PP / CP))

# ============================================================================
# Banner
# ============================================================================
echo "============================================"
echo "[INFO] v4 baseline training (Qwen3-MoE)"
echo "  model:        ${NUM_LAYERS}L × hidden=${HIDDEN_SIZE}, head_dim=${HEAD_DIM} (heads=${NUM_ATTENTION_HEADS}, kv_groups=${NUM_QUERY_GROUPS})"
echo "  MoE:          ${NUM_EXPERTS} experts, top-${MOE_ROUTER_TOPK}, expert_ffn=${MOE_FFN_HIDDEN_SIZE}"
echo "  parallelism:  TP=${TP} PP=${PP} EP=${EP} CP=${CP}  (shared-DP=${SHARED_DP}, expert-DP=${EXPERT_DP})"
echo "  batch:        MBS=${MBS}, GBS=${GBS}, tokens/step=${TOKENS_PER_STEP}"
echo "  schedule:     lr=${LR} → min_lr=${MIN_LR}, warmup=3%, iters=${TRAIN_ITERS}"
echo "  stability:    qk-norm ON, z-loss=1e-3, aux-loss=1e-3, norm-topk-prob ON"
echo "  monitor:      enabled, windows=[10, 50, 200, 500]"
echo "  log:          ${LOG_FILE}"
echo "  metrics:      ${METRICS_FILE}"
echo "============================================"

# ============================================================================
# Launch
# ============================================================================
cmd=(
    python -m torch.distributed.launch
    $DISTRIBUTED_ARGS
    /data02/npu_stablity/LLM/MindSpeed-LLM/pretrain_gpt.py
    $GPT_ARGS
    $MODEL_PARALLEL_ARGS
    $MOE_ARGS
    $TRAIN_ARGS
    $DATA_ARGS
    $OUTPUT_ARGS
    $OPTIMIZE_ARGS
    --monitor-delta-windows 10 50 200 500 \
    --distributed-backend nccl
)

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
