#!/bin/bash
set -euo pipefail

# ============================================================================
# v4 single-node training, Qwen3-MoE
# ----------------------------------------------------------------------------
# Hardware: 16 x Ascend 910B, single node
# Model:    ~9.6B total / ~1.2B active (20L x hidden 1536, 64 experts top-4)
# Monitor:  enabled by default; JSONL is written by global rank 0 only.
#
# Usage:
#   bash train_qwen3_moe_v4.sh
#   bash train_qwen3_moe_v4.sh --lr 3.0e-4 --aux-loss 0.002
#   bash train_qwen3_moe_v4.sh --bits-o 7 --bits-do 7
# ============================================================================

# ============================================================================
# User config: edit this block for the common experiments.
# ============================================================================
LR=${LR:-5.0e-4}
TRAIN_ITERS=${TRAIN_ITERS:-30000}

# MoE stability loss coefficients.
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}
MOE_Z_LOSS_COEFF=${MOE_Z_LOSS_COEFF:-1e-3}

# Fault injection into FlashAttention backward.
#   0 disables the target. 1..7 zero that many lower bf16 mantissa bits.
#   8 is an extra-severe setting: it also zeros the lowest exponent bit.
#   Common cases:
#     baseline: BITS_O=0, BITS_DO=0
#     O only:   BITS_O=7, BITS_DO=0
#     dO only:  BITS_O=0, BITS_DO=7
#     O+dO:     BITS_O=8, BITS_DO=8
BITS_O=${BITS_O:-0}
BITS_DO=${BITS_DO:-0}

# ============================================================================
# Argument parsing: command-line values override the config block above.
# ============================================================================
usage() {
    cat <<'EOF'
Usage: train_qwen3_moe_v4.sh [options]

Options:
  --lr RATE             Learning rate. Default: 5.0e-4
  --train-iters N       Training iterations. Default: 30000
  --aux-loss VALUE      MoE aux-loss coefficient. Default: 0.001
  --z-loss VALUE        MoE z-loss coefficient. Default: 1e-3
  --bits-o N            Fault O in FA backward by zeroing low bits. Range: 0..8.
  --bits-do N           Fault dO in FA backward by zeroing low bits. Range: 0..8.
  --no-fault            Force both --bits-o and --bits-do to 0.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr)           LR="$2"; shift 2 ;;
        --train-iters)  TRAIN_ITERS="$2"; shift 2 ;;
        --aux-loss|--aux-loss-coeff)
                        MOE_AUX_LOSS_COEFF="$2"; shift 2 ;;
        --z-loss|--z-loss-coeff)
                        MOE_Z_LOSS_COEFF="$2"; shift 2 ;;
        --bits-o)       BITS_O="$2"; shift 2 ;;
        --bits-do)      BITS_DO="$2"; shift 2 ;;
        --no-fault)     BITS_O=0; BITS_DO=0; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

validate_bits() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} must be an integer in [0, 8], got: ${value}"
        exit 1
    fi
    local value_10=$((10#$value))
    if (( value_10 < 0 || value_10 > 8 )); then
        echo "[ERROR] ${name} must be in [0, 8], got: ${value}"
        exit 1
    fi
}

validate_bits "BITS_O" "$BITS_O"
validate_bits "BITS_DO" "$BITS_DO"

# Derived values must come after argparse so overrides propagate.
MIN_LR=$(awk -v l="$LR" 'BEGIN {printf "%.10f", l/10}')

# ============================================================================
# Fault injection configuration
# ============================================================================
export bitshift_fa_backward_O=${BITS_O}
export bitshift_fa_backward_dO=${BITS_DO}
export round2nearest_fa_backward_O=0
export round2nearest_fa_backward_dO=0

FAULT_ENABLED=0
FAULT_TAG=""
if (( 10#$BITS_O > 0 )); then
    FAULT_ENABLED=1
    FAULT_TAG+="_bsO${BITS_O}"
fi
if (( 10#$BITS_DO > 0 )); then
    FAULT_ENABLED=1
    FAULT_TAG+="_bsdO${BITS_DO}"
fi

# ============================================================================
# Hardware: use all 16 NPUs.
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
MASTER_PORT=1234

# ============================================================================
# Data & tokenizer
# ============================================================================
DATA_PATH="/mnt/hdfs/training_data/fineweb-edu_100BT"
TOKENIZER_PATH="/mnt/hdfs/training_data/Qwen3-14B"

# ============================================================================
# Parallelism
#   TP=1, PP=1, CP=1, EP=8 over 16 NPUs gives:
#     shared-DP = 16
#     expert-DP = 2
# ============================================================================
TP=1
PP=1
EP=8
CP=1

# ============================================================================
# Model architecture (~9.6B total / ~1.2B active)
# ============================================================================
NUM_LAYERS=20
HIDDEN_SIZE=1536
NUM_ATTENTION_HEADS=12
NUM_QUERY_GROUPS=4
FFN_HIDDEN_SIZE=1536

NUM_EXPERTS=64
MOE_ROUTER_TOPK=4
MOE_FFN_HIDDEN_SIZE=1536

# ============================================================================
# Batch
#   GBS=384, MBS=6, shared-DP=16 -> num_micro_batches = 4
#   tokens/step = 384 x 4096 = 1,572,864
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
    --moe-aux-loss-coeff $MOE_AUX_LOSS_COEFF \
    --moe-z-loss-coeff $MOE_Z_LOSS_COEFF \
    --norm-topk-prob \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-layer-freq -1 \
    --first-k-dense-replace 0
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
# NOTE: --qk-layernorm is required for qwen3_spec to wire PTNorm on q/k.

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
    --lr-decay-iters $TRAIN_ITERS \
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
# All single-node v4 runs share the same artifact namespace. The tag carries
# the experiment type: baseline, bsO<N>, bsdO<N>, or both.
# ============================================================================
LOG_DIR="logs/v4"
METRICS_DIR="metrics/v4"

safe_tag_value() {
    printf "%s" "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LR_TAG=$(safe_tag_value "$LR")
AUX_TAG=$(safe_tag_value "$MOE_AUX_LOSS_COEFF")
if (( FAULT_ENABLED )); then
    TAG="v4${FAULT_TAG}_lr${LR_TAG}_aux${AUX_TAG}_iters${TRAIN_ITERS}_${TIMESTAMP}"
else
    TAG="v4_baseline_lr${LR_TAG}_aux${AUX_TAG}_iters${TRAIN_ITERS}_${TIMESTAMP}"
fi
LOG_FILE="${LOG_DIR}/train_${TAG}.log"
METRICS_FILE="${METRICS_DIR}/monitor_${TAG}.jsonl"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"
export STABILITY_MONITOR_OUTPUT_PATH="${METRICS_FILE}"

# Derived display values
HEAD_DIM=$((HIDDEN_SIZE / NUM_ATTENTION_HEADS))
TOKENS_PER_STEP=$((GBS * SEQ_LENGTH))
SHARED_DP=$((NPUS_PER_NODE / TP / PP / CP))
EXPERT_DP=$((NPUS_PER_NODE / EP / TP / PP / CP))
NUM_MICRO_BATCHES=$((GBS / MBS / SHARED_DP))

# ============================================================================
# Banner
# ============================================================================
echo "============================================"
if (( FAULT_ENABLED )); then
    echo "[INFO] v4 training with fault injection (Qwen3-MoE)"
else
    echo "[INFO] v4 baseline training (Qwen3-MoE)"
fi
echo "  model:        ${NUM_LAYERS}L x hidden=${HIDDEN_SIZE}, head_dim=${HEAD_DIM} (heads=${NUM_ATTENTION_HEADS}, kv_groups=${NUM_QUERY_GROUPS})"
echo "  MoE:          ${NUM_EXPERTS} experts, top-${MOE_ROUTER_TOPK}, expert_ffn=${MOE_FFN_HIDDEN_SIZE}"
echo "  parallelism:  TP=${TP} PP=${PP} EP=${EP} CP=${CP}  (shared-DP=${SHARED_DP}, expert-DP=${EXPERT_DP})"
echo "  batch:        MBS=${MBS}, GBS=${GBS}, micro_batches=${NUM_MICRO_BATCHES}, tokens/step=${TOKENS_PER_STEP}"
echo "  schedule:     lr=${LR} -> min_lr=${MIN_LR}, warmup=3%, iters=${TRAIN_ITERS}"
echo "  stability:    qk-norm ON, z-loss=${MOE_Z_LOSS_COEFF}, aux-loss=${MOE_AUX_LOSS_COEFF}, norm-topk-prob ON"
echo "  fault:        bitshift_O=${BITS_O}, bitshift_dO=${BITS_DO}"
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
    /opt/tiger/npu-stability/MindSpeed-LLM/pretrain_gpt.py
    $GPT_ARGS
    $MODEL_PARALLEL_ARGS
    $MOE_ARGS
    $TRAIN_ARGS
    $DATA_ARGS
    $OUTPUT_ARGS
    $OPTIMIZE_ARGS
    --monitor-delta-windows 10 50 200 500
    --distributed-backend nccl
)

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
