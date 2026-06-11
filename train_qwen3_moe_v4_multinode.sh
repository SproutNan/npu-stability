#!/bin/bash
set -euo pipefail

# ============================================================================
# v4 multi-node training, Qwen3-MoE
# ----------------------------------------------------------------------------
# Default target: 8 nodes x 16 Ascend 910B NPUs.
# Model/data/monitor settings match train_qwen3_moe_v4.sh.
#
# This script intentionally uses the standard PyTorch launch module only:
#   python -m torch.distributed.launch
#
# Usage:
#   bash train_qwen3_moe_v4_multinode.sh --train-iters 50000 --bits-o 8 --bits-do 8
#   bash train_qwen3_moe_v4_multinode.sh --nnodes 8 --node-rank 0 --master-addr HOST --master-port 1234
# ============================================================================

# ============================================================================
# User config: edit this block for common experiments.
# ============================================================================
LR=${LR:-5.0e-4}
TRAIN_ITERS=${TRAIN_ITERS:-30000}

# Keep GBS the same as the single-node run so each optimizer step sees the same
# number of tokens. With 8 x 16 NPUs and DP=128, MBS=3 gives 1 micro-batch.
MBS=${MBS:-3}
GBS=${GBS:-384}

# MoE stability loss coefficients.
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}
MOE_Z_LOSS_COEFF=${MOE_Z_LOSS_COEFF:-1e-3}

# Fault injection into FlashAttention backward.
#   0 disables the target. 1..7 zero that many lower bf16 mantissa bits.
#   8 is an extra-severe setting: it also zeros the lowest exponent bit.
BITS_O=${BITS_O:-0}
BITS_DO=${BITS_DO:-0}

# Data paths.
DATA_PATH=${DATA_PATH:-/mnt/hdfs/training_data/fineweb-edu_100BT}
TOKENIZER_PATH=${TOKENIZER_PATH:-/mnt/hdfs/training_data/Qwen3-14B}

# Distributed defaults. Command-line args override these values; if omitted,
# Arnold executor variables are used.
NNODES=${NNODES:-}
NODE_RANK=${NODE_RANK:-}
NPUS_PER_NODE=${NPUS_PER_NODE:-${NPROC_PER_NODE:-}}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-}

# ============================================================================
# Argument parsing: command-line values override the config block above.
# ============================================================================
usage() {
    cat <<'EOF'
Usage: train_qwen3_moe_v4_multinode.sh [options]

Experiment options:
  --lr RATE             Learning rate. Default: 5.0e-4
  --train-iters N       Training iterations. Default: 30000
  --aux-loss VALUE      MoE aux-loss coefficient. Default: 0.001
  --z-loss VALUE        MoE z-loss coefficient. Default: 1e-3
  --bits-o N            Fault O in FA backward by zeroing low bits. Range: 0..8.
  --bits-do N           Fault dO in FA backward by zeroing low bits. Range: 0..8.
  --no-fault            Force both --bits-o and --bits-do to 0.
  --mbs N               Micro-batch size. Default: 3 for 8 x 16.
  --gbs N               Global batch size. Default: 384.
  --data-path PATH      Megatron indexed dataset prefix.
  --tokenizer-path PATH HuggingFace tokenizer path.

Distributed options:
  --nnodes N            Number of nodes. Default: ARNOLD_EXECUTOR_NUM, else 8.
  --node-rank N         Current node rank. Default: ARNOLD_ID, else 0.
  --nproc-per-node N    NPUs per node. Default: ARNOLD_EXECUTOR_GPU, else 16.
  --npus-per-node N     Alias for --nproc-per-node.
  --master-addr HOST    Rank-0 host. Default: ARNOLD_EXECUTOR_0_HOST.
  --master-port PORT    Rendezvous port. Default: first ARNOLD_EXECUTOR_0_PORT, else 1234.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr)               LR="$2"; shift 2 ;;
        --train-iters)      TRAIN_ITERS="$2"; shift 2 ;;
        --aux-loss|--aux-loss-coeff)
                            MOE_AUX_LOSS_COEFF="$2"; shift 2 ;;
        --z-loss|--z-loss-coeff)
                            MOE_Z_LOSS_COEFF="$2"; shift 2 ;;
        --bits-o)           BITS_O="$2"; shift 2 ;;
        --bits-do)          BITS_DO="$2"; shift 2 ;;
        --no-fault)         BITS_O=0; BITS_DO=0; shift ;;
        --mbs|--micro-batch-size)
                            MBS="$2"; shift 2 ;;
        --gbs|--global-batch-size)
                            GBS="$2"; shift 2 ;;
        --data-path)        DATA_PATH="$2"; shift 2 ;;
        --tokenizer-path)   TOKENIZER_PATH="$2"; shift 2 ;;
        --nnodes)           NNODES="$2"; shift 2 ;;
        --node-rank|--node_rank)
                            NODE_RANK="$2"; shift 2 ;;
        --nproc-per-node|--nproc_per_node|--npus-per-node)
                            NPUS_PER_NODE="$2"; shift 2 ;;
        --master-addr|--master_addr)
                            MASTER_ADDR="$2"; shift 2 ;;
        --master-port|--master_port)
                            MASTER_PORT="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

validate_positive_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} must be a positive integer, got: ${value}"
        exit 1
    fi
    local value_10=$((10#$value))
    if (( value_10 < 1 )); then
        echo "[ERROR] ${name} must be a positive integer, got: ${value}"
        exit 1
    fi
}

validate_nonnegative_int() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] ${name} must be a non-negative integer, got: ${value}"
        exit 1
    fi
}

validate_bits() {
    local name="$1"
    local value="$2"
    validate_nonnegative_int "$name" "$value"
    local value_10=$((10#$value))
    if (( value_10 > 8 )); then
        echo "[ERROR] ${name} must be in [0, 8], got: ${value}"
        exit 1
    fi
}

# Resolve Arnold distributed defaults after parsing so empty command-line/env
# values still fall back cleanly.
NNODES=${NNODES:-${ARNOLD_EXECUTOR_NUM:-8}}
NODE_RANK=${NODE_RANK:-${ARNOLD_ID:-0}}
NPUS_PER_NODE=${NPUS_PER_NODE:-${ARNOLD_EXECUTOR_GPU:-16}}
MASTER_ADDR=${MASTER_ADDR:-${ARNOLD_EXECUTOR_0_HOST:-${METIS_WORKER_0_HOST:-${ARNOLD_WORKER_0_HOST:-}}}}
MASTER_PORT_RAW=${MASTER_PORT:-${ARNOLD_EXECUTOR_0_PORT:-${METIS_WORKER_0_PORT:-${ARNOLD_WORKER_0_PORT:-1234}}}}
MASTER_PORT=${MASTER_PORT_RAW%%,*}

validate_bits "BITS_O" "$BITS_O"
validate_bits "BITS_DO" "$BITS_DO"
validate_positive_int "TRAIN_ITERS" "$TRAIN_ITERS"
validate_positive_int "MBS" "$MBS"
validate_positive_int "GBS" "$GBS"
validate_positive_int "NNODES" "$NNODES"
validate_positive_int "NPUS_PER_NODE" "$NPUS_PER_NODE"
validate_positive_int "MASTER_PORT" "$MASTER_PORT"
validate_nonnegative_int "NODE_RANK" "$NODE_RANK"

if (( 10#$NODE_RANK >= 10#$NNODES )); then
    echo "[ERROR] NODE_RANK (${NODE_RANK}) must be smaller than NNODES (${NNODES})."
    exit 1
fi

if [[ -z "$MASTER_ADDR" ]]; then
    if (( 10#$NNODES == 1 )); then
        MASTER_ADDR=127.0.0.1
    else
        echo "[ERROR] MASTER_ADDR is empty for multi-node launch."
        echo "Set --master-addr, MASTER_ADDR, or ARNOLD_EXECUTOR_0_HOST."
        exit 1
    fi
fi

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
# Hardware: use all visible NPUs on each node.
# ============================================================================
make_device_list() {
    local count="$1"
    local devices="0"
    local i
    for (( i = 1; i < count; i++ )); do
        devices+=",${i}"
    done
    printf "%s" "$devices"
}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(make_device_list "$NPUS_PER_NODE")}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$ASCEND_RT_VISIBLE_DEVICES}

# ============================================================================
# Standard Ascend / runtime environment
# ============================================================================
first_ipv4() {
    local ips=""
    local candidate
    ips=$(hostname -I 2>/dev/null || true)
    if [[ -z "$ips" ]]; then
        ips=$(hostname -i 2>/dev/null || true)
    fi
    for candidate in $ips; do
        if [[ "$candidate" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            printf "%s" "$candidate"
            return 0
        fi
    done
    return 1
}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}

export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64500}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-${ARNOLD_RDMA_INTERFACE:-eth0}}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export P2P_HCCL_BUFFSIZE=${P2P_HCCL_BUFFSIZE:-20}

if [[ -z "${HCCL_IF_IP:-}" ]]; then
    if HCCL_IF_IP_DETECTED=$(first_ipv4); then
        export HCCL_IF_IP="$HCCL_IF_IP_DETECTED"
    else
        echo "[WARN] Could not auto-detect an IPv4 address for HCCL_IF_IP."
    fi
else
    export HCCL_IF_IP
fi

# Stability monitor toggle (consumed by stability_monitor/integration.py)
export STABILITY_MONITOR_ENABLED=${STABILITY_MONITOR_ENABLED:-1}

# ============================================================================
# Parallelism
#   Single-node uses EP=8 over 16 NPUs.
#   Multi-node keeps EP=8 and increases data parallelism.
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
#   Default multi-node: GBS=384, MBS=3, world=8*16 -> num_micro_batches=1
#   tokens/step = 384 x 4096 = 1,572,864, same as single-node.
# ============================================================================
SEQ_LENGTH=4096

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
# Each node writes its own stdout/stderr log. Stability metrics are still
# emitted by global rank 0 only.
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
    TAG="v4_multinode${FAULT_TAG}_lr${LR_TAG}_aux${AUX_TAG}_nodes${NNODES}_npu${NPUS_PER_NODE}_mbs${MBS}_gbs${GBS}_iters${TRAIN_ITERS}_${TIMESTAMP}"
else
    TAG="v4_multinode_baseline_lr${LR_TAG}_aux${AUX_TAG}_nodes${NNODES}_npu${NPUS_PER_NODE}_mbs${MBS}_gbs${GBS}_iters${TRAIN_ITERS}_${TIMESTAMP}"
fi
LOG_FILE="${LOG_DIR}/train_${TAG}_node${NODE_RANK}.log"
METRICS_FILE="${METRICS_DIR}/monitor_${TAG}.jsonl"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"
export STABILITY_MONITOR_OUTPUT_PATH="${METRICS_FILE}"

# Derived display values and config checks.
WORLD_SIZE=$((10#$NNODES * 10#$NPUS_PER_NODE))
HEAD_DIM=$((HIDDEN_SIZE / NUM_ATTENTION_HEADS))
TOKENS_PER_STEP=$((GBS * SEQ_LENGTH))
SHARED_DP=$((WORLD_SIZE / TP / PP / CP))
EXPERT_DP=$((WORLD_SIZE / EP / TP / PP / CP))

if (( WORLD_SIZE % (TP * PP * CP) != 0 )); then
    echo "[ERROR] WORLD_SIZE=${WORLD_SIZE} must be divisible by TP*PP*CP=$((TP * PP * CP))."
    exit 1
fi
if (( WORLD_SIZE % (EP * TP * PP * CP) != 0 )); then
    echo "[ERROR] WORLD_SIZE=${WORLD_SIZE} must be divisible by EP*TP*PP*CP=$((EP * TP * PP * CP))."
    exit 1
fi
if (( GBS % (MBS * SHARED_DP) != 0 )); then
    echo "[ERROR] invalid micro-batch config: GBS=${GBS}, MBS=${MBS}, shared_DP=${SHARED_DP}"
    echo "Need GBS = MBS * shared_DP * k (k >= 1)."
    echo "For 8 x 16 with GBS=384, use MBS=3 or MBS=1."
    exit 1
fi
NUM_MICRO_BATCHES=$((GBS / MBS / SHARED_DP))

# ============================================================================
# Banner
# ============================================================================
echo "============================================"
if (( FAULT_ENABLED )); then
    echo "[INFO] v4 multi-node training with fault injection (Qwen3-MoE)"
else
    echo "[INFO] v4 multi-node baseline training (Qwen3-MoE)"
fi
echo "  model:        ${NUM_LAYERS}L x hidden=${HIDDEN_SIZE}, head_dim=${HEAD_DIM} (heads=${NUM_ATTENTION_HEADS}, kv_groups=${NUM_QUERY_GROUPS})"
echo "  MoE:          ${NUM_EXPERTS} experts, top-${MOE_ROUTER_TOPK}, expert_ffn=${MOE_FFN_HIDDEN_SIZE}"
echo "  distributed:  nnodes=${NNODES}, node_rank=${NODE_RANK}, nproc_per_node=${NPUS_PER_NODE}, world=${WORLD_SIZE}"
echo "  rendezvous:   ${MASTER_ADDR}:${MASTER_PORT}"
echo "  hccl:         HCCL_IF_IP=${HCCL_IF_IP:-}, HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}"
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
