#!/bin/bash
set -euo pipefail

# ============================================================================
# v4 multi-node training, Qwen3-MoE
# ----------------------------------------------------------------------------
# Hardware: 16 x Ascend 910B per node
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
detect_ipv4() {
    local dev="$1"
    local ip_addr=""
    if command -v ip >/dev/null 2>&1; then
        ip_addr=$(ip -4 -o addr show dev "$dev" scope global 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}') || true
    fi
    if [[ -z "$ip_addr" ]]; then
        ip_addr=$(hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\./ {print; exit}') || true
    fi
    if [[ -z "$ip_addr" ]]; then
        ip_addr=$(hostname -i 2>/dev/null | tr ' ' '\n' | awk '/^[0-9]+\./ {print; exit}') || true
    fi
    printf "%s" "$ip_addr"
}

is_ipv6_addr() {
    [[ "$1" == *:* ]]
}

write_multinode_debug_marker() {
    local sync_dir="$1"
    mkdir -p "$sync_dir" 2>/dev/null || return 1
    {
        echo "date=$(date)"
        echo "node_rank=${NODE_RANK}"
        echo "arnold_id=${ARNOLD_ID:-}"
        echo "host=$(hostname)"
        echo "hostname_i=$(hostname -i 2>/dev/null || true)"
        echo "hostname_I=$(hostname -I 2>/dev/null || true)"
        echo "local_ipv4=${LOCAL_IPV4}"
        echo "master_addr_before=${MASTER_ADDR}"
        echo "master_port=${MASTER_PORT}"
        echo "hccl_socket_ifname=${HCCL_SOCKET_IFNAME}"
        echo "hccl_if_ip=${HCCL_IF_IP}"
    } > "${sync_dir}/node_${NODE_RANK}.txt" 2>/dev/null || true
}

resolve_master_addr_from_node0() {
    local sync_dir="${MULTINODE_SYNC_DIR:-/mnt/hdfs/__INFRA_OUTPUT__/npu_debug/${ARNOLD_TRIAL_ID:-${ARNOLD_RUN_ID:-default}}}"
    local master_file="${sync_dir}/master_ipv4"
    local wait_seconds="${MULTINODE_MASTER_WAIT_SECONDS:-300}"

    export MULTINODE_SYNC_DIR="$sync_dir"
    write_multinode_debug_marker "$sync_dir" || return 1

    if [[ "$NODE_RANK" == "0" ]]; then
        [[ -n "$LOCAL_IPV4" ]] || return 1
        printf "%s\n" "$LOCAL_IPV4" > "$master_file" 2>/dev/null || return 1
    fi

    local waited=0
    while [[ ! -s "$master_file" && "$waited" -lt "$wait_seconds" ]]; do
        sleep 1
        waited=$((waited + 1))
    done

    if [[ -s "$master_file" ]]; then
        local resolved_addr
        resolved_addr=$(head -n 1 "$master_file" | tr -d '[:space:]')
        if [[ -n "$resolved_addr" ]]; then
            MASTER_ADDR="$resolved_addr"
            return 0
        fi
    fi
    return 1
}

LR=${LR:-5.0e-4}
TRAIN_ITERS=${TRAIN_ITERS:-30000}
NNODES=${NNODES:-${ARNOLD_EXECUTOR_NUM:-${ARNOLD_NUM:-${ARNOLD_WORKER_NUM:-1}}}}
NODE_RANK=${NODE_RANK:-${ARNOLD_ID:-0}}
MASTER_ADDR=${MASTER_ADDR:-${METIS_WORKER_0_HOST:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}}
MASTER_PORT=${MASTER_PORT:-${METIS_WORKER_0_PORT:-${ARNOLD_WORKER_0_PORT:-1234}}}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-${ARNOLD_RDMA_INTERFACE:-eth0}}
LOCAL_IPV4=$(detect_ipv4 "$HCCL_SOCKET_IFNAME")
if [[ -z "${HCCL_IF_IP:-}" ]] || { [[ "${ALLOW_IPV6_HCCL_IF_IP:-0}" != "1" ]] && is_ipv6_addr "$HCCL_IF_IP"; }; then
    HCCL_IF_IP=$LOCAL_IPV4
fi

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
Usage: train_qwen3_moe_v4_multinode.sh [options]

Options:
  --lr RATE             Learning rate. Default: 5.0e-4
  --train-iters N       Training iterations. Default: 30000
  --aux-loss VALUE      MoE aux-loss coefficient. Default: 0.001
  --z-loss VALUE        MoE z-loss coefficient. Default: 1e-3
  --bits-o N            Fault O in FA backward by zeroing low bits. Range: 0..8.
  --bits-do N           Fault dO in FA backward by zeroing low bits. Range: 0..8.
  --no-fault            Force both --bits-o and --bits-do to 0.
  --nnodes N            Total nodes for distributed launch. Default: ARNOLD_EXECUTOR_NUM / ARNOLD_NUM.
  --node-rank N         Rank of this node. Default from NODE_RANK / ARNOLD_ID.
  --master-addr HOST    Master node host/IP. Default: METIS_WORKER_0_HOST, then ARNOLD_WORKER_0_HOST.
  --master-port PORT    Master port for rendezvous. Default: METIS_WORKER_0_PORT, then 1234.
  --npu-per-node N      NPUs per node. Default: 16.
  --ascend-visible-devices LIST  ASCEND/NPU visible devices list. Default: 0..15.
  --mbs N               Micro-batch size. Default: 3.
  --gbs N               Global batch size. Default: 384.
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
        --nnodes)       NNODES="$2"; shift 2 ;;
        --node-rank)    NODE_RANK="$2"; shift 2 ;;
        --master-addr)  MASTER_ADDR="$2"; shift 2 ;;
        --master-port)  MASTER_PORT="$2"; shift 2 ;;
        --npu-per-node) NPUS_PER_NODE="$2"; shift 2 ;;
        --ascend-visible-devices)
                        ASCEND_RT_VISIBLE_DEVICES="$2"
                        shift 2 ;;
        --mbs|--micro-batch-size)
                        MBS="$2"; shift 2 ;;
        --gbs|--global-batch-size)
                        GBS="$2"; shift 2 ;;
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

if (( NNODES > 1 )) && [[ "${ALLOW_IPV6_MASTER:-0}" != "1" ]] && { [[ "$MASTER_ADDR" == "127.0.0.1" ]] || is_ipv6_addr "$MASTER_ADDR"; }; then
    if ! resolve_master_addr_from_node0; then
        echo "[ERROR] failed to resolve IPv4 MASTER_ADDR from node0."
        echo "Set --master-addr <node0_ipv4> explicitly, or check MULTINODE_SYNC_DIR=${MULTINODE_SYNC_DIR:-unset}."
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
# Hardware visibility.
# ============================================================================
export ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES

# ============================================================================
# Standard Ascend / runtime environment
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME
export HCCL_IF_IP=$HCCL_IF_IP
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64500}
export HCCL_RDMA_TC=${HCCL_RDMA_TC:-236}
export HCCL_RDMA_SL=${HCCL_RDMA_SL:-5}
export HCCL_INTRA_PCIE_ENABLE=${HCCL_INTRA_PCIE_ENABLE:-0}
export HCCL_INTRA_ROCE_ENABLE=${HCCL_INTRA_ROCE_ENABLE:-1}
export P2P_HCCL_BUFFSIZE=${P2P_HCCL_BUFFSIZE:-20}
export INF_NAN_MODE_ENABLE=${INF_NAN_MODE_ENABLE:-1}
export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export STREAMS_PER_DEVICE=32

# Stability monitor toggle (consumed by stability_monitor/integration.py)
export STABILITY_MONITOR_ENABLED=1

# ============================================================================
# Distributed launch
# ============================================================================
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" ]]; then
    echo "[ERROR] multi-node run requires a real MASTER_ADDR. Set MASTER_ADDR, METIS_WORKER_0_HOST, or pass --master-addr."
    exit 1
fi
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == *:* ]] && [[ "${ALLOW_IPV6_MASTER:-0}" != "1" ]]; then
    echo "[ERROR] MASTER_ADDR looks like IPv6: ${MASTER_ADDR}"
    echo "HCCL multi-node runs are more reliable with IPv4. Pass --master-addr <node0_ipv4> or set ALLOW_IPV6_MASTER=1."
    exit 1
fi
if [[ -z "$HCCL_IF_IP" ]]; then
    echo "[ERROR] failed to detect local IPv4 for HCCL_IF_IP. Set HCCL_IF_IP explicitly."
    exit 1
fi
if (( NNODES > 1 )) && [[ "${ALLOW_IPV6_HCCL_IF_IP:-0}" != "1" ]] && is_ipv6_addr "$HCCL_IF_IP"; then
    echo "[ERROR] HCCL_IF_IP looks like IPv6: ${HCCL_IF_IP}"
    echo "Set HCCL_IF_IP to this node's IPv4, or set ALLOW_IPV6_HCCL_IF_IP=1."
    exit 1
fi

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
#   Default keeps the single-node global batch so loss dynamics stay comparable.
#   8x16 nodes: GBS=384, MBS=3, shared-DP=128 -> num_micro_batches = 1.
#   tokens/step = GBS x 4096
# ============================================================================
SEQ_LENGTH=4096
MBS=${MBS:-3}
GBS=${GBS:-384}

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
LOG_FILE="${LOG_DIR}/train_${TAG}_node${NODE_RANK}.log"
METRICS_FILE="${METRICS_DIR}/monitor_${TAG}_node${NODE_RANK}.jsonl"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"
export STABILITY_MONITOR_OUTPUT_PATH="${METRICS_FILE}"

# Derived display values
HEAD_DIM=$((HIDDEN_SIZE / NUM_ATTENTION_HEADS))
TOKENS_PER_STEP=$((GBS * SEQ_LENGTH))
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))
SHARED_DP=$((WORLD_SIZE / TP / PP / CP))
EXPERT_DP=$((WORLD_SIZE / EP / TP / PP / CP))
NUM_MICRO_BATCHES=$((GBS / MBS / SHARED_DP))
if (( NUM_MICRO_BATCHES < 1 )); then
    echo "[ERROR] invalid micro-batch config: GBS=${GBS}, MBS=${MBS}, shared_DP=${SHARED_DP}"
    echo "Need GBS = MBS * shared_DP * k (k >= 1)."
    exit 1
fi

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
echo "  distributed:  nnodes=${NNODES}, node_rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "  hccl:         HCCL_IF_IP=${HCCL_IF_IP}, HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}"
echo "  hccl-extra:   base_port=${HCCL_IF_BASE_PORT}, roce=${HCCL_INTRA_ROCE_ENABLE}, pcie=${HCCL_INTRA_PCIE_ENABLE}, rdma_tc=${HCCL_RDMA_TC}, rdma_sl=${HCCL_RDMA_SL}"
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
