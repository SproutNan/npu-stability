#!/bin/bash
set -euo pipefail

cd /opt/tiger

if [ -d npu-stability/.git ]; then
    cd npu-stability
    git pull
else
    git clone https://github.com/SproutNan/npu-stability.git
    cd npu-stability
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install -r MindSpeed/requirements.txt
pip install -e MindSpeed

# ============================================================================
# v4 baseline training, Qwen3-MoE, 8 nodes x 16 NPUs
# ============================================================================

LR=5.0e-4
TRAIN_ITERS=1250

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

MIN_LR=$(awk -v l="$LR" 'BEGIN {printf "%.10f", l/10}')

# ============================================================================
# Hardware
# ============================================================================
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
NPUS_PER_NODE=16

# ============================================================================
# Ascend / runtime environment
# ============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export STREAMS_PER_DEVICE=32
export STABILITY_MONITOR_ENABLED=1

# ============================================================================
# Distributed launch
#   Arnold should provide:
#     ARNOLD_WORKER_NUM=8
#     ARNOLD_ID=0..7
#     ARNOLD_WORKER_0_HOST=<rank0 host/ip>
#     ARNOLD_WORKER_0_PORT=<free rendezvous port>
# ============================================================================
NNODES=${ARNOLD_WORKER_NUM:-8}
NODE_RANK=${ARNOLD_ID:-0}
MASTER_ADDR=${ARNOLD_WORKER_0_HOST:?ARNOLD_WORKER_0_HOST is required}
MASTER_PORT=${ARNOLD_WORKER_0_PORT:-1234}
WORLD_SIZE=$((NNODES * NPUS_PER_NODE))

# ============================================================================
# Data & tokenizer
# ============================================================================
DATA_PATH="/mnt/hdfs/training_data/fineweb-edu_100BT"
TOKENIZER_PATH="/mnt/hdfs/training_data/Qwen3-14B"

# ============================================================================
# Parallelism
#   8 nodes x 16 NPUs = 128 ranks.
#   Keep the single-node model-parallel shape and scale only data parallelism:
#     TP=1, PP=1, CP=1, EP=8
#     shared-DP = 128
#     expert-DP = 16
# ============================================================================
TP=1
PP=1
EP=8
CP=1

# ============================================================================
# Model architecture
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
#   Single node used: MBS=6, GBS=384, DP=16 -> 4 micro-batches.
#   8 nodes use:      MBS=6, GBS=3072, DP=128 -> 4 micro-batches.
#   TRAIN_ITERS defaults to 1250 to keep total tokens close to 10k single-node
#   steps. Use --train-iters 10000 if you intentionally want 8x more tokens.
# ============================================================================
SEQ_LENGTH=4096
MBS=6
GBS=3072

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

LOG_DIR="logs/v4_8node"
METRICS_DIR="metrics/v4_8node"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="v4_8node_lr${LR}_iters${TRAIN_ITERS}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/train_${TAG}_rank${NODE_RANK}.log"
METRICS_FILE="${METRICS_DIR}/monitor_${TAG}_rank${NODE_RANK}.jsonl"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"
export STABILITY_MONITOR_OUTPUT_PATH="${METRICS_FILE}"

HEAD_DIM=$((HIDDEN_SIZE / NUM_ATTENTION_HEADS))
TOKENS_PER_STEP=$((GBS * SEQ_LENGTH))
SHARED_DP=$((WORLD_SIZE / TP / PP / CP))
EXPERT_DP=$((WORLD_SIZE / EP / TP / PP / CP))
NUM_MICRO_BATCHES=$((GBS / MBS / SHARED_DP))

echo "============================================"
echo "[INFO] v4 baseline training, 8-node config"
echo "  nnodes:       ${NNODES}, node_rank=${NODE_RANK}, npu/node=${NPUS_PER_NODE}, world=${WORLD_SIZE}"
echo "  master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "  model:        ${NUM_LAYERS}L x hidden=${HIDDEN_SIZE}, head_dim=${HEAD_DIM} (heads=${NUM_ATTENTION_HEADS}, kv_groups=${NUM_QUERY_GROUPS})"
echo "  MoE:          ${NUM_EXPERTS} experts, top-${MOE_ROUTER_TOPK}, expert_ffn=${MOE_FFN_HIDDEN_SIZE}"
echo "  parallelism:  TP=${TP} PP=${PP} EP=${EP} CP=${CP}  (shared-DP=${SHARED_DP}, expert-DP=${EXPERT_DP})"
echo "  batch:        MBS=${MBS}, GBS=${GBS}, micro_batches=${NUM_MICRO_BATCHES}, tokens/step=${TOKENS_PER_STEP}"
echo "  schedule:     lr=${LR} -> min_lr=${MIN_LR}, warmup=3%, iters=${TRAIN_ITERS}"
echo "  data:         ${DATA_PATH}"
echo "  tokenizer:    ${TOKENIZER_PATH}"
echo "  log:          ${LOG_FILE}"
echo "  metrics:      ${METRICS_FILE}"
echo "============================================"

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
