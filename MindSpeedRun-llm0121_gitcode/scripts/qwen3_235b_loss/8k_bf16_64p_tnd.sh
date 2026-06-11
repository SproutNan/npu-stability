#!/bin/bash
set -euo pipefail

# ==============================================================================
# 1. Device detection and environment initialization
# ==============================================================================
detect_device_name() {
    local name
    name=$(python3 -c "import torch_npu;n=torch_npu.npu.get_device_name();print('A5'if('Ascend910_95'in n or'Ascend950'in n)else'A3'if'Ascend910_93'in n else'A2'if'Ascend910B'in n else'unknown')" 2>/dev/null || echo "unknown")
    echo "${name:-unknown}"
}

DEVICE_NAME=A5
echo "[INFO] Detected device type: $DEVICE_NAME"

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2
export STREAMS_PER_DEVICE=32

if [[ "$DEVICE_NAME" == "A5" ]]; then
    export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47,npu4:48-71,npu5:72-95,npu6:240-263,npu7:264-287
else
    export CPU_AFFINITY_CONF=1
fi

# ==============================================================================
# 2. Parameters
# ==============================================================================
if [ $# -ne 2 ]; then
    echo "Usage: $0 \"<IP list>\" <inet prefix>"
    echo "Example: $0 \"141.61.50.149 141.61.50.145 141.61.50.141 141.61.50.137\" 141"
    exit 1
fi

IP_LIST="$1"
INET_PREFIX="$2"
IPs=($IP_LIST)
LOCAL_HOST=$(ip addr show 2>/dev/null | grep "inet $INET_PREFIX" | awk '{print $2}' | cut -d/ -f1 | head -n 1)
if [[ -z "$LOCAL_HOST" ]]; then
    LOCAL_HOST=$(ifconfig 2>/dev/null | grep "inet $INET_PREFIX" | awk '{print $2}' | head -n 1)
fi
if [[ -z "$LOCAL_HOST" ]]; then
    echo "[ERROR] Failed to resolve local IP with prefix $INET_PREFIX"
    exit 1
fi
echo "Local host IP: $LOCAL_HOST"

WORLD_SIZE_KEY="64p"
WORLD_SIZE_EXPECTED=64
NPUS_PER_NODE=8
if [[ "$DEVICE_NAME" == "A3" ]]; then
    if [[ "$WORLD_SIZE_KEY" == "8p" ]]; then
        NPUS_PER_NODE=8
    else
        NPUS_PER_NODE=16
    fi
fi

MASTER_ADDR=${IPs[0]}
MASTER_PORT=6300
NNODES=${#IPs[@]}
NODE_RANK=""

for i in "${!IPs[@]}"; do
    if [[ "$LOCAL_HOST" == "${IPs[$i]}" ]]; then
        echo "Node Rank: ${i}"
        NODE_RANK=$i
        break
    fi
done

if [[ -z "$NODE_RANK" ]]; then
    echo "[ERROR] NODE_RANK is not set"
    exit 1
fi

WORLD_SIZE=$((NPUS_PER_NODE * NNODES))
if [[ "$WORLD_SIZE" -ne "$WORLD_SIZE_EXPECTED" ]]; then
    echo "[ERROR] Actual world size ${WORLD_SIZE} does not match fixed value ${WORLD_SIZE_EXPECTED}"
    exit 1
fi

DATA_PATH="/home/dataset/qwen3_235b/train-00000-of-00042-d964455e17e96d5a_eod_text_document"
TOKENIZER_PATH="/home/dataset/hf_weights/qwen3-235B"
if [[ "$DEVICE_NAME" == "A3" ]]; then
    PROF_DIR="/mnt/share/prof"
else
    PROF_DIR="/home/prof"
fi

LAYOUT="tnd"
PRECISION="bf16"
TP=1
PP=1
EP=64
CP=1
MBS=1
GBS=64
LAYER=36
EXPERTS=128
SEQ_LENGTH=8192
SUB_SEQ=8192
SEQ_LEN_KEY="$((SEQ_LENGTH / 1024))k"
if [[ "$SUB_SEQ" -eq -1 ]]; then
    SUB_SEQ_LABEL="0k"
else
    SUB_SEQ_LABEL="$((SUB_SEQ / 1024))k"
fi
TRAIN_ITERS=2000
CP_TYPE="kvallgather_cp_algo"
CP_SHORT="ag"
ROUTER_BALANCING_TYPE="aux_loss"
SWAP_OPT="true"
FIX_ROUTER_ENABLED="false"
PROFILE_ENABLED="false"
PROFILE_STEP_START=6
PROFILE_STEP_END=7
PROFILE_RANKS=-1

# ==============================================================================
# 3. Argument assembly and file naming
# ==============================================================================
DISTRIBUTED_ARGS=(
    --nproc_per_node "$NPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

MOE_ARGS=(
    --num-experts "$EXPERTS"
    --moe-router-topk 8
    --moe-ffn-hidden-size 1536
    --moe-router-load-balancing-type "${ROUTER_BALANCING_TYPE}"
    --norm-topk-prob
    --moe-grouped-gemm
    --moe-permutation-async-comm
    --moe-token-dispatcher-type alltoall
    --moe-layer-freq -1
    --first-k-dense-replace 0
    --moe-aux-loss-coeff 0.001
)

OPTIMIZE_ARGS=(
    --use-flash-attn
    --use-fused-rotary-pos-emb
    --sequence-parallel
    --use-rotary-position-embeddings
    --use-fused-swiglu
    --use-fused-rmsnorm
    --no-masked-softmax-fusion
    --use-distributed-optimizer
    --use-cp-send-recv-overlap
    --gemm-gradient-accumulation-fusion
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --manual-gc
    --manual-gc-interval 50
)

TRAIN_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --lr 1.25e-6
    --lr-decay-style cosine
    --min-lr 1.25e-7
    --weight-decay 1e-1
    --lr-warmup-fraction 0.01
    --attention-dropout 0.0
    --init-method-std 0.01
    --hidden-dropout 0.0
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --initial-loss-scale 4096
    --seed 42
    --bf16
    --train-iters "$TRAIN_ITERS"
    --seq-length "$SEQ_LENGTH"
    --no-shared-storage
)

PRECISION_ARGS=()

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP"
    --context-parallel-size "$CP"
    --context-parallel-algo "$CP_TYPE"
    --expert-tensor-parallel-size 1
    --attention-mask-type causal
)

GPT_ARGS=(
    --use-mcore-models
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec
    --kv-channels 128
    --qk-layernorm
    --tokenizer-name-or-path "$TOKENIZER_PATH"
    --max-position-embeddings "$SEQ_LENGTH"
    --num-layers "$LAYER"
    --hidden-size 4096
    --ffn-hidden-size 12288
    --num-attention-heads 64
    --tokenizer-type PretrainedFromHF
    --make-vocab-size-divisible-by 1
    --padded-vocab-size 151936
    --rotary-base 1000000
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --attention-softmax-in-fp32
    --group-query-attention
    --num-query-groups 4
    --use-fused-ring-attention-update
    --swap-optimizer
)

LAYOUT_ARGS=(
    --fix-sub-seq-length "$SUB_SEQ"
    --reset-attention-mask
    --reset-position-ids
    --variable-seq-lengths
)

OVERLAP_ARGS=(
    --moe-fb-overlap
)

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --split 100,0,0
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval "$TRAIN_ITERS"
    --eval-interval "$TRAIN_ITERS"
    --eval-iters 0
    --no-load-optim
    --no-load-rng
)

LOAD_ARGS=(
    --load "/home/weights/qwen3_235b/${WORLD_SIZE}p_l${LAYER}_e${EXPERTS}"
)

EXIT_INTERVAL_ARGS=()

MODEL_NAME=qwen3
TIME_STAMP=$(date +%m%d_%H%M)
NAME="${DEVICE_NAME}_${MODEL_NAME}_${SEQ_LEN_KEY}_${SUB_SEQ_LABEL}_${LAYOUT}_${WORLD_SIZE}p_l${LAYER}_n${NODE_RANK}_${PRECISION}_cp${CP}_${CP_SHORT}_${TIME_STAMP}"
TARGET_PATH="$PROF_DIR/${MODEL_NAME}/${NAME}"
mkdir -p "$TARGET_PATH"

LOG_FILE="$TARGET_PATH/${NAME}.log"
PARENT_LOG_FILE="$(dirname "$TARGET_PATH")/${NAME}.log"

cp "$0" "$TARGET_PATH/${NAME}.sh"
cp "$0" "$(dirname "$TARGET_PATH")/${NAME}.sh"

PROFILE_ARGS=()

CMD=(
    torchrun
    "${DISTRIBUTED_ARGS[@]}"
    pretrain_gpt.py
    "${GPT_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${OUTPUT_ARGS[@]}"
    "${OPTIMIZE_ARGS[@]}"
    "${OVERLAP_ARGS[@]}"
    "${TRAIN_ARGS[@]}"
    "${PRECISION_ARGS[@]}"
    "${MODEL_PARALLEL_ARGS[@]}"
    "${LAYOUT_ARGS[@]}"
    "${PROFILE_ARGS[@]}"
    --transformer-impl transformer_engine
    "${EXIT_INTERVAL_ARGS[@]}"
    "${LOAD_ARGS[@]}"
    --distributed-backend nccl
)

# ==============================================================================
# 4. Write environment information to the log file
# ==============================================================================
{
    echo "================================================================================"
    echo "Environment collection time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"

    echo "[IP Info]"
    echo "----------------------------------------"
    echo "Detected local IP (INET_PREFIX=$INET_PREFIX): $LOCAL_HOST"
    echo "Node rank: $NODE_RANK / $NNODES"
    echo "All node IPs:"
    for i in "${!IPs[@]}"; do
        echo "  node$i: ${IPs[$i]}"
    done
    echo "Master node: $MASTER_ADDR:$MASTER_PORT"
    echo ""
    echo "[mindspeed Package]"
    echo "----------------------------------------"

    MINSPEED_PKG_INFO=$(pip list 2>/dev/null | grep -E '^mindspeed\s+' || true)
    echo "pip list | grep mindspeed:"
    if [[ -n "$MINSPEED_PKG_INFO" ]]; then
        echo "$MINSPEED_PKG_INFO"
    else
        echo "none"
    fi
    echo ""

    MINSPEED_PKG_NAME=$(printf '%s\n' "$MINSPEED_PKG_INFO" | awk '{print $1}' | head -1 || true)

    if [[ "$MINSPEED_PKG_NAME" == "mindspeed" ]]; then
        echo "Found mindspeed package: $MINSPEED_PKG_NAME"
        MINSPEED_PATH=$(python3 -c "
try:
    import mindspeed
    print(mindspeed.__path__[0])
except (ImportError, AttributeError, IndexError):
    print('')
" 2>/dev/null)

        if [[ -n "$MINSPEED_PATH" && -d "$MINSPEED_PATH" ]]; then
            echo "mindspeed install path: $MINSPEED_PATH"
            if [[ -d "$MINSPEED_PATH/.git" ]]; then
                GIT_PATH="$MINSPEED_PATH"
                echo "Found git repository in install path"
            else
                PARENT_DIR=$(dirname "$MINSPEED_PATH")
                if [[ -d "$PARENT_DIR/.git" ]]; then
                    GIT_PATH="$PARENT_DIR"
                    echo "Found git repository in parent path: $PARENT_DIR"
                else
                    CURRENT_DIR="$MINSPEED_PATH"
                    FOUND_GIT=""
                    for i in {1..5}; do
                        CURRENT_DIR=$(dirname "$CURRENT_DIR")
                        if [[ "$CURRENT_DIR" == "/" ]]; then
                            break
                        fi
                        if [[ -d "$CURRENT_DIR/.git" ]]; then
                            FOUND_GIT="$CURRENT_DIR"
                            break
                        fi
                    done
                    if [[ -n "$FOUND_GIT" ]]; then
                        GIT_PATH="$FOUND_GIT"
                        echo "Found git repository path: $GIT_PATH"
                    else
                        GIT_PATH=""
                        echo "Git repository not found near mindspeed install path"
                    fi
                fi
            fi

            if [[ -n "$GIT_PATH" && -d "$GIT_PATH/.git" ]]; then
                echo ""
                echo "--- git log -1 ---"
                (cd "$GIT_PATH" && git log -1 2>&1) || echo "Failed to get git log"
                echo ""
                echo "--- git status ---"
                (cd "$GIT_PATH" && git status 2>&1) || echo "Failed to get git status"
                echo ""
                echo "--- git diff ---"
                (cd "$GIT_PATH" && git diff 2>&1) || echo "Failed to get git diff"
                echo ""
                echo "--- git diff --cached ---"
                (cd "$GIT_PATH" && git diff --cached 2>&1) || echo "Failed to get git diff --cached"
                echo ""
                echo "--- git branch ---"
                (cd "$GIT_PATH" && git branch 2>&1) || echo "Failed to get git branch"
                echo ""
                echo "--- git remote -v ---"
                (cd "$GIT_PATH" && git remote -v 2>&1) || echo "Failed to get git remote"
                echo ""
                echo "--- git rev-parse HEAD ---"
                (cd "$GIT_PATH" && git rev-parse HEAD 2>&1) || echo "Failed to get commit hash"
                echo ""
                echo "--- git describe --tags ---"
                (cd "$GIT_PATH" && git describe --tags 2>&1) || echo "Failed to get git describe"
            else
                echo "mindspeed git repository metadata not found"
                echo "mindspeed install path: $MINSPEED_PATH"
                echo "Directory listing:"
                ls -la "$MINSPEED_PATH" 2>/dev/null | head -10
            fi
        else
            echo "Failed to resolve mindspeed install path"
            echo "Trying direct module inspection:"
            python3 -c "
try:
    import mindspeed
    print('mindspeed module imported')
    print('mindspeed file:', mindspeed.__file__ if hasattr(mindspeed, '__file__') else 'unknown')
    if hasattr(mindspeed, '__path__'):
        print('mindspeed path:', mindspeed.__path__)
except ImportError:
    print('failed to import mindspeed module')
" 2>&1
        fi
    else
        echo "No installed mindspeed package detected; continuing"
        echo "Current pip entries matching mindspeed:"
        pip list 2>/dev/null | grep -i mindspeed || echo "none"
        echo ""
        echo "Searching mindspeed under Python site-packages:"
        python3 -c "
import site
import glob
for path in site.getsitepackages():
    mindspeed_dirs = glob.glob(path + '/mindspeed*')
    if mindspeed_dirs:
        print('found:', mindspeed_dirs)
" 2>/dev/null || echo "Failed to query site-packages"
    fi

    echo ""
    echo "[Device Info]"
    echo "----------------------------------------"
    echo "Detected device type: $DEVICE_NAME"

    echo ""
    echo "[Environment Variables]"
    echo "----------------------------------------"
    env

    echo "[Installed Python Packages]"
    echo "----------------------------------------"
    pip list 2>/dev/null | grep -E 'torch|mindspeed' || pip3 list 2>/dev/null

} | tee -a "$LOG_FILE"

# ==============================================================================
# 5. Launch
# ==============================================================================
echo "[INFO] Launch summary:"
echo "  SeqLen: $SEQ_LENGTH, SUB_SEQ: $SUB_SEQ ($SUB_SEQ_LABEL), LAYOUT: $LAYOUT, PRECISION: $PRECISION"
echo "  Parallel: ${WORLD_SIZE}p, CP: $CP, CP_TYPE: $CP_TYPE"
echo "  Layer: $LAYER, EP: $EP, EXPERTS: $EXPERTS"
echo "  SwapOpt: $SWAP_OPT, FixRouter: $FIX_ROUTER_ENABLED, Profile: $PROFILE_ENABLED, TrainIters: $TRAIN_ITERS"
echo "  Log path: $LOG_FILE"
echo "  OVERLAP_ARGS=${OVERLAP_ARGS[*]:-<empty>}"
echo "  Command: $(printf '%q ' "${CMD[@]}")"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
cp "$LOG_FILE" "$PARENT_LOG_FILE"
