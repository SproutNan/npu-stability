#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export NPU_ASD_ENABLE=0

export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47,npu4:48-71,npu5:72-95,npu6:240-263,npu7:264-287
export TASK_QUEUE_ENABLE=2
export STREAMS_PER_DEVICE=32

# 检查传入的参数数量是否正确
if [ $# -ne 2 ]; then
    echo "用法: $0 <IP列表> <inet前缀数字>"
    echo "示例: $0 '141.61.50.149 141.61.50.145 141.61.50.141 141.61.50.137' 141"
    exit 1
fi

# 接收启动参数
IP_LIST=$1          # 第一个参数：IP列表（空格分隔）
INET_PREFIX=$2      # 第二个参数：inet前缀数字

# 将传入的IP列表转换为数组
IPs=($IP_LIST)

# 根据传入的前缀数字匹配本机IP
LOCAL_HOST=`ifconfig | grep "inet $INET_PREFIX" | awk '{print $2}'`
echo "本机匹配的IP: $LOCAL_HOST"

NPUS_PER_NODE=8
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6066
NNODES=${#IPs[@]}
NODE_RANK=""

# 查找当前节点在IP列表中的排名
for i in "${!IPs[@]}";
do
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
    then
        echo "Node Rank : ${i}"
        NODE_RANK=$i
        break
    fi
done

# 检查NODE_RANK是否获取成功
if [[ $NODE_RANK == "" ]];then
    echo "[Error] para \"NODE_RANK\" must be config"
    exit 1
fi

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="/home/dataset/longcat/datasets/train-00000-of-00042-d964455e17e96d5a_text_document"
TOKENIZER_PATH="/home/dataset/longcat/longcat-flash-chat"

TP=1
PP=1
EP=8
CP=1
NUM_LAYER=1

MBS=1
GBS=16
SEQ_LENGTH=8192
TRAIN_ITERS=10
ROUTER_BALANCING_TYPE='softmax_topk'
PROS_SAVE_PATH="A5_8die_longcat_8k_tp1pp1ep${EP}_bf16_profiling_layer${NUM_LAYER}-noswap"
time=$(date "+%Y%m%d-%H%M%S")

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 256 \
    --num-zero-experts 128 \
    --moe-router-topk 12 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-router-topk-scaling-factor 6.0 \
    --moe-ffn-hidden-size 2048 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff 0.001 \
    --fix-router
"
# --fix-router会影响精度

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --enable-mla-scale-q-lora \
    --enable-mla-scale-kv-lora \
    --mla-fa-without-pad \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --manual-gc \
    --manual-gc-interval 50 \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
"


MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --expert-tensor-parallel-size 1 \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.longcat_spec layer_spec \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers ${NUM_LAYER} \
    --hidden-size 6144 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
    --kv-channels 64 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 131072 \
    --rotary-base 10000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --no-bias-dropout-fusion \
    --transformer-impl transformer_engine \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"
# transformer-impl local/transformer_engine

FP8_ARGS="
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
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
    --no-load-rng
"

PROFILE_ARGS="
    --profile \
    --profile-step-start 6 \
    --profile-step-end 7 \
    --profile-ranks 0  \
    --profile-level level1  \
    --profile-with-cpu  \
    --profile-record-shapes  \
    --profile-save-path $PROS_SAVE_PATH \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $MLA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $PROFILE_ARGS \
    --distributed-backend nccl \
    --no-shared-storage \
    | tee ./logs/$PROS_SAVE_PATH--${time}.log