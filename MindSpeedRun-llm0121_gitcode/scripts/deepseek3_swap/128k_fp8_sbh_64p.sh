#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export CPU_AFFINITY_CONF=1,npu0:192-215,npu1:216-239,npu2:0-23,npu3:24-47,npu4:48-71,npu5:72-95,npu6:240-263,npu7:264-287
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
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
MASTER_PORT=6300
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
time=$(date +%Y%m%d%H%M)

DATA_PATH="/home/dataset/deepseek3/enwiki_text_document"
TOKENIZER_PATH="/home/dataset/deepseek3"

TP=1
PP=1
EP=64
CP=64
CP_TYPE=ulysses_cp_algo
NUM_LAYERS=12
SEQ_LEN=131072
MBS=1
GBS=16
PROS_SAVE_PATH="./A5_pretrain_deepseek_128k_sbh_uly_sub16k_fp8_64p_ep${EP}_cp${CP}_gbs${GBS}"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

FP8="
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --transformer-impl transformer_engine
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-fa-without-pad \
    --mla-mm-split \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --moe-shared-expert-intermediate-size 2048 \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 2048 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-aux-loss-coeff 0.0001 \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --fix-router \
"


MEM_ARGS="
    --recompute-activation-function \
    --swap-optimizer \
"

ROPE_ARGS="
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim 1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --no-check-for-nan-in-loss-and-grad \
    --fix-sub-seq-length 16384 \
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --transformer-impl transformer_engine \
    --manual-gc \
    --manual-gc-interval 50 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --expert-tensor-parallel-size 1 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --swiglu \
    --shape-order BNSD \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --distributed-timeout-minutes 120 \
    --no-gradient-accumulation-fusion \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
"

CKPT_ARGS="
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    --no-shared-storage \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.0e-5 \
    --train-iters 10 \
    --lr-decay-style cosine \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --lr-warmup-iters 0 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

PROFILE_ARGS="
    --profile \
    --profile-step-start 6 \
    --profile-step-end 7 \
    --profile-ranks 0 \
    --profile-level level1 \
    --profile-with-cpu  \
    --profile-record-shapes  \
    --profile-save-path $PROS_SAVE_PATH \
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $PROFILE_ARGS \
    $FP8 \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $MEM_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $CKPT_ARGS \
    $TRAIN_ARGS \
    --distributed-backend nccl \
    | tee ./logs/$PROS_SAVE_PATH.log
