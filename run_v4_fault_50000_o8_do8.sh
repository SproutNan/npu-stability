#!/bin/bash
set -euo pipefail

cd /opt/tiger

if [ -d npu-stability/.git ]; then
    cd npu-stability
    git pull --ff-only
else
    git clone https://github.com/SproutNan/npu-stability.git
    cd npu-stability
fi

bash setup.sh

LR=${LR:-5.0e-4}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}
TRAIN_ITERS=${TRAIN_ITERS:-50000}
BITS_O=${BITS_O:-8}
BITS_DO=${BITS_DO:-8}

bash train_qwen3_moe_v4.sh \
    --lr "$LR" \
    --aux-loss "$MOE_AUX_LOSS_COEFF" \
    --train-iters "$TRAIN_ITERS" \
    --bits-o "$BITS_O" \
    --bits-do "$BITS_DO" \
    "$@"
