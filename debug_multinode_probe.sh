#!/bin/bash
set -uo pipefail

# Multi-node NPU/HCCL diagnostic entry.
# Run this from every executor through the same Arnold/Merlin user script.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

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

NNODES=${NNODES:-${ARNOLD_EXECUTOR_NUM:-${ARNOLD_NUM:-${ARNOLD_WORKER_NUM:-1}}}}
NODE_RANK=${NODE_RANK:-${ARNOLD_ID:-0}}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}
HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-${ARNOLD_RDMA_INTERFACE:-eth0}}
LOCAL_IPV4=$(detect_ipv4 "$HCCL_SOCKET_IFNAME")

MASTER_ADDR=${MASTER_ADDR:-${METIS_WORKER_0_HOST:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}}
MASTER_PORT=${MASTER_PORT:-${METIS_WORKER_0_PORT:-${ARNOLD_WORKER_0_PORT:-1234}}}

PROBE_ID=${PROBE_ID:-${ARNOLD_TRIAL_ID:-${ARNOLD_RUN_ID:-manual}}_${ARNOLD_START_TS:-0}}
SYNC_DIR=${MULTINODE_SYNC_DIR:-/mnt/hdfs/__INFRA_OUTPUT__/npu_debug/probe_${PROBE_ID}}
PROBE_TIMEOUT_SECONDS=${PROBE_TIMEOUT_SECONDS:-300}
PROBE_TRAIN_TIMEOUT_SECONDS=${PROBE_TRAIN_TIMEOUT_SECONDS:-1800}
PROBE_ITERS=${PROBE_ITERS:-5}
PROBE_RUN_TRAIN=${PROBE_RUN_TRAIN:-1}
PROBE_RUN_FAULT_TRAIN=${PROBE_RUN_FAULT_TRAIN:-1}
PROBE_TRAIN_ITERS=${PROBE_TRAIN_ITERS:-2}

if [[ -z "$LOCAL_IPV4" ]]; then
    echo "[PROBE][ERROR] failed to detect local IPv4 on ${HCCL_SOCKET_IFNAME}"
    exit 1
fi
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || ! [[ "$NPUS_PER_NODE" =~ ^[0-9]+$ ]]; then
    echo "[PROBE][ERROR] NNODES/NODE_RANK/NPUS_PER_NODE must be numeric: ${NNODES}/${NODE_RANK}/${NPUS_PER_NODE}"
    exit 1
fi
if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
    echo "[PROBE][WARN] invalid MASTER_PORT=${MASTER_PORT}; fallback to 1234"
    MASTER_PORT=1234
fi

mkdir -p "$SYNC_DIR"

write_marker() {
    {
        echo "date=$(date)"
        echo "node_rank=${NODE_RANK}"
        echo "arnold_id=${ARNOLD_ID:-}"
        echo "nnodes=${NNODES}"
        echo "host=$(hostname)"
        echo "hostname_i=$(hostname -i 2>/dev/null || true)"
        echo "hostname_I=$(hostname -I 2>/dev/null || true)"
        echo "local_ipv4=${LOCAL_IPV4}"
        echo "master_addr_before=${MASTER_ADDR}"
        echo "master_port_before=${MASTER_PORT}"
        echo "hccl_socket_ifname=${HCCL_SOCKET_IFNAME}"
        echo "arnold_executor_num=${ARNOLD_EXECUTOR_NUM:-}"
        echo "arnold_num=${ARNOLD_NUM:-}"
        echo "metis_worker_0_host=${METIS_WORKER_0_HOST:-}"
        echo "metis_worker_0_port=${METIS_WORKER_0_PORT:-}"
    } > "${SYNC_DIR}/node_${NODE_RANK}.env"
}

wait_for_count() {
    local pattern="$1"
    local expected="$2"
    local seconds="$3"
    local waited=0
    local count=0
    while (( waited < seconds )); do
        count=$(find "$SYNC_DIR" -maxdepth 1 -name "$pattern" 2>/dev/null | wc -l | awk '{print $1}')
        if (( count >= expected )); then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "[PROBE][ERROR] timeout waiting for ${expected} files matching ${pattern}; got ${count}"
    return 1
}

write_marker
wait_for_count "node_*.env" "$NNODES" 180 || true

MASTER_FILE="${SYNC_DIR}/master_ipv4"
MASTER_PORT_FILE="${SYNC_DIR}/master_port"
if [[ "$NODE_RANK" == "0" ]]; then
    printf "%s\n" "$LOCAL_IPV4" > "$MASTER_FILE"
    printf "%s\n" "$MASTER_PORT" > "$MASTER_PORT_FILE"
fi

wait_for_count "master_ipv4" 1 180 || exit 1
wait_for_count "master_port" 1 180 || exit 1
MASTER_ADDR=$(head -n 1 "$MASTER_FILE" | tr -d '[:space:]')
MASTER_PORT=$(head -n 1 "$MASTER_PORT_FILE" | tr -d '[:space:]')

if [[ -z "$MASTER_ADDR" ]] || is_ipv6_addr "$MASTER_ADDR"; then
    echo "[PROBE][ERROR] resolved MASTER_ADDR is invalid: ${MASTER_ADDR}"
    exit 1
fi
if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
    echo "[PROBE][ERROR] resolved MASTER_PORT is invalid: ${MASTER_PORT}"
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export CUDA_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME
export HCCL_IF_IP=$LOCAL_IPV4
export HCCL_WHITELIST_DISABLE=${HCCL_WHITELIST_DISABLE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-300}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
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
export PROBE_ITERS

cat > "${SYNC_DIR}/node_${NODE_RANK}.resolved" <<EOF
node_rank=${NODE_RANK}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
hccl_if_ip=${HCCL_IF_IP}
hccl_socket_ifname=${HCCL_SOCKET_IFNAME}
sync_dir=${SYNC_DIR}
EOF

echo "============================================"
echo "[PROBE] multinode NPU diagnostic"
echo "  nnodes:       ${NNODES}"
echo "  node_rank:    ${NODE_RANK}"
echo "  npu/node:     ${NPUS_PER_NODE}"
echo "  master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "  hccl:         HCCL_IF_IP=${HCCL_IF_IP}, HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}"
echo "  sync_dir:     ${SYNC_DIR}"
echo "============================================"

PROBE_PY="/tmp/npu_dist_probe_${PROBE_ID}_${NODE_RANK}.py"
cat > "$PROBE_PY" <<'PY'
import argparse
import datetime
import os
import sys

import torch
import torch.distributed as dist
import torch_npu


parser = argparse.ArgumentParser()
parser.add_argument("op", choices=["barrier", "all_reduce", "all_to_all"])
parser.add_argument("--local-rank", "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
args, _ = parser.parse_known_args()

torch.npu.set_device(args.local_rank)
timeout = datetime.timedelta(seconds=int(os.environ.get("DIST_TIMEOUT_SECONDS", "120")))
dist.init_process_group("hccl", timeout=timeout)

rank = dist.get_rank()
world = dist.get_world_size()
iters = int(os.environ.get("PROBE_ITERS", "5"))

if rank == 0:
    print(f"[probe:{args.op}] world={world}", flush=True)

if args.op == "barrier":
    for i in range(iters):
        dist.barrier()
        if rank == 0:
            print(f"[probe:barrier] step={i} ok", flush=True)

elif args.op == "all_reduce":
    x = torch.ones(1024, device="npu", dtype=torch.float32) * (rank + 1)
    expected = world * (world + 1) / 2.0
    for i in range(iters):
        dist.all_reduce(x)
        torch.npu.synchronize()
        if rank == 0:
            print(f"[probe:all_reduce] step={i} value={float(x[0].item())} expected={expected}", flush=True)
        x.fill_(rank + 1)

elif args.op == "all_to_all":
    x = torch.full((world, 4), rank, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    for i in range(iters):
        dist.all_to_all_single(y, x)
        torch.npu.synchronize()
        if rank == 0:
            print(f"[probe:all_to_all] step={i} first={float(y[0, 0].item())} last={float(y[-1, 0].item())}", flush=True)

dist.barrier()
dist.destroy_process_group()
PY

status_file_for() {
    local stage="$1"
    printf "%s/%s_node_%s.status" "$SYNC_DIR" "$stage" "$NODE_RANK"
}

wait_stage_statuses() {
    local stage="$1"
    local waited=0
    local count=0
    while (( waited < 180 )); do
        count=$(find "$SYNC_DIR" -maxdepth 1 -name "${stage}_node_*.status" 2>/dev/null | wc -l | awk '{print $1}')
        if (( count >= NNODES )); then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    if (( count < NNODES )); then
        echo "[PROBE][ERROR] stage=${stage} only ${count}/${NNODES} nodes wrote status"
        return 1
    fi

    local failed=0
    local f
    for f in "$SYNC_DIR"/"${stage}"_node_*.status; do
        if [[ "$(cat "$f")" != "0" ]]; then
            failed=1
        fi
    done
    if (( failed )); then
        echo "[PROBE][ERROR] stage=${stage} failed. Status files:"
        cat "$SYNC_DIR"/"${stage}"_node_*.status
        return 1
    fi
    echo "[PROBE] stage=${stage} passed on all nodes"
    return 0
}

port_for_offset() {
    local offset="$1"
    echo $((10#$MASTER_PORT + offset))
}

run_stage() {
    local stage="$1"
    local seconds="$2"
    shift 2
    local log_file="${SYNC_DIR}/${stage}_node_${NODE_RANK}.log"
    echo "[PROBE] start stage=${stage}; log=${log_file}"
    timeout -k 30s "$seconds" "$@" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    echo "$status" > "$(status_file_for "$stage")"
    wait_stage_statuses "$stage"
}

launch_probe() {
    local op="$1"
    local port="$2"
    python -m torch.distributed.launch \
        --nproc_per_node "$NPUS_PER_NODE" \
        --nnodes "$NNODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$port" \
        "$PROBE_PY" "$op"
}

run_stage barrier "$PROBE_TIMEOUT_SECONDS" launch_probe barrier "$(port_for_offset 0)" || exit 10
run_stage all_reduce "$PROBE_TIMEOUT_SECONDS" launch_probe all_reduce "$(port_for_offset 1)" || exit 11
run_stage all_to_all "$PROBE_TIMEOUT_SECONDS" launch_probe all_to_all "$(port_for_offset 2)" || exit 12

if [[ "$PROBE_RUN_TRAIN" == "1" ]]; then
    run_stage train_no_fault "$PROBE_TRAIN_TIMEOUT_SECONDS" \
        bash train_qwen3_moe_v4_multinode.sh \
            --nnodes "$NNODES" \
            --node-rank "$NODE_RANK" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$(port_for_offset 3)" \
            --train-iters "$PROBE_TRAIN_ITERS" \
            --no-fault \
            --mbs 3 \
            --gbs 384 || exit 13
fi

if [[ "$PROBE_RUN_FAULT_TRAIN" == "1" ]]; then
    run_stage train_fault_o8_do8 "$PROBE_TRAIN_TIMEOUT_SECONDS" \
        bash train_qwen3_moe_v4_multinode.sh \
            --nnodes "$NNODES" \
            --node-rank "$NODE_RANK" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$(port_for_offset 4)" \
            --train-iters "$PROBE_TRAIN_ITERS" \
            --bits-o 8 \
            --bits-do 8 \
            --mbs 3 \
            --gbs 384 || exit 14
fi

echo "[PROBE] all requested stages passed"
