#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  prepare_pipeline_workspace.sh --ips "ip1 ip2 ..." --inet-prefix 141 [options]

Required:
  --ips <IP_LIST>                  Space separated IP list.
  --inet-prefix <PREFIX>           Prefix used to resolve the local host IP.

Optional:
  --world-size <auto|8p|32p|64p>   Expected world size. Default: auto.
  --short-scenes <csv>             Short-run scene directories. Default: deepseek3_swap,longcat_swap,qwen3_235b_swap.
  --loss-scenes <csv>              Long-run loss scene directories. Default: deepseek3_noswap_8p_loss,longcat_loss,qwen3_235b_loss.
  --workspace-dir <path>           Workspace root. Default: current directory.
  --confirm-cleanup <true|false>   Must be true to allow destructive cleanup.
  --help                           Show this message.

Example:
  bash MindSpeedRun/scripts/auto_pipeline/prepare_pipeline_workspace.sh \
    --ips "141.62.24.228" \
    --inet-prefix 141 \
    --confirm-cleanup true
EOF
}

parse_bool() {
    case "${1,,}" in
        true|1|yes|y|on) echo true ;;
        false|0|no|n|off) echo false ;;
        *)
            echo "Error: invalid boolean value: $1" >&2
            exit 1
            ;;
    esac
}

IP_LIST=""
INET_PREFIX=""
WORLD_SIZE="auto"
SHORT_SCENES="deepseek3_swap,longcat_swap,qwen3_235b_swap"
LOSS_SCENES="deepseek3_noswap_8p_loss,longcat_loss,qwen3_235b_loss"
WORKSPACE_DIR="$(pwd)"
CONFIRM_CLEANUP="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips) IP_LIST="$2"; shift 2 ;;
        --inet-prefix) INET_PREFIX="$2"; shift 2 ;;
        --world-size) WORLD_SIZE="$2"; shift 2 ;;
        --short-scenes) SHORT_SCENES="$2"; shift 2 ;;
        --loss-scenes) LOSS_SCENES="$2"; shift 2 ;;
        --workspace-dir) WORKSPACE_DIR="$2"; shift 2 ;;
        --confirm-cleanup) CONFIRM_CLEANUP="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *)
            echo "Error: unsupported option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

[[ -n "$IP_LIST" ]] || { echo 'Error: --ips is required' >&2; usage; exit 1; }
[[ -n "$INET_PREFIX" ]] || { echo 'Error: --inet-prefix is required' >&2; usage; exit 1; }
CONFIRM_CLEANUP="$(parse_bool "$CONFIRM_CLEANUP")"

cd "$WORKSPACE_DIR"

echo "Workspace: $WORKSPACE_DIR"
echo "IP_LIST: $IP_LIST"
echo "INET_PREFIX: $INET_PREFIX"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SHORT_SCENES: $SHORT_SCENES"
echo "LOSS_SCENES: $LOSS_SCENES"

echo "The following destructive cleanup will be performed:"
echo "  rm -rf MindSpeed"
echo "  rm -rf Megatron-LM"
echo "  rm -rf MindSpeed-LLM"
echo "  rm -f /root/.gitconfig"

if [[ "$CONFIRM_CLEANUP" != "true" ]]; then
    echo "Cleanup not confirmed. Re-run with --confirm-cleanup true to continue." >&2
    exit 1
fi

rm -rf MindSpeed
rm -rf Megatron-LM
rm -rf MindSpeed-LLM
rm -f /root/.gitconfig

git config --global http.sslVerify false

bash MindSpeedRun/download_code.sh
if [[ ! -d mstt ]]; then
    git clone https://gitcode.com/Ascend/mstt.git
else
    echo "Reuse existing mstt directory: $(pwd)/mstt"
fi

pip uninstall -e MindSpeed || true
pip install -e MindSpeedRun
pip install -r MindSpeed/requirements.txt
pip install -e MindSpeed
pip install ijson
pip install Xlsxwriter
pip install prettytable

cd MindSpeed-LLM
pip install -r requirements.txt
pip install transformers==4.51.0
cd ..

bash MindSpeedRun/scripts/auto_pipeline/start_pipeline_tasks.sh \
    --ips "$IP_LIST" \
    --inet-prefix "$INET_PREFIX" \
    --world-size "$WORLD_SIZE" \
    --short-scenes "$SHORT_SCENES" \
    --loss-scenes "$LOSS_SCENES"