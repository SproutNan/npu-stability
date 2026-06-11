#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./detect_device.sh
source "${SCRIPT_DIR}/detect_device.sh"

usage() {
    cat <<'EOF'
Usage:
  start_pipeline_tasks.sh --ips "ip1 ip2 ..." --inet-prefix 141 [options]

Required:
  --ips <IP_LIST>                  Space separated IP list.
  --inet-prefix <PREFIX>           Prefix used to resolve the local host IP.

Optional:
  --world-size <auto|8p|32p|64p>   Expected world size. Default: auto.
  --short-scenes <csv>             Short-run scene directories. Default: deepseek3_swap,longcat_swap,qwen3_235b_swap.
  --loss-scenes <csv>              Long-run loss scene directories. Default: deepseek3_noswap_8p_loss,longcat_loss,qwen3_235b_loss.
  --pipeline-dir <path>            Pipeline root. Default: auto-detected.
  --log-dir <path>                 Log root. Default: logs/<DEVICE>/<time> under pipeline root.
  --skip-analysis <true|false>     Default: false.
  --help                           Show this message.
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

csv_to_array() {
    local raw=$1
    local -n ref=$2
    local item
    IFS=',' read -r -a ref <<< "$raw"
    for item in "${!ref[@]}"; do
        ref[$item]="$(echo "${ref[$item]}" | xargs)"
    done
}

IP_LIST=""
INET_PREFIX=""
WORLD_SIZE="auto"
SHORT_SCENES_RAW="deepseek3_swap,longcat_swap,qwen3_235b_swap"
LOSS_SCENES_RAW="deepseek3_noswap_8p_loss,longcat_loss,qwen3_235b_loss"
PIPELINE_DIR=""
LOG_DIR=""
SKIP_ANALYSIS="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ips) IP_LIST="$2"; shift 2 ;;
        --inet-prefix) INET_PREFIX="$2"; shift 2 ;;
        --world-size) WORLD_SIZE="$2"; shift 2 ;;
        --short-scenes) SHORT_SCENES_RAW="$2"; shift 2 ;;
        --loss-scenes) LOSS_SCENES_RAW="$2"; shift 2 ;;
        --pipeline-dir) PIPELINE_DIR="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --skip-analysis) SKIP_ANALYSIS="$2"; shift 2 ;;
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
SKIP_ANALYSIS="$(parse_bool "$SKIP_ANALYSIS")"

DEVICE_NAME=$(detect_device_name)
echo "Detected device type: $DEVICE_NAME"

current_dir=$(dirname "$(readlink -f "$0")")
if [[ -z "$PIPELINE_DIR" ]]; then
    PIPELINE_DIR=$(dirname "$(dirname "$(dirname "$current_dir")")")
fi
source_scripts_dir="$PIPELINE_DIR/MindSpeedRun/scripts"

if [[ -z "$LOG_DIR" ]]; then
    date_str="$(date +'%m-%d-%H-%M')"
    LOG_DIR="$PIPELINE_DIR/logs/${DEVICE_NAME}/$date_str"
fi
mkdir -p "$LOG_DIR"

short_scenes=()
loss_scenes=()
csv_to_array "$SHORT_SCENES_RAW" short_scenes
csv_to_array "$LOSS_SCENES_RAW" loss_scenes

scene_exists() {
    local dir_name=$1
    [[ -d "$source_scripts_dir/$dir_name" ]]
}

run_scene_list() {
    local phase_name=$1
    shift
    local scene
    echo "===== ${phase_name} ====="
    for scene in "$@"; do
        [[ -n "$scene" ]] || continue
        if scene_exists "$scene"; then
            bash "${current_dir}/run_training_pipeline.sh" "$PIPELINE_DIR" "$scene" "$LOG_DIR" False "$IP_LIST" "$INET_PREFIX" "$WORLD_SIZE"
        else
            echo "Skip ${scene} because directory does not exist under $source_scripts_dir"
        fi
    done
}

run_scene_list "Short-Run Scenes" "${short_scenes[@]}"

A3_PROS_PATH="${A3_PROS_PATH:-${LOG_DIR}}"
A5_PROS_PATH="${A5_PROS_PATH:-${LOG_DIR}}"
MSTT_PATH="${MSTT_PATH:-${PIPELINE_DIR}/mstt}"
ANALYSIS_OUTPUT_PATH="${ANALYSIS_OUTPUT_PATH:-${LOG_DIR}/analysis_out/}"

if [[ "$SKIP_ANALYSIS" != "true" ]]; then
    a3_ready=$(find "${A3_PROS_PATH}" -type d -path '*/profiles/A3' | head -n1 || true)
    a5_ready=$(find "${A5_PROS_PATH}" -type d -path '*/profiles/A5' | head -n1 || true)

    if [[ -n "$a3_ready" && -n "$a5_ready" ]]; then
        echo "===== Profiling Analysis ====="
        bash "${current_dir}/run_profiling_analysis.sh" \
            --a3-pros-path "${A3_PROS_PATH}" \
            --a5-pros-path "${A5_PROS_PATH}" \
            --mstt-path "${MSTT_PATH}" \
            --output-path "${ANALYSIS_OUTPUT_PATH}"
    else
        echo "Skip profiling analysis because A3/A5 profile roots are not both ready"
        echo "A3 path: ${A3_PROS_PATH}"
        echo "A5 path: ${A5_PROS_PATH}"
    fi
else
    echo "Skip profiling analysis because --skip-analysis true"
fi

run_scene_list "Loss Scenes" "${loss_scenes[@]}"