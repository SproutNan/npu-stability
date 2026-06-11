#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./detect_device.sh
source "${SCRIPT_DIR}/detect_device.sh"

usage() {
    cat <<'EOF'
Usage:
  run_training_pipeline.sh <pipeline_dir> <sub_task> <log_dir> <full_training> <ips> [inet_prefix] [world_size_override]
EOF
}

if [[ $# -lt 5 ]]; then
    usage
    exit 1
fi

pipeline_dir=$1
sub_task=$2
log_dir=$3
full_training=$4
ips=$5
inet_prefix=${6:-141}
world_size_override=${7:-auto}

DEVICE_NAME=$(detect_device_name)
echo "Detected device type: $DEVICE_NAME"

mindspeed_llm_dir="$pipeline_dir/MindSpeed-LLM"
source_scripts_dir="$pipeline_dir/MindSpeedRun/scripts"
sub_task_dir="$mindspeed_llm_dir/scripts/$sub_task"

if [[ ! -d "$source_scripts_dir" ]]; then
    echo "Error: source scripts dir not found: $source_scripts_dir"
    exit 1
fi

rm -rf "$mindspeed_llm_dir/scripts"
cp -r "$source_scripts_dir" "$mindspeed_llm_dir/"

if [[ ! -d "$sub_task_dir" ]]; then
    echo "Skip $sub_task because directory does not exist: $sub_task_dir"
    exit 0
fi

ip_count=$(echo "$ips" | wc -w)
inferred_parallelism=""
if [[ "$DEVICE_NAME" == "A3" ]]; then
    case "$ip_count" in
        1) inferred_parallelism="8p" ;;
        2) inferred_parallelism="32p" ;;
        4) inferred_parallelism="64p" ;;
        *)
            echo "Error: A3 supports only 1/2/4 IPs for 8p/32p/64p, got ${ip_count}"
            exit 1
            ;;
    esac
else
    case "$ip_count" in
        1) inferred_parallelism="8p" ;;
        4) inferred_parallelism="32p" ;;
        8) inferred_parallelism="64p" ;;
        *)
            echo "Error: device supports only 1/4/8 IPs for 8p/32p/64p, got ${ip_count}"
            exit 1
            ;;
    esac
fi

target_parallelism="$inferred_parallelism"
if [[ "$world_size_override" != "auto" ]]; then
    case "$world_size_override" in
        8p|32p|64p) ;;
        *)
            echo "Error: unsupported world size override: $world_size_override"
            exit 1
            ;;
    esac
    if [[ "$world_size_override" != "$inferred_parallelism" ]]; then
        echo "Error: requested world size ${world_size_override} does not match inferred ${inferred_parallelism} for device ${DEVICE_NAME} and ip_count ${ip_count}"
        exit 1
    fi
    target_parallelism="$world_size_override"
fi

mkdir -p "${log_dir}/${sub_task}" "${log_dir}/${sub_task}/profiles/${DEVICE_NAME}"
cd "$mindspeed_llm_dir"

model_family=unknown
case "$sub_task" in
    qwen3_235b*) model_family=qwen ;;
    deepseek3*) model_family=deepseek3 ;;
    longcat*) model_family=longcat ;;
esac

uses_scene_directory=false
case "$sub_task" in
    qwen3_235b_*|deepseek3_*|longcat_*) uses_scene_directory=true ;;
esac

copy_scene_artifacts() {
    local task_name=$1
    local target_script=$2
    local profiling_dir
    profiling_dir=$(find "${log_dir}/${sub_task}/profiles/${DEVICE_NAME}" -maxdepth 1 -type d -name "${DEVICE_NAME}_${sub_task}_${task_name}*" | sort -r | head -n1 || true)
    if [[ -n "$profiling_dir" && -d "$profiling_dir" ]]; then
        if [[ -f "${log_dir}/${sub_task}/${task_name}.log" ]]; then
            mv "${log_dir}/${sub_task}/${task_name}.log" "$profiling_dir/"
        fi
        cp "$target_script" "$profiling_dir/"
    fi
}

prepare_non_qwen_script() {
    local target_script=$1
    local task_name=$2

    if [[ "$DEVICE_NAME" == "A3" ]]; then
        sed -i 's|NPUS_PER_NODE=.*|NPUS_PER_NODE=16|g' "$target_script"
    fi

    if grep -q '^PROS_SAVE_PATH=' "$target_script"; then
        sed -i "s|^PROS_SAVE_PATH=.*|PROS_SAVE_PATH=\"${log_dir}/${sub_task}/profiles/${DEVICE_NAME}/${DEVICE_NAME}_${sub_task}_${task_name}_rank\${NODE_RANK}\"|" "$target_script"
    fi
    sed -i 's|--profile-save-path \\.\/|--profile-save-path |g' "$target_script"
    sed -i 's/[[:space:]]*2>&1[[:space:]]*|[[:space:]]*tee[^|]*$//' "$target_script"
    sed -i 's/[[:space:]]*|[[:space:]]*tee[^|]*$//' "$target_script"

    if [[ "$uses_scene_directory" == false ]]; then
        sed -i '/\$GPT_ARGS/a\    --swap-optimizer \\' "$target_script"

        if [[ "$full_training" == "True" ]]; then
            sed -i '/--fix-router/d' "$target_script"
            sed -i '/\$PROFILE_ARGS[[:space:]]*\\/d' "$target_script"
            sed -i '/--exit-interval/d' "$target_script"
            sed -i 's/--train-iters[[:space:]]\+[^[:space:]\\]\+/--train-iters 2000/g' "$target_script"
        elif [[ "$full_training" != "False" ]]; then
            echo "Error: Unsupported training method $full_training"
            exit 1
        fi
    fi
}

has_runnable_tasks=false
for target_script in "$sub_task_dir"/*.sh; do
    [[ -f "$target_script" ]] || continue
    file_name=${target_script##*/}
    case "$file_name" in
        qwen3_235b.sh|deepseek3.sh|longcat.sh) continue ;;
    esac

    task_name="${file_name%.*}"
    if [[ "$DEVICE_NAME" == "A3" && "$task_name" == *"fp8"* ]]; then
        continue
    fi
    if [[ "$task_name" != *"$target_parallelism"* ]]; then
        echo "Skip $task_name (ip_count=${ip_count}, target_parallelism=${target_parallelism})"
        continue
    fi

    has_runnable_tasks=true
    echo "------------------ Task $sub_task $task_name Start ------------------"
    echo "==================== $sub_task $task_name ====================" >> "${log_dir}/result.log"
    echo "[Start time] $(date '+%Y/%m/%d %H:%M:%S')" >> "${log_dir}/result.log"

    case "$model_family" in
        qwen)
            extra_args=(
                --profile-save-root "${log_dir}/${sub_task}/profiles/${DEVICE_NAME}"
                --log-root "${log_dir}/${sub_task}"
            )
            bash "./scripts/${sub_task}/${file_name}" "$ips" "$inet_prefix" "${extra_args[@]}" 2>&1 | tee "${log_dir}/${sub_task}/${task_name}.log"
            copy_scene_artifacts "$task_name" "$target_script"
            ;;
        deepseek3|longcat)
            prepare_non_qwen_script "$target_script" "$task_name"
            bash "./scripts/${sub_task}/${file_name}" "$ips" "$inet_prefix" 2>&1 | tee "${log_dir}/${sub_task}/${task_name}.log"
            copy_scene_artifacts "$task_name" "$target_script"
            ;;
        *)
            echo "Skip unsupported task family: $sub_task"
            ;;
    esac

    echo "[End time] $(date '+%Y/%m/%d %H:%M:%S')" >> "${log_dir}/result.log"
    echo >> "${log_dir}/result.log"
    echo "------------------ Task $sub_task $task_name Done ------------------"
done

if [[ "$has_runnable_tasks" == false ]]; then
    echo "Skip $sub_task because no runnable task matched the current device/world-size"
fi