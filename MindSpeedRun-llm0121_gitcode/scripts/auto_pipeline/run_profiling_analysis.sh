#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 导入设备检测函数
source "${SCRIPT_DIR}/detect_device.sh"

# 自动检测设备名称
DEVICE_NAME=$(detect_device_name)
echo "检测到设备类型: $DEVICE_NAME"

# Profiling批量分析工具 - 一键启动

# 设置默认路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
A3_PROS_PATH="${SCRIPT_DIR}/../pros/A3"
A5_PROS_PATH="${SCRIPT_DIR}/../pros/A5"
MSTT_PATH="${SCRIPT_DIR}/../mstt"
OUTPUT_PATH="${SCRIPT_DIR}/../pros/output"

# 检查是否存在配置文件
CONFIG_FILE="${SCRIPT_DIR}/config.ini"
if [ -f "${CONFIG_FILE}" ]; then
    echo "读取配置文件: config.ini"
    # 读取配置文件
    while IFS='=' read -r key value; do
        # 跳过注释和空行
        if [[ ! $key =~ ^# && ! -z $key ]]; then
            # 去除前后空格
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            # 设置变量
            declare "$key=$value"
        fi
    done < "${CONFIG_FILE}"
    echo ""
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --a3-pros-path)
            A3_PROS_PATH="$2"
            shift 2
            ;;
        --a5-pros-path)
            A5_PROS_PATH="$2"
            shift 2
            ;;
        --mstt-path)
            MSTT_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "错误: 不支持的参数: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Profiling批量分析工具 - 一键启动"
echo "============================================================"
echo ""

echo "输入路径:"
echo "  A3 Pros路径: ${A3_PROS_PATH}"
echo "  A5 Pros路径: ${A5_PROS_PATH}"
echo "  MSTT路径: ${MSTT_PATH}"
echo "  输出路径: ${OUTPUT_PATH}"
echo ""

# 检查路径是否存在
if [ ! -d "${A3_PROS_PATH}" ]; then
    echo "错误: A3 Pros路径不存在: ${A3_PROS_PATH}"
    echo ""
    echo "提示: 请检查config.ini中的A3_PROS_PATH配置"
    exit 1
fi

if [ ! -d "${A5_PROS_PATH}" ]; then
    echo "错误: A5 Pros路径不存在: ${A5_PROS_PATH}"
    echo ""
    echo "提示: 请检查config.ini中的A5_PROS_PATH配置"
    exit 1
fi

if [ ! -d "${MSTT_PATH}" ]; then
    echo "错误: MSTT路径不存在: ${MSTT_PATH}"
    echo ""
    echo "提示: 请检查config.ini中的MSTT_PATH配置"
    exit 1
fi

echo "路径检查通过！"
echo ""
echo "============================================================"
echo "开始批量分析..."
echo "============================================================"
echo ""

# 运行批量分析
python batch_profiling_analyzer.py \
    --a3-pros-path "${A3_PROS_PATH}" \
    --a5-pros-path "${A5_PROS_PATH}" \
    --mstt-path "${MSTT_PATH}" \
    --output-path "${OUTPUT_PATH}"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "批量分析完成！"
    echo "============================================================"
    echo ""
    echo "输出文件位于: ${OUTPUT_PATH}"
    echo ""
    echo "查看结果:"
    echo "  - 汇总报告: ${OUTPUT_PATH}/分析汇总.csv"
    echo "  - 各模型结果: ${OUTPUT_PATH}/{模型名}/"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "批量分析失败！"
    echo "============================================================"
    echo ""
    echo "请检查:"
    echo "  1. 路径配置是否正确"
    echo "  2. profiling数据是否完整"
    echo "  3. mstt是否正确安装"
    echo ""
fi
