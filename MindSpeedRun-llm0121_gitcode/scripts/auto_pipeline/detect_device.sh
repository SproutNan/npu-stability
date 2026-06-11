#!/bin/bash

detect_device_name() {
    # 尝试使用Python检测设备类型
    local device_name=$(python3 -c "
import torch_npu
try:
    device_name = torch_npu.npu.get_device_name()
    if 'Ascend910_95' in device_name or 'Ascend950' in device_name:
        print('A5')
    elif 'Ascend910_93' in device_name:
        print('A3')
    else:
        print('unknown')  # 默认值
except:
    print('unknown')  # 如果出错，返回默认值
" 2>/dev/null)

    # 如果Python命令失败，返回默认值
    if [ -z "$device_name" ]; then
        echo "unknown"
    else
        echo "$device_name"
    fi
}

# 如果直接执行此脚本，输出设备名称
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    detect_device_name
fi