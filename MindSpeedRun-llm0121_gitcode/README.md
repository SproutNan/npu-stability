# MindSpeedRun

#### 介绍
MindSpeedRun

#### 使用说明
```shell
# 基于gitcode主仓0121分支
git clone -b llm0121_gitcode https://gitcode.com/RyanWang1022/MindSpeedRun.git

# pod上执行，需规避部分问题
# 执行download_code脚本可自动拉取gitcode主仓0121代码并打patch
bash MindSpeedRun/download_code.sh
```
#### patch说明
MindSpeed-LLM/mindspeed_llm/features_manager/megatron_basic/transformer_engine_basic.py文件的修改为指定AICPU/CCU/CCUMS模式。
默认为CCU调度模式。
如果需要修改为AICPU模式，将pm.register_patch的两行取消注释。
如果需要修改为ccums模式，将pm.register_patch的两行取消注释，并将其中的`"hccl_op_expansion_mode":2`改为`"hccl_op_expansion_mode":5`。