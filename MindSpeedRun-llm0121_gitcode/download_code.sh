#! /bin/bash
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

# 在当前目录拉取代码并应用patch
# MindSpeed
git clone --depth 1 https://gitcode.com/ascend/MindSpeed.git -b master

# Megatron
git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git -b core_v0.12.1

# LLM
git clone https://gitcode.com/ascend/MindSpeed-LLM.git 
cd MindSpeed-LLM
cp -r ../Megatron-LM/megatron ./
git apply ../MindSpeedRun/patch/MindSpeed-LLM.patch
