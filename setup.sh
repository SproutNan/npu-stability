#!/bin/bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install -r MindSpeed/requirements.txt
pip install -e MindSpeed
