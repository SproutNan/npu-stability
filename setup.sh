#!/bin/bash
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export no_proxy="*.byted.org"

pip install -r MindSpeed/requirements.txt
pip install -e MindSpeed
