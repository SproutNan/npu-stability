#!/bin/bash
set -euo pipefail

# Compatibility wrapper.
# The unified single-node entry point now lives in train_qwen3_moe_v4.sh.
# Examples:
#   bash train_qwen3_moe_v4_fault.sh --bits-o 7 --bits-do 7
#   BITS_O=7 BITS_DO=0 bash train_qwen3_moe_v4_fault.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/train_qwen3_moe_v4.sh" "$@"
