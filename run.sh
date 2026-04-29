#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/train.yaml"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${TIMESTAMP}_training.log"

echo "============================================"
echo " 3D Visual Grounding - Training"
echo " Conda env : sonata"
echo " Config    : $CONFIG"
echo " Overrides : $*"
echo " Log file  : $LOG_FILE"
echo "============================================"

mkdir -p logs

python main.py --config "$CONFIG" "$@" > "$LOG_FILE" 2>&1