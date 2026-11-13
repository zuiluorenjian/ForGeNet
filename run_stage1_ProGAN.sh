#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$REPO_DIR/train_stage1.py" \
  --epochs 2 \
  --batch_size 128 \
  --lr 1e-4 \
  --use_lora \
  --lora_layers 4 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_lr 1e-4 \
  --loss_weighting learned \
  --gate_probe lora \
  --gate_lr 3e-4 \
  --patience 15 \
  "$@"
