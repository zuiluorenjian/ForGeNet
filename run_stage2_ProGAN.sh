#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$REPO_DIR/train_stage2_msfe.py" \
  --epochs 1 \
  --batch_size 8 \
  --lr 4e-4 \
  --real_count 360 \
  --fake_count 360 \
  --msfe_hidden_dim 64 \
  --msfe_kernel_sizes 3 5 7 \
  --msfe_dropout 0.1 \
  --msfe_scale 0.1 \
  --num_adapted_layers 12 \
  --require_pretrain \
  "$@"
