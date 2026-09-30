#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
export PYTHONPATH=/workspace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

out=outputs/evals/pretrain65-d-scaled-final-20260929
config=outputs/diagnostics/pretrain65-d-20260929/production-config.json
checkpoint=outputs/checkpoints/starscream-pretrain65-d-scaled-20260929/best-step-029465874-selection_suite_timely_success-0.7375.pt
mkdir -p "$out"
sha256sum "$config" "$checkpoint" configs/eval/v6_22_real60.manifest.json \
  configs/eval/real100_v2_timed_protocol_v1.json > "$out/input-sha256.txt"

python -u scripts/eval_privileged_dagger.py \
  --config "$config" --checkpoint "$checkpoint" --section dagger \
  --curriculum validation --track-suite configs/eval/v6_22_real60.yaml \
  --matched-track-seeds --episodes 32 --seed 62422001 \
  --randomized-environment --max-steps 6000 --workers 16 --device cuda \
  --output "$out/real60-e32.json" > "$out/real60.log" 2>&1

python -u scripts/audits/eval_privileged_real100_v2_dual.py \
  --config "$config" --checkpoint "$checkpoint" \
  --protocol configs/eval/real100_v2_timed_protocol_v1.json \
  --output "$out/real100-v2-e32.json" --episodes 32 --seed 2034091462 \
  --workers 16 --device cuda --max-steps 6000 \
  --retry-steps 6001 --dwell-steps 6001 --reference-aware \
  > "$out/real100-v2.log" 2>&1
