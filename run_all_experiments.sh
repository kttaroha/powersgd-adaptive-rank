#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <rank> [start_from]"
  echo "  rank=0 on master node, rank=1 on worker node"
  echo "  start_from: index (0-based) or config filename (e.g., exp_xxx.yaml)"
  exit 1
fi

RANK="$1"
START_FROM="${2:-${RESUME_FROM:-}}"
WORLD_SIZE=${WORLD_SIZE:-2}
MASTER_ADDR=${MASTER_ADDR:-"150.65.114.112"}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-12355}
NCCL_IFNAME=${NCCL_IFNAME:-eth0}

# Keep this list explicit to avoid rank0/rank1 mismatch.
EXPERIMENT_CONFIGS=(
  configs/yyyymmdd/exp_p1_powerSGD_adaptive_epochnorm_nolimit_example.yaml
  configs/yyyymmdd/exp_p1_powerSGD_adaptive_epochnorm_nolimit_seed0_example.yaml
  configs/yyyymmdd/exp_p1_powerSGD_adaptive_epochnorm_nolimit_seed1_example.yaml
)

resolve_start_index() {
  local start_from="$1"
  local start_idx=0

  if [ -z "${start_from}" ]; then
    echo "${start_idx}"
    return 0
  fi

  if [[ "${start_from}" =~ ^[0-9]+$ ]]; then
    start_idx="${start_from}"
    echo "${start_idx}"
    return 0
  fi

  local cfg_name="${start_from}"
  if [[ "${cfg_name}" != *.yaml ]]; then
    cfg_name="${cfg_name}.yaml"
  fi

  local found_idx=-1
  local i
  for i in "${!EXPERIMENT_CONFIGS[@]}"; do
    if [ "${EXPERIMENT_CONFIGS[$i]}" = "${cfg_name}" ]; then
      found_idx="${i}"
      break
    fi
  done

  if [ "${found_idx}" -lt 0 ]; then
    echo "ERROR: start_from config not found: ${cfg_name}" >&2
    return 1
  fi

  echo "${found_idx}"
}

run_one_experiment() {
  local idx="$1"
  local cfg="${EXPERIMENT_CONFIGS[$idx]}"
  local master_port=$((MASTER_PORT_BASE + idx))

  echo "=== [RANK=${RANK}] Start experiment with ${cfg} ==="
  echo "=== [RANK=${RANK}] Using MASTER_PORT=${master_port} ==="

  NCCL_SOCKET_IFNAME="${NCCL_IFNAME}" python3 src/main.py \
    --config "${cfg}" \
    --rank "${RANK}" \
    --local_rank 0 \
    --world_size "${WORLD_SIZE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${master_port}" \
    --benchmark

  echo "=== [RANK=${RANK}] Finished experiment with ${cfg} ==="
}

start_idx="$(resolve_start_index "${START_FROM}")"
if [ "${start_idx}" -ge "${#EXPERIMENT_CONFIGS[@]}" ]; then
  echo "ERROR: start_from index out of range: ${start_idx}" >&2
  exit 1
fi

for ((idx=start_idx; idx<${#EXPERIMENT_CONFIGS[@]}; idx++)); do
  run_one_experiment "${idx}"
done

echo "=== [RANK=${RANK}] All experiments finished. ==="
