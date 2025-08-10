#!/bin/bash

# W&Bエージェントを常駐監視し、自動再起動する簡易スーパーバイザ
# 前提: wandb CLI が PATH 上にあり、WANDB_ENTITY/PROJECT/SWEEP_ID が設定済み

set -euo pipefail

LOG_DIR="$(pwd)/output/latest"
mkdir -p "$LOG_DIR"
AGENT_LOG="$LOG_DIR/wandb_agent.log"
PID_FILE="$LOG_DIR/wandb_agent.pid"

WANDB_ENTITY="${WANDB_ENTITY:-nakanishi-shuntaro-638-kyushu-university}"
WANDB_PROJECT="${WANDB_PROJECT:-aichallenge-2025}"
SWEEP_ID_FILE="$LOG_DIR/wandb_sweep_id"

if [ -z "${SWEEP_ID:-}" ]; then
  if [ -f "$SWEEP_ID_FILE" ]; then
    SWEEP_ID="$(cat "$SWEEP_ID_FILE" | tr -d '\n' | tr -d '\r')"
  else
    echo "ERROR: SWEEP_ID is not set and $SWEEP_ID_FILE not found" >&2
    exit 1
  fi
fi

CMD=(wandb agent "${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}")

echo "[supervisor] Starting W&B agent supervisor for ${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
echo "[supervisor] Logs: $AGENT_LOG, PID: $PID_FILE"

restart_delay_initial=5
restart_delay_max=60
restart_delay=$restart_delay_initial

trap 'echo "[supervisor] Caught signal, stopping"; exit 0' INT TERM

while true; do
  echo "[supervisor] Launching agent: ${CMD[*]}"
  # ENABLE_AUTO_STOP/SPEED_* 環境変数は監視側（Python）でも参照するため、ここで渡す
  ( "${CMD[@]}" ) >>"$AGENT_LOG" 2>&1 &
  agent_pid=$!
  echo "$agent_pid" >"$PID_FILE"
  echo "[supervisor] Agent started with PID $agent_pid"

  # プロセス終了待ち
  wait $agent_pid
  exit_code=$?
  echo "[supervisor] Agent exited with code $exit_code"

  # 短時間での再起動スパムを避ける指数バックオフ
  echo "[supervisor] Restarting in ${restart_delay}s..."
  sleep $restart_delay
  if [ $restart_delay -lt $restart_delay_max ]; then
    restart_delay=$((restart_delay * 2))
    if [ $restart_delay -gt $restart_delay_max ]; then
      restart_delay=$restart_delay_max
    fi
  fi
done


