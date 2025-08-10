#!/bin/bash
AWSIM_DIRECTORY=/aichallenge/simulator/AWSIM
LOGFILE=/output/latest/AWSIM_Player.log

mode="${1}"

if command -v nvidia-smi &>/dev/null && [[ -e /dev/nvidia0 ]]; then
    echo "[INFO] NVIDIA GPU detected"
    # GPU利用時もUnityのログを取得
    opts=("-logfile" "${LOGFILE}")
else
    echo "[INFO] No NVIDIA GPU detected → running on headless mode"
    # ヘッドレス安定化: batchmode/nographics を付与しログを明示保存
    opts=("-headless" "-batchmode" "-nographics" "-logfile" "${LOGFILE}")
fi

case "${mode}" in
"endless")
    opts+=(" --endless")
    ;;
*) ;;
esac

# shellcheck disable=SC1091
source /aichallenge/workspace/install/setup.bash

# ネットワーク設定（rootならsudo不要）
if [ "$(id -u)" -eq 0 ]; then
    ip link set multicast on lo || true
    sysctl -w net.core.rmem_max=2147483647 >/dev/null || true
else
    sudo ip link set multicast on lo || true
    sudo sysctl -w net.core.rmem_max=2147483647 >/dev/null || true
fi

# Unity側のコアダンプを抑止（巨大ファイル生成回避）
ulimit -c 0 || true
$AWSIM_DIRECTORY/AWSIM.x86_64 "${opts[@]}"
