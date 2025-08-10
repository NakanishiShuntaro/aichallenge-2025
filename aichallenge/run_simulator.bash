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
    # ソフトウェアレンダリングを強制（UnityのGL周りでのSegfault緩和）
    export LIBGL_ALWAYS_SOFTWARE=1
    export GALLIUM_DRIVER=llvmpipe
    export MESA_GL_VERSION_OVERRIDE=3.3
    export XDG_RUNTIME_DIR=/tmp
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

# ログファイルディレクトリの作成（ログ未生成問題の回避）
mkdir -p "$(dirname "${LOGFILE}")" || true

# 安定化のためリトライ（初回GPUで失敗したらheadlessへフォールバック）
attempt=0
max_attempts=2
while :; do
    "${AWSIM_DIRECTORY}/AWSIM.x86_64" "${opts[@]}"
    rc=$?
    if [ $rc -eq 0 ]; then
        break
    fi
    echo "[ERROR] AWSIM exited with code $rc"
    # 最初の失敗時、GPU検出されている場合はheadlessに切替えて再試行
    if [ $attempt -eq 0 ] && command -v nvidia-smi &>/dev/null && [[ -e /dev/nvidia0 ]]; then
        echo "[INFO] Retrying with headless fallback options"
        opts=("-headless" "-batchmode" "-nographics" "-logfile" "${LOGFILE}")
        export LIBGL_ALWAYS_SOFTWARE=1
        export GALLIUM_DRIVER=llvmpipe
        export MESA_GL_VERSION_OVERRIDE=3.3
        export XDG_RUNTIME_DIR=/tmp
    fi
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "[INFO] Reached max attempts ($max_attempts). Exiting AWSIM launcher."
        break
    fi
    echo "[INFO] Retry #$attempt in 2s..."
    sleep 2
done
