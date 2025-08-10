#!/bin/bash

target="${1}"
device="${2}"
user_flag="--user"
case "${target}" in
"eval")
    # Bind mount host output to ensure results/logs are visible on host
    volume="$(pwd)/output:/output"
    # Run as root in eval to allow sudo/sysctl and writing under /output
    user_flag=""
    ;;
"dev")
    volume="output:/output aichallenge:/aichallenge remote:/remote vehicle:/vehicle"
    ;;
"rm")
    # clean up old <none> images
    docker image prune -f
    exit 1
    ;;
*)
    echo "invalid argument (use 'dev' or 'eval')"
    exit 1
    ;;
esac

if [ "${device}" = "cpu" ]; then
    opts=""
    echo "[INFO] Running in CPU mode (forced by argument)"
elif [ "${device}" = "gpu" ]; then
    opts="--nvidia"
    echo "[INFO] Running in GPU mode (forced by argument)"
elif command -v nvidia-smi &>/dev/null && [[ -e /dev/nvidia0 ]]; then
    opts="--nvidia"
    echo "[INFO] NVIDIA GPU detected → enabling --nvidia"
else
    opts=""
    echo "[INFO] No NVIDIA GPU detected → running on CPU"
fi

mkdir -p output

LOG_DIR="output/latest"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/docker_run.log"
echo "A rocker run log is stored at : file://$LOG_FILE"

# Allow external override of container name via CONTAINER_NAME env var
container_name_env=${CONTAINER_NAME}
if [ -z "$container_name_env" ]; then
  container_name_env="aichallenge-2025-$(date "+%Y-%m-%d-%H-%M-%S")"
fi

# Write out the chosen container name so external tools can reference it reliably
echo "$container_name_env" > "$LOG_DIR/container_name"

# shellcheck disable=SC2086
rocker ${opts} --x11 --devices /dev/dri ${user_flag} --net host --privileged --name "$container_name_env" --volume ${volume} -- "aichallenge-2025-${target}-${USER}" 2>&1 | tee "$LOG_FILE"
