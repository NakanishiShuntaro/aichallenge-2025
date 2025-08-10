#!/bin/bash

# Help function to display usage
usage() {
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  screen      Capture screen via service call"
    echo "  control     Request control mode change"
    echo "  initial     Set initial pose"
    echo "  all         Execute all commands in sequence"
    echo "  help        Display this help message"
    exit 1
}

# Function to capture screen
capture_screen() {
    echo "Capturing screen..."
    timeout 10s ros2 service call /debug/service/capture_screen std_srvs/srv/Trigger >/dev/null
    if [ $? -eq 124 ]; then
        echo "Warning: Screen capture service call timed out after 10 seconds"
    else
        echo "Screen capture requested successfully"
    fi
}

# Function to request control mode
request_control() {
    echo "Requesting control mode change..."
    timeout 20s ros2 service call /control/control_mode_request autoware_auto_vehicle_msgs/srv/ControlModeCommand '{mode: 1}' >/dev/null
    if [ $? -eq 124 ]; then
        echo "Warning: Control mode request timed out after 20 seconds"
    else
        echo "Control mode change requested successfully"
    fi
}

wait_localization_ready() {
    echo "Waiting for localization to become ready..."
    local timeout_seconds=90
    local elapsed=0
    # Try to observe /localization/kinematic_state appearing and becoming readable
    while true; do
        # Try to get one sample of linear.x with relaxed QoS
        if timeout 5s bash -lc 'ros2 topic echo -n 1 --qos-reliability best_effort --qos-durability volatile /localization/kinematic_state --field twist.twist.linear.x 2>/dev/null | grep -Eq "^-?[0-9]"'; then
            echo "Localization topic is available"
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo "Waiting for /localization/kinematic_state... (${elapsed}s elapsed)"
        if [ $elapsed -ge $timeout_seconds ]; then
            echo "Warning: localization not ready after ${timeout_seconds}s. Continuing anyway..."
            break
        fi
    done
}

# Function to set initial pose
# Assignment 1 set correct initial pose
#            x: 89633.29,
#            y: 43127.57,
#            z: 0.8778,
#            w: 0.4788
set_initial_pose() {
    echo "Setting initial pose..."
    # QoS を明示（reliable / transient_local）し、取りこぼしを減らす
    # 推奨初期姿勢（Assignment 1）に合わせる
    # position: x=89633.29, y=43127.57 （zは0.0のまま）
    # orientation: z=0.8778, w=0.4788
    local payload='{
      header: {frame_id: "map"},
      pose: {pose: {position: {x: 89633.29, y: 43127.57, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.8778, w: 0.4788}}}
    }'

    local tries=0
    local max_tries=3
    local ok=0
    while [ $tries -lt $max_tries ]; do
        timeout 20s bash -lc "ros2 topic pub -1 \
          --qos-reliability reliable \
          --qos-durability transient_local \
          --qos-history keep_last \
          --qos-depth 1 \
          /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '${payload}'" >/dev/null
        rc=$?
        if [ $rc -eq 0 ]; then
            ok=1
            break
        fi
        tries=$((tries + 1))
        echo "Retry initial pose ($tries/$max_tries)"
        sleep 1
    done
    if [ $ok -eq 1 ]; then
        echo "Initial pose set successfully"
    else
        echo "Warning: Initial pose publication failed"
    fi
}

check_awsim() {
    timeout_seconds=60
    elapsed=0
    while ! timeout 10s ros2 topic echo /awsim/control_cmd 2>/dev/null | grep -q "sec:"; do
        sleep 0.5
        elapsed=$((elapsed + 10))
        echo "Waiting for /awsim/control_cmd topic to be available... (${elapsed}s elapsed)"

        if [ $elapsed -ge $timeout_seconds ]; then
            echo "Warning: /awsim/control_cmd topic not available after ${timeout_seconds}s timeout. Continuing anyway..."
            break
        fi
    done
    sleep 1
    echo "System is ready, executing publish commands..."
}

check_capture() {
    # Start recording rviz2
    echo "Check if screen capture is ready"
    timeout_seconds=60 # 1 minute timeout
    elapsed=0
    until (ros2 service type /debug/service/capture_screen >/dev/null); do
        sleep 5
        elapsed=$((elapsed + 5))
        echo "Screen capture is not ready (${elapsed}s elapsed)"

        if [ $elapsed -ge $timeout_seconds ]; then
            echo "Warning: Screen capture service not available after ${timeout_seconds}s timeout. Continuing anyway..."
            break
        fi
    done
}

# Check if an argument was provided
if [ $# -eq 0 ]; then
    usage
fi

# Process based on provided argument
case "$1" in
check)
    check_capture
    check_awsim
    ;;
screen)
    capture_screen
    ;;
control)
    request_control
    ;;
initial)
    set_initial_pose
    ;;
all)
    # Give system a bit more time to finish bring-up
    sleep 15
    # Publish initial pose twice for robustness
    set_initial_pose
    sleep 3
    set_initial_pose
    request_control
    # Wait for localization readiness (do not hard-fail on timeout)
    wait_localization_ready
    ;;
help)
    usage
    ;;
*)
    echo "Error: Invalid option '$1'"
    usage
    ;;
esac

exit 0
