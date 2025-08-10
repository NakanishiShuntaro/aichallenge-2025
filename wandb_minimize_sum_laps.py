#!/home/ubuntu/aichallenge-2025/.venv/bin/python
"""
W&B Sweep を使用して、Autoware のターン走行における 6 周ラップタイム合計の最小化を行うスクリプトです。
Docker を用いて評価実行し、各試行のメトリクスとアーティファクトを W&B に記録します。

使い方（W&B Sweep 推奨）:
- スイープ作成:  wandb sweep sweep.yaml
- エージェント:  wandb agent <SWEEP_ID>
  （sweep.yaml は本スクリプトを `--sweep` 付きで起動します）
"""

import argparse
import glob
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, cast

try:
    wandb = cast(Any, importlib.import_module("wandb"))
except Exception:

    class _WandbStub:  # pragma: no cover - fallback for type-checkers
        def init(self, *args, **kwargs):
            class _Run:
                def log_artifact(self, *a, **k):
                    pass

                def finish(self):
                    pass

            return _Run()

        def log(self, *args, **kwargs):
            pass

        class Settings:
            def __init__(self, *a, **k):
                pass

        class Artifact:
            def __init__(self, *a, **k):
                pass

    wandb = cast(Any, _WandbStub())


def spinner_animation(message, stop_event):
    """Display a spinning animation while subprocess is running"""
    spinner_chars = ["|", "/", "-", "\\"]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message} {spinner_chars[idx]}")
        sys.stdout.flush()
        idx = (idx + 1) % len(spinner_chars)
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 2) + "\r")
    sys.stdout.flush()


def run_subprocess_with_spinner(cmd, message, **kwargs):
    """Run subprocess with spinner animation"""
    stop_event = threading.Event()
    spinner_thread = threading.Thread(
        target=spinner_animation, args=(message, stop_event)
    )
    spinner_thread.start()

    try:
        result = subprocess.run(cmd, **kwargs)
        return result
    finally:
        stop_event.set()
        spinner_thread.join()


def backup_launch_file(launch_file_path, backup_dir):
    """Create a backup of the launch file"""
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    backup_path = os.path.join(backup_dir, "reference.launch.xml.backup")
    shutil.copy2(launch_file_path, backup_path)
    return backup_path


def restore_launch_file(launch_file_path, backup_path):
    """Restore the launch file from backup"""
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, launch_file_path)
        return True
    return False


def modify_xml_parameter(xml_file_path, parameter_updates):
    """Modify XML parameters directly

    Args:
        xml_file_path: Path to the XML file
        parameter_updates: Dict of parameter updates in format {tag_name: {attribute: value}}
    """
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Define the parameter mapping with their XPath-like locations
    param_mappings = {
        "twist_smoothing_steps": (
            "include",
            "ekf_localizer",
            "arg",
            "twist_smoothing_steps",
        ),
        "pose_smoothing_steps": (
            "include",
            "ekf_localizer",
            "arg",
            "pose_smoothing_steps",
        ),
        "proc_stddev_vx_c": ("include", "ekf_localizer", "arg", "proc_stddev_vx_c"),
        "proc_stddev_wz_c": ("include", "ekf_localizer", "arg", "proc_stddev_wz_c"),
        "accel_lowpass_gain": ("node", "twist2accel", "param", "accel_lowpass_gain"),
        "external_target_vel": (
            "node",
            "simple_pure_pursuit",
            "param",
            "external_target_vel",
        ),
        "lookahead_gain": ("node", "simple_pure_pursuit", "param", "lookahead_gain"),
        "lookahead_min_distance": (
            "node",
            "simple_pure_pursuit",
            "param",
            "lookahead_min_distance",
        ),
        "speed_proportional_gain": (
            "node",
            "simple_pure_pursuit",
            "param",
            "speed_proportional_gain",
        ),
        "pose_additional_delay_var": (
            "let",
            "pose_additional_delay_var",
            "value",
            "pose_additional_delay_var",
        ),
        "tf_rate": ("include", "ekf_localizer", "arg", "tf_rate"),
        "extend_state_step": ("include", "ekf_localizer", "arg", "extend_state_step"),
        "steering_tire_angle_gain_var": (
            "let",
            "steering_tire_angle_gain_var",
            "value",
            "steering_tire_angle_gain_var",
        ),
    }

    # Update each parameter
    for param_name, value in parameter_updates.items():
        if param_name in param_mappings:
            # Find and update the parameter
            if param_name in [
                "twist_smoothing_steps",
                "pose_smoothing_steps",
                "proc_stddev_vx_c",
                "proc_stddev_wz_c",
                "tf_rate",
                "extend_state_step",
            ]:
                # These are in ekf_localizer include
                for include in root.findall(".//include[@file]"):
                    if "ekf_localizer" in include.get("file", ""):
                        for arg in include.findall("arg"):
                            if arg.get("name") == param_name:
                                arg.set("value", str(value))
                                break
                        break
            elif param_name == "accel_lowpass_gain":
                # This is in twist2accel node
                for node in root.findall('.//node[@pkg="twist2accel"]'):
                    for param in node.findall("param"):
                        if param.get("name") == param_name:
                            param.set("value", str(value))
                            break
                    break
            elif param_name in [
                "pose_additional_delay_var",
                "steering_tire_angle_gain_var",
            ]:
                # These are <let> tags
                for let in root.findall('.//let[@name="{}"]'.format(param_name)):
                    let.set("value", str(value))
                    break
            else:
                # These are in simple_pure_pursuit node
                for node in root.findall('.//node[@pkg="simple_pure_pursuit"]'):
                    for param in node.findall("param"):
                        if param.get("name") == param_name:
                            param.set("value", str(value))
                            break
                    break

    # Write back to file
    tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)


def _parse_linear_xy_from_yaml(text: str) -> tuple[float, float] | None:
    """YAML出力から linear.x と linear.y を抽出するフォールバックパーサ。

    例:
      twist:
        twist:
          linear:
            x: 0.0
            y: 0.0
    """
    try:
        m = re.search(
            r"linear:\s*\n\s*x:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\n\s*y:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            text,
        )
        if not m:
            return None
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def _get_current_speed_mps(container_name: str) -> float | None:
    """コンテナ内で /localization/kinematic_state の合成速度を1回取得する。

    twist.twist.linear.{x,y} を取得し、sqrt(x^2 + y^2) を返す。取得失敗時は None。
    """
    debug = os.environ.get("SPEED_MONITOR_DEBUG", "1") == "1"

    try:
        # Prefer Autoware workspace setup if available, else fallback to system install
        source_cmd = (
            "if [ -f /aichallenge/workspace/install/setup.bash ]; then "
            "source /aichallenge/workspace/install/setup.bash; "
            "elif [ -f /autoware/install/setup.bash ]; then "
            "source /autoware/install/setup.bash; fi"
        )

        if debug:
            print(f"[speed] Getting speed data from container: {container_name}")

        # まずは --field でシンプルに取得（QoSを緩めて受信しやすくする）
        cmd_x = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            f"{source_cmd} >/dev/null 2>&1; ros2 topic echo -n 1 --qos-reliability best_effort --qos-durability volatile /localization/kinematic_state --field twist.twist.linear.x 2>/dev/null | tr -d '\r'",
        ]
        cmd_y = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            f"{source_cmd} >/dev/null 2>&1; ros2 topic echo -n 1 --qos-reliability best_effort --qos-durability volatile /localization/kinematic_state --field twist.twist.linear.y 2>/dev/null | tr -d '\r'",
        ]

        res_x = subprocess.run(cmd_x, capture_output=True, text=True, timeout=20)
        res_y = subprocess.run(cmd_y, capture_output=True, text=True, timeout=20)

        if debug:
            print(
                f"[speed] Command results: vx_rc={res_x.returncode}, vy_rc={res_y.returncode}"
            )
            if res_x.returncode != 0:
                print(f"[speed] vx command error: stderr='{res_x.stderr.strip()}'")
            if res_y.returncode != 0:
                print(f"[speed] vy command error: stderr='{res_y.stderr.strip()}'")

        if res_x.returncode == 0 and res_y.returncode == 0:
            sx = res_x.stdout.strip()
            sy = res_y.stdout.strip()

            if debug:
                print(f"[speed] Raw topic output: vx='{sx}', vy='{sy}'")

            if sx and sy:
                vx = float(sx)
                vy = float(sy)
                speed = math.hypot(vx, vy)

                if debug:
                    print(
                        f"[speed] Parsed values: vx={vx:.6f}, vy={vy:.6f}, combined={speed:.6f} m/s"
                    )

                return speed
            else:
                if debug:
                    print(f"[speed] Empty output from topic: vx='{sx}', vy='{sy}'")

        # 別トピックのフィールド取得を試す（車速系）
        try_alt = False
        if (res_x.returncode != 0 or not res_x.stdout.strip()) or (
            res_y.returncode != 0 or not res_y.stdout.strip()
        ):
            try_alt = True
        if try_alt:
            alt_topic = "/vehicle/status/twist"
            cmd2_x = [
                "docker",
                "exec",
                container_name,
                "bash",
                "-lc",
                f"{source_cmd} >/dev/null 2>&1; ros2 topic echo -n 1 --qos-reliability best_effort --qos-durability volatile {alt_topic} --field twist.linear.x 2>/dev/null | tr -d '\r'",
            ]
            cmd2_y = [
                "docker",
                "exec",
                container_name,
                "bash",
                "-lc",
                f"{source_cmd} >/dev/null 2>&1; ros2 topic echo -n 1 --qos-reliability best_effort --qos-durability volatile {alt_topic} --field twist.linear.y 2>/dev/null | tr -d '\r'",
            ]
            alt_x = subprocess.run(cmd2_x, capture_output=True, text=True, timeout=20)
            alt_y = subprocess.run(cmd2_y, capture_output=True, text=True, timeout=20)
            if debug:
                print(
                    f"[speed] Alt topic results: vx_rc={alt_x.returncode}, vy_rc={alt_y.returncode}"
                )
            if alt_x.returncode == 0 and alt_y.returncode == 0:
                sx = alt_x.stdout.strip()
                sy = alt_y.stdout.strip()
                if sx and sy:
                    try:
                        vx = float(sx)
                        vy = float(sy)
                        speed = math.hypot(vx, vy)
                        if debug:
                            print(
                                f"[speed] Alt parsed values: vx={vx:.6f}, vy={vy:.6f}, combined={speed:.6f} m/s"
                            )
                        return speed
                    except Exception:
                        pass

        # フィールド取得に失敗、または空出力だった場合は YAML 全体を1回取得してパース
        if debug:
            print("[speed] Falling back to YAML parse")
        cmd_yaml = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            f"{source_cmd} >/dev/null 2>&1; ros2 topic echo -n 1 --qos-reliability best_effort --qos-durability volatile /localization/kinematic_state 2>/dev/null | tr -d '\r'",
        ]
        res = subprocess.run(cmd_yaml, capture_output=True, text=True, timeout=20)
        if res.returncode != 0 or not res.stdout:
            if debug:
                err = res.stderr.strip() if res.stderr else ""
                print(f"[speed] YAML echo failed: rc={res.returncode}, stderr='{err}'")
            return None

        parsed = _parse_linear_xy_from_yaml(res.stdout)
        if not parsed:
            if debug:
                print("[speed] YAML parse returned no values")
            return None
        vx, vy = parsed
        speed = math.hypot(vx, vy)
        if debug:
            print(
                f"[speed] Parsed from YAML: vx={vx:.6f}, vy={vy:.6f}, combined={speed:.6f} m/s"
            )
        return speed

    except Exception as e:
        if debug:
            print(f"[speed] Exception getting speed: {e}")
        return None


def objective(trial):
    run = None
    wandb_entity = os.environ.get(
        "WANDB_ENTITY", "nakanishi-shuntaro-638-kyushu-university"
    )

    twist_smoothing_steps = trial.suggest_int("twist_smoothing_steps", 1, 5)
    pose_smoothing_steps = trial.suggest_int("pose_smoothing_steps", 1, 5)
    proc_stddev_vx_c = trial.suggest_float("proc_stddev_vx_c", 15.0, 25.0)
    proc_stddev_wz_c = trial.suggest_float("proc_stddev_wz_c", 0.5, 2.5)
    accel_lowpass_gain = trial.suggest_float("accel_lowpass_gain", 0.1, 1.0)
    external_target_vel = trial.suggest_float("external_target_vel", 5.0, 30.0)
    lookahead_gain = trial.suggest_float("lookahead_gain", 0.1, 1.0)
    lookahead_min_distance = trial.suggest_float("lookahead_min_distance", 1.0, 5.0)
    speed_proportional_gain = trial.suggest_float("speed_proportional_gain", 0.5, 3.0)
    pose_additional_delay_var = trial.suggest_float(
        "pose_additional_delay_var", 0.1, 1.0
    )
    tf_rate = trial.suggest_float("tf_rate", 10.0, 50.0)
    extend_state_step = trial.suggest_int("extend_state_step", 50, 200)
    steering_tire_angle_gain_var = trial.suggest_float(
        "steering_tire_angle_gain_var", 1.0, 1.7
    )

    run = wandb.init(
        project="aichallenge-2025",
        entity=wandb_entity,
        group="autoware-turning",
        job_type="optuna",
        name=f"trial-{trial.number}",
        config={
            "twist_smoothing_steps": twist_smoothing_steps,
            "pose_smoothing_steps": pose_smoothing_steps,
            "proc_stddev_vx_c": proc_stddev_vx_c,
            "proc_stddev_wz_c": proc_stddev_wz_c,
            "accel_lowpass_gain": accel_lowpass_gain,
            "external_target_vel": external_target_vel,
            "lookahead_gain": lookahead_gain,
            "lookahead_min_distance": lookahead_min_distance,
            "speed_proportional_gain": speed_proportional_gain,
            "pose_additional_delay_var": pose_additional_delay_var,
            "tf_rate": tf_rate,
            "extend_state_step": extend_state_step,
            "steering_tire_angle_gain_var": steering_tire_angle_gain_var,
        },
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
    )

    # Define paths
    launch_file_path = "./aichallenge/workspace/src/aichallenge_submit/aichallenge_submit_launch/launch/reference.launch.xml"

    try:
        # 1. Modify XML parameters directly
        parameter_updates = {
            "twist_smoothing_steps": twist_smoothing_steps,
            "pose_smoothing_steps": pose_smoothing_steps,
            "proc_stddev_vx_c": proc_stddev_vx_c,
            "proc_stddev_wz_c": proc_stddev_wz_c,
            "accel_lowpass_gain": accel_lowpass_gain,
            "external_target_vel": external_target_vel,
            "lookahead_gain": lookahead_gain,
            "lookahead_min_distance": lookahead_min_distance,
            "speed_proportional_gain": speed_proportional_gain,
            "pose_additional_delay_var": pose_additional_delay_var,
            "tf_rate": tf_rate,
            "extend_state_step": extend_state_step,
            "steering_tire_angle_gain_var": steering_tire_angle_gain_var,
        }

        modify_xml_parameter(launch_file_path, parameter_updates)
        print("XML parameters updated successfully")

        # 2. Docker build
        run_subprocess_with_spinner(
            ["./create_submit_file.bash"],
            "Creating submit file...",
            capture_output=True,
            text=True,
            timeout=60,
        )
        build_result = run_subprocess_with_spinner(
            ["./docker_build.sh", "eval"],
            "Building Docker image...",
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if build_result.returncode != 0:
            print(f"Docker build failed: {build_result.stderr}")
            return 225.0

        # 3. Docker run (without environment variables)
        # Assign deterministic container name so we can monitor/stop it by name
        # Use W&B run id if available, else fallback to timestamp
        # Include PID and epoch seconds to avoid name collisions across parallel runs
        deterministic_name = f"aichallenge-2025-eval-{os.getpid()}-{int(time.time())}"
        env_for_run = os.environ.copy()
        env_for_run["CONTAINER_NAME"] = deterministic_name

        run_result = run_subprocess_with_spinner(
            [
                "env",
                f"CONTAINER_NAME={deterministic_name}",
                "./docker_run.sh",
                "eval",
                "cpu",
            ],
            "Running Docker container...",
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if run_result.returncode != 0:
            print(f"Docker run failed: {run_result.stderr}")
            return 225.0

        # 4. Find latest result folder
        output_dir = "./output"
        if not os.path.exists(output_dir):
            print("Output directory not found")
            return 225.0

        # Get all timestamped folders
        result_folders = glob.glob(
            os.path.join(
                output_dir,
                "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]",
            )
        )
        if not result_folders:
            print("No result folders found")
            return 225.0

        # Find latest folder
        latest_folder = max(result_folders, key=os.path.getctime)
        result_file = os.path.join(latest_folder, "result-summary.json")

        # 5. Save modified XML file to result folder（書き込み不可の可能性に備えフォールバック）
        result_xml_path = os.path.join(latest_folder, "reference.launch.xml")
        try:
            # 可能ならフォルダに書き込み権限を付与
            try:
                os.chmod(latest_folder, 0o777)
            except Exception:
                pass
            shutil.copy2(launch_file_path, result_xml_path)
        except PermissionError:
            alt_dir = os.path.join("./output", "artifacts")
            os.makedirs(alt_dir, exist_ok=True)
            result_xml_path = os.path.join(
                alt_dir, f"reference.{int(time.time())}.launch.xml"
            )
            shutil.copy2(launch_file_path, result_xml_path)
        print(f"Modified XML saved to: {result_xml_path}")

        print(f"Reading results from: {result_file}")

        # 6. Read and calculate score
        if not os.path.exists(result_file):
            print("Result file not found")
            return 225.0

        with open(result_file, "r") as f:
            result = json.load(f)
            print(f"Result data: {result}")

            if "laps" not in result or not result["laps"]:
                print("No laps data found")
                return 225.0

            # Calculate sum of lap times (must be exactly 6 laps)
            lap_times = result["laps"]
            if len(lap_times) != 6:
                print(f"Must complete exactly 6 laps, got {len(lap_times)} laps")
                return 225.0

            total_time = sum(lap_times)
            print(f"Total lap time for 6 laps: {total_time}")
            print("=" * 50)
            metrics = {f"lap_{i + 1}": t for i, t in enumerate(lap_times)}
            metrics.update({"total_time": total_time})
            try:
                wandb.log(metrics)
            except Exception as e:
                print(f"Warning: failed to log metrics to W&B: {e}")

            try:
                artifact = wandb.Artifact(
                    name=f"trial-{trial.number}-results", type="evaluation"
                )
                artifact.add_file(result_file, name="result-summary.json")
                artifact.add_file(result_xml_path, name="reference.launch.xml")
                run.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: failed to log artifacts to W&B: {e}")
            return total_time

    except subprocess.TimeoutExpired:
        print("Process timed out")
        try:
            wandb.log({"total_time": 225.0, "status": "timeout"})
        except Exception:
            pass
        return 225.0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        try:
            wandb.log({"total_time": 225.0, "status": "error"})
        except Exception:
            pass
        return 225.0
    finally:
        # Skip per-trial restoration - will be handled at script end
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass


def sweep_run() -> None:
    """W&B Sweep から呼び出す 1 トライアル分の実行。

    wandb.config に与えられたハイパーパラメータで評価を実行し、
    メトリクスとアーティファクトをログして終了します。
    """
    wandb_entity = os.environ.get(
        "WANDB_ENTITY", "nakanishi-shuntaro-638-kyushu-university"
    )
    run = wandb.init(
        project="aichallenge-2025",
        entity=wandb_entity,
        group="autoware-turning",
        job_type="sweep",
        settings=wandb.Settings(start_method="thread"),
    )

    cfg = wandb.config
    # 速度停止判定の既定値をプログラム側で設定（環境変数未設定時のみ）
    stop_threshold = getattr(cfg, "stop_threshold", 0.1)
    stop_hold_sec = getattr(cfg, "stop_hold_sec", 15)
    os.environ.setdefault("SPEED_STOP_THRESHOLD", str(stop_threshold))
    os.environ.setdefault("SPEED_STOP_HOLD_SEC", str(stop_hold_sec))
    # 期待パラメータが欠けている場合のフォールバック値（スイープ側で全て指定する想定）
    twist_smoothing_steps = int(getattr(cfg, "twist_smoothing_steps", 3))
    pose_smoothing_steps = int(getattr(cfg, "pose_smoothing_steps", 3))
    proc_stddev_vx_c = float(getattr(cfg, "proc_stddev_vx_c", 20.0))
    proc_stddev_wz_c = float(getattr(cfg, "proc_stddev_wz_c", 1.5))
    accel_lowpass_gain = float(getattr(cfg, "accel_lowpass_gain", 0.5))
    external_target_vel = float(getattr(cfg, "external_target_vel", 20.0))
    lookahead_gain = float(getattr(cfg, "lookahead_gain", 0.5))
    lookahead_min_distance = float(getattr(cfg, "lookahead_min_distance", 3.0))
    speed_proportional_gain = float(getattr(cfg, "speed_proportional_gain", 1.0))
    pose_additional_delay_var = float(getattr(cfg, "pose_additional_delay_var", 0.5))
    tf_rate = float(getattr(cfg, "tf_rate", 30.0))
    extend_state_step = int(getattr(cfg, "extend_state_step", 100))
    steering_tire_angle_gain_var = float(
        getattr(cfg, "steering_tire_angle_gain_var", 1.3)
    )

    launch_file_path = "./aichallenge/workspace/src/aichallenge_submit/aichallenge_submit_launch/launch/reference.launch.xml"
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    backup_path = backup_launch_file(launch_file_path, output_dir)
    print(f"Initial backup created at: {backup_path}")

    try:
        parameter_updates = {
            "twist_smoothing_steps": twist_smoothing_steps,
            "pose_smoothing_steps": pose_smoothing_steps,
            "proc_stddev_vx_c": proc_stddev_vx_c,
            "proc_stddev_wz_c": proc_stddev_wz_c,
            "accel_lowpass_gain": accel_lowpass_gain,
            "external_target_vel": external_target_vel,
            "lookahead_gain": lookahead_gain,
            "lookahead_min_distance": lookahead_min_distance,
            "speed_proportional_gain": speed_proportional_gain,
            "pose_additional_delay_var": pose_additional_delay_var,
            "tf_rate": tf_rate,
            "extend_state_step": extend_state_step,
            "steering_tire_angle_gain_var": steering_tire_angle_gain_var,
        }
        modify_xml_parameter(launch_file_path, parameter_updates)
        print("XML parameters updated successfully (sweep)")

        run_subprocess_with_spinner(
            ["./create_submit_file.bash"],
            "Creating submit file...",
            capture_output=True,
            text=True,
            timeout=60,
        )
        build_result = run_subprocess_with_spinner(
            ["./docker_build.sh", "eval"],
            "Building Docker image...",
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if build_result.returncode != 0:
            print(f"Docker build failed: {build_result.stderr}")
            wandb.log({"total_time": 225.0, "status": "build_failed"})
            return

        deterministic_name = f"aichallenge-2025-eval-{os.getpid()}"
        # Run container in background and monitor speed
        # Note: Do not capture stdout/stderr here; let docker_run.sh tee to file to avoid pipe backpressure
        proc = subprocess.Popen(
            [
                "env",
                f"CONTAINER_NAME={deterministic_name}",
                "./docker_run.sh",
                "eval",
                "cpu",
            ],
            stdout=None,
            stderr=None,
        )

        # Give the container and monitor more time to initialize
        time.sleep(10)

        def _resolve_container_name(expected_name: str) -> str | None:
            """Resolve actual running container name.

            Order changed to reduce race with shared container_name file:
            1) Try exact expected_name via `docker ps`
            2) Try output/latest/container_name written by docker_run.sh
            3) Fallback: pick running container that starts with 'aichallenge-2025-'
            """
            try:
                out = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
                    if expected_name in names:
                        return expected_name
            except Exception:
                pass
            # Try file (may be overwritten by other runs)
            try:
                latest_path = os.path.join("output", "latest", "container_name")
                if os.path.exists(latest_path):
                    with open(latest_path) as f:
                        name = f.read().strip()
                        if name:
                            return name
            except Exception:
                pass
            # Fallback prefix search
            try:
                out = subprocess.run(
                    ["docker", "ps", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if out.returncode == 0:
                    names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
                    for n in names:
                        if n.startswith("aichallenge-2025-"):
                            return n
            except Exception:
                pass
            return None

        def monitor_and_stop_on_slow_speed():
            print("[monitor] Monitor thread started, initializing...")
            # 合成速度（sqrt(vx^2 + vy^2)）が threshold 未満の状態が hold_seconds 継続したら停止
            threshold = float(os.environ.get("SPEED_STOP_THRESHOLD", "0.1"))
            hold_seconds = float(os.environ.get("SPEED_STOP_HOLD_SEC", "15"))
            consecutive = 0.0
            interval = 1.0
            resolved = None
            debug = os.environ.get("SPEED_MONITOR_DEBUG", "1") == "1"
            had_valid = False
            last_speed: float | None = None
            miss_cnt = 0
            loop_count = 0
            print(
                f"[monitor] Configured: threshold={threshold}, hold_seconds={hold_seconds}, debug={debug}"
            )
            print(
                f"[monitor] Environment variables: SPEED_STOP_THRESHOLD={os.environ.get('SPEED_STOP_THRESHOLD', 'not_set')}, SPEED_STOP_HOLD_SEC={os.environ.get('SPEED_STOP_HOLD_SEC', 'not_set')}"
            )

            # Warm-up: wait until topic appears to reduce None readings
            # Try for up to 60s before starting low-speed checks
            warmup_deadline = time.time() + 60.0
            while proc.poll() is None and time.time() < warmup_deadline:
                resolved = _resolve_container_name(deterministic_name)
                if resolved:
                    check_cmd = [
                        "docker",
                        "exec",
                        resolved,
                        "bash",
                        "-lc",
                        "if [ -f /aichallenge/workspace/install/setup.bash ]; then source /aichallenge/workspace/install/setup.bash; elif [ -f /autoware/install/setup.bash ]; then source /autoware/install/setup.bash; fi >/dev/null 2>&1; ros2 topic list | grep -q '^/localization/kinematic_state$'",
                    ]
                    ok = subprocess.run(check_cmd).returncode == 0
                    if ok:
                        print(
                            "[monitor] Topic /localization/kinematic_state is available; starting speed monitor"
                        )
                        break
                time.sleep(1.0)

            while proc.poll() is None:
                loop_count += 1
                if debug and loop_count % 10 == 0:  # Every 10 seconds
                    print(
                        f"[monitor] Loop #{loop_count}: Still monitoring, proc status: {proc.poll()}"
                    )

                if resolved is None:
                    resolved = _resolve_container_name(deterministic_name)
                    if resolved is None:
                        if debug and loop_count % 5 == 0:  # Every 5 seconds
                            print(
                                f"[monitor] Waiting for container (expected: {deterministic_name})"
                            )
                        time.sleep(interval)
                        continue
                    else:
                        print(f"[monitor] Container resolved: {resolved}")

                speed = _get_current_speed_mps(resolved)
                if speed is not None:
                    had_valid = True
                    last_speed = speed
                    miss_cnt = 0
                    if abs(speed) < threshold:
                        consecutive += interval
                        if debug:
                            print(
                                f"[monitor] LOW SPEED: {speed:.4f} m/s < {threshold} (consecutive: {consecutive:.1f}s/{hold_seconds}s)"
                            )
                    else:
                        if (
                            consecutive > 0
                        ):  # Only log when resetting from a low speed period
                            print(
                                f"[monitor] SPEED OK: {speed:.4f} m/s >= {threshold} (reset consecutive timer)"
                            )
                        consecutive = 0.0
                else:
                    # 取得失敗(None)が続く場合のフォールバック：
                    # 一度でも有効値を取得済みかつ直近が低速域なら、ミス2回目以降は低速継続としてカウント
                    miss_cnt += 1
                    if debug:
                        print(
                            f"[monitor] Speed data miss #{miss_cnt} (last_speed: {last_speed})"
                        )
                    if (
                        had_valid
                        and last_speed is not None
                        and abs(last_speed) < threshold
                        and miss_cnt >= 2
                    ):
                        consecutive += interval
                        if debug:
                            print(
                                f"[monitor] Using fallback: assuming low speed continues (consecutive: {consecutive:.1f}s/{hold_seconds}s)"
                            )
                    else:
                        consecutive = 0.0

                if debug:
                    try:
                        status = f"container={resolved} speed={speed} last={last_speed} consec={consecutive:.1f}s thresh={threshold} hold={hold_seconds}s loop={loop_count}"
                        if (
                            loop_count % 30 == 0 or consecutive > 0
                        ):  # Every 30s or when low speed
                            print(f"[monitor] Status: {status}")
                    except Exception:
                        pass

                if consecutive >= hold_seconds:
                    print(
                        f"[monitor] *** STOP CONDITION MET: {consecutive:.1f}s >= {hold_seconds}s ***"
                    )
                    target = resolved or deterministic_name
                    print(f"[monitor] Attempting to stop container: {target}")

                    stop_res = subprocess.run(
                        ["docker", "stop", "-t", "5", target],
                        capture_output=True,
                        text=True,
                    )
                    print(
                        f"[monitor] docker stop result: rc={stop_res.returncode}, out='{stop_res.stdout.strip()}', err='{stop_res.stderr.strip() if stop_res.stderr else ''}'"
                    )

                    # まだ動いているようなら kill を試す
                    try:
                        ps = subprocess.run(
                            ["docker", "ps", "--format", "{{.Names}}"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if ps.returncode == 0 and target in [
                            n.strip() for n in ps.stdout.splitlines()
                        ]:
                            print(
                                f"[monitor] Container {target} still running, attempting kill..."
                            )
                            kill_res = subprocess.run(
                                ["docker", "kill", target],
                                capture_output=True,
                                text=True,
                            )
                            print(
                                f"[monitor] docker kill result: rc={kill_res.returncode}, out='{kill_res.stdout.strip()}', err='{kill_res.stderr.strip() if kill_res.stderr else ''}'"
                            )
                        else:
                            print(f"[monitor] Container {target} successfully stopped")
                    except Exception as e:
                        print(f"[monitor] Exception during kill attempt: {e}")

                    print("[monitor] Stop sequence completed, exiting monitor loop")
                    break
                time.sleep(interval)

        # Start monitoring thread immediately after starting container
        if os.getenv("ENABLE_AUTO_STOP", "1") == "1":
            print("[monitor] Starting monitoring thread...")
            monitor_thread = threading.Thread(
                target=monitor_and_stop_on_slow_speed, daemon=True
            )
            monitor_thread.start()
            print("[monitor] Monitor thread started")
        else:
            print("[monitor] Auto-stop disabled by ENABLE_AUTO_STOP=0")

        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            # If timed out, resolve name and stop
            target = _resolve_container_name(deterministic_name) or deterministic_name
            subprocess.run(["docker", "stop", "-t", "5", target], capture_output=True)
            proc.wait()

        if proc.returncode != 0:
            print("Docker run failed or stopped")
            wandb.log({"total_time": 225.0, "status": "run_failed_or_stopped"})
            return

        result_folders = glob.glob(
            os.path.join(
                output_dir,
                "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]",
            )
        )
        if not result_folders:
            print("No result folders found")
            wandb.log({"total_time": 225.0, "status": "no_result_folders"})
            return

        latest_folder = max(result_folders, key=os.path.getctime)
        result_file = os.path.join(latest_folder, "result-summary.json")
        result_xml_try = os.path.join(latest_folder, "reference.launch.xml")
        try:
            try:
                os.chmod(latest_folder, 0o777)
            except Exception:
                pass
            shutil.copy2(launch_file_path, result_xml_try)
            result_xml_path = result_xml_try
        except PermissionError:
            alt_dir = os.path.join("./output", "artifacts")
            os.makedirs(alt_dir, exist_ok=True)
            result_xml_path = os.path.join(
                alt_dir, f"reference.{int(time.time())}.launch.xml"
            )
            shutil.copy2(launch_file_path, result_xml_path)
        print(f"Modified XML saved to: {result_xml_path}")

        if not os.path.exists(result_file):
            print("Result file not found")
            wandb.log({"total_time": 225.0, "status": "no_result_file"})
            return

        with open(result_file, "r") as f:
            result = json.load(f)
            print(f"Result data: {result}")

            if "laps" not in result or not result["laps"]:
                print("No laps data found")
                wandb.log({"total_time": 225.0, "status": "no_laps"})
                return

            lap_times = result["laps"]
            if len(lap_times) != 6:
                print(f"Must complete exactly 6 laps, got {len(lap_times)} laps")
                wandb.log({"total_time": 225.0, "status": "not_6_laps"})
                return

            total_time = sum(lap_times)
            print(f"Total lap time for 6 laps: {total_time}")
            print("=" * 50)

            metrics = {f"lap_{i + 1}": t for i, t in enumerate(lap_times)}
            metrics.update({"total_time": total_time})
            try:
                wandb.log(metrics)
            except Exception as e:
                print(f"Warning: failed to log metrics to W&B: {e}")

            try:
                artifact = wandb.Artifact(
                    name=f"sweep-run-{run.id}-results", type="evaluation"
                )
                artifact.add_file(result_file, name="result-summary.json")
                artifact.add_file(result_xml_path, name="reference.launch.xml")
                run.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: failed to log artifacts to W&B: {e}")

    except subprocess.TimeoutExpired:
        print("Process timed out")
        try:
            wandb.log({"total_time": 225.0, "status": "timeout"})
        except Exception:
            pass
    except Exception as e:
        print(f"Error during sweep run: {e}")
        try:
            wandb.log({"total_time": 225.0, "status": "error"})
        except Exception:
            pass
    finally:
        try:
            run.finish()
        except Exception:
            pass
        # Restore backup file at run end
        try:
            if os.path.exists(backup_path):
                if restore_launch_file(launch_file_path, backup_path):
                    print(f"Launch file restored from backup: {backup_path}")
                    os.remove(backup_path)
                    print("Backup file cleaned up")
        except Exception as e:
            print(f"Warning: failed to restore launch file: {e}")


def main():
    # Sweep 実行のエントリに一本化
    sweep_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run as W&B sweep program")
    parser.add_argument(
        "--single", action="store_true", help="Run once with defaults for debugging"
    )
    args = parser.parse_args()
    if args.sweep:
        sweep_run()
    elif args.single:
        # デフォルト値で 1 回だけ実行（スイープ外のデバッグ用）
        default_params = {
            "twist_smoothing_steps": 3,
            "pose_smoothing_steps": 3,
            "proc_stddev_vx_c": 20.0,
            "proc_stddev_wz_c": 1.5,
            "accel_lowpass_gain": 0.5,
            "external_target_vel": 20.0,
            "lookahead_gain": 0.5,
            "lookahead_min_distance": 3.0,
            "speed_proportional_gain": 1.0,
            "pose_additional_delay_var": 0.5,
            "tf_rate": 30.0,
            "extend_state_step": 100,
            "steering_tire_angle_gain_var": 1.3,
        }
        # 実行は sweep_run と同じ流れのため、簡易に環境変数として渡すより直呼びが明快
        # ここではパラメータ適用・実行ロジックを重複させず、sweep_run のコードパスを再利用する方針でもよい
        # 現状は簡潔性を優先し、sweep_run を利用してください
        sweep_run()
    else:
        main()

#    docker exec <container> bash -lc 'if [ -f /aichallenge/workspace/install/setup.bash ]; then source /aichallenge/workspace/install/setup.bash; elif [ -f /autoware/install/setup.bash ]; then source /autoware/install/setup.bash; fi; ros2 topic list | grep -E "(kinematic_state|/vehicle/status/twist|/localization/twist)"'
#    docker exec <container> bash -lc 'if [ -f /aichallenge/workspace/install/setup.bash ]; then source /aichallenge/workspace/install/setup.bash; elif [ -f /autoware/install/setup.bash ]; then source /autoware/install/setup.bash; fi; ros2 topic echo -n 1 /vehicle/status/twist --field twist.linear.x'
""" [INFO] [launch]: All log files can be found below /root/.ros/log/2025-08-10-19-02-11-182242-ubuntu-HP-Laptop-14s-dq5xxx-175
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [launch.user]: The arguments for aichallenge_system_launch.
[INFO] [launch.user]:  - simulation: true
[INFO] [launch.user]:  - use_sim_time: true
[INFO] [launch.user]:  - sensor_model: racing_kart_sensor_kit
[INFO] [launch.user]:  - launch_vehicle_interface: true
[INFO] [launch.user]:  - rviz config: /aichallenge/workspace/install/aichallenge_system_launch/share/aichallenge_system_launch/config/autoware.rviz
[INFO] [launch.user]: The arguments for aichallenge_submit_launch.
[INFO] [launch.user]:  - simulation: true
[INFO] [launch.user]:  - use_sim_time: true
[INFO] [launch.user]:  - sensor_model: racing_kart_sensor_kit
[INFO] [launch.user]:  - launch_vehicle_interface: true
[INFO] [launch.user]: echo launch param use_sim_time: true launch_vehicle_interface: true sensor_model: racing_kart_sensor_kit
[INFO] [robot_state_publisher-1]: process started with pid [199]
[INFO] [vehicle_velocity_converter-2]: process started with pid [201]
[INFO] [imu_corrector-3]: process started with pid [203]
[INFO] [gnss_poser-4]: process started with pid [205]
[INFO] [gyro_odometer-5]: process started with pid [207]
[INFO] [imu_gnss_poser_node-6]: process started with pid [209]
[INFO] [ekf_localizer-7]: process started with pid [211]
[INFO] [twist2accel-8]: process started with pid [213]
[INFO] [simple_trajectory_generator_node-9]: process started with pid [215]
[INFO] [interactive_server-10]: process started with pid [217]
[INFO] [simple_pure_pursuit-11]: process started with pid [219]
[INFO] [component_container-12]: process started with pid [221]
[INFO] [raw_vehicle_cmd_converter_node-13]: process started with pid [223]
[INFO] [goal_pose_setter_node-14]: process started with pid [225]
[INFO] [rviz2-15]: process started with pid [227]
[INFO] [relay-16]: process started with pid [229]
[INFO] [control_mode_adapter.py-17]: process started with pid [235]
[INFO] [actuation_cmd_converter-18]: process started with pid [258]
[INFO] [relay-19]: process started with pid [277]
[INFO] [component_container_mt-20]: process started with pid [286]
[simple_trajectory_generator_node-9] [INFO] [1754820133.344045245] [planning.scenario_planning.simple_trajectory_generator]: csv_path parameter changed from '' to '/aichallenge/workspace/install/simple_trajectory_generator/share/simple_trajectory_generator/data/raceline_awsim_15km.csv'
[simple_trajectory_generator_node-9] [INFO] [1754820133.346767446] [planning.scenario_planning.simple_trajectory_generator]: Successfully loaded new trajectory from CSV: /aichallenge/workspace/install/simple_trajectory_generator/share/simple_trajectory_generator/data/raceline_awsim_15km.csv with 121 points
[interactive_server-10] [INFO] [1754820133.347395732] [editor_tool_server]: Waiting for 5.000000 seconds before loading CSV and publishing markers.
[simple_trajectory_generator_node-9] [INFO] [1754820133.348468347] [planning.scenario_planning.simple_trajectory_generator]: z parameter changed to 6.500000
[simple_trajectory_generator_node-9] [INFO] [1754820133.351110413] [planning.scenario_planning.simple_trajectory_generator]: Loaded trajectory from CSV with 121 points
[simple_pure_pursuit-11] [INFO] [1754820133.363222682] [simple_pure_pursuit_node]: odometry is not available
[imu_corrector-3] [WARN] [1754820133.366447328] [sensing.imu.imu_corrector]: failed to get transform from imu_link to base_link: "imu_link" passed to lookupTransform argument target_frame does not exist. 
[imu_corrector-3] [ERROR] [1754820133.366613182] [sensing.imu.imu_corrector]: Please publish TF base_link to imu_link
[ekf_localizer-7] [WARN] [1754820133.382252920] [localization.ekf_localizer]: The node is not activated. Provide initial pose to pose_initializer
[component_container-12] [INFO] [1754820133.404280374] [map.map_container]: Load Library: /autoware/install/map_loader/lib/liblanelet2_map_loader_node.so
[imu_corrector-3] [WARN] [1754820133.406782417] [sensing.imu.imu_corrector]: failed to get transform from imu_link to base_link: "imu_link" passed to lookupTransform argument target_frame does not exist. 
[robot_state_publisher-1] [INFO] [1754820133.420518119] [robot_state_publisher]: got segment base_link
[robot_state_publisher-1] [INFO] [1754820133.420749503] [robot_state_publisher]: got segment gnss_link
[robot_state_publisher-1] [INFO] [1754820133.420762664] [robot_state_publisher]: got segment imu_link
[robot_state_publisher-1] [INFO] [1754820133.420771017] [robot_state_publisher]: got segment sensor_kit_base_link
[component_container-12] [INFO] [1754820133.445115212] [map.map_container]: Found class: rclcpp_components::NodeFactoryTemplate<Lanelet2MapLoaderNode>
[component_container-12] [INFO] [1754820133.445188912] [map.map_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<Lanelet2MapLoaderNode>
[gyro_odometer-5] [WARN] [1754820133.453985524] [localization.gyro_odometer]: Twist msg is not subscribed
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/map/lanelet2_map_loader' in container '/map/map_container'
[component_container-12] [INFO] [1754820133.470529507] [map.map_container]: Load Library: /autoware/install/map_loader/lib/liblanelet2_map_visualization_node.so
[component_container-12] [INFO] [1754820133.473316316] [map.map_container]: Found class: rclcpp_components::NodeFactoryTemplate<Lanelet2MapVisualizationNode>
[component_container-12] [INFO] [1754820133.473376470] [map.map_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<Lanelet2MapVisualizationNode>
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/map/lanelet2_map_visualization' in container '/map/map_container'
[component_container-12] [INFO] [1754820133.487708377] [map.lanelet2_map_visualization]: Map is loaded
[component_container-12] 
[component_container-12] [INFO] [1754820133.492155828] [map.map_container]: Load Library: /autoware/install/map_tf_generator/lib/libvector_map_tf_generator_node.so
[component_container-12] [INFO] [1754820133.498956211] [map.map_container]: Found class: rclcpp_components::NodeFactoryTemplate<VectorMapTFGeneratorNode>
[component_container-12] [INFO] [1754820133.499133161] [map.map_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<VectorMapTFGeneratorNode>
[component_container_mt-20] [INFO] [1754820133.506029656] [system.system_monitor.system_monitor.system_monitor_container]: Load Library: /autoware/install/system_monitor/lib/libcpu_monitor_lib.so
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/map/vector_map_tf_generator' in container '/map/map_container'
[component_container-12] [INFO] [1754820133.518478637] [map.vector_map_tf_generator]: broadcast static tf. map_frame:map, viewer_frame:viewer, x:89648.2, y:43157.2, z:6.5
[component_container_mt-20] [INFO] [1754820133.519841728] [system.system_monitor.system_monitor.system_monitor_container]: Found class: rclcpp_components::NodeFactoryTemplate<CPUMonitor>
[component_container_mt-20] [INFO] [1754820133.519897455] [system.system_monitor.system_monitor.system_monitor_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<CPUMonitor>
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/system/system_monitor/cpu_monitor' in container '/system/system_monitor/system_monitor/system_monitor_container'
[component_container_mt-20] [INFO] [1754820133.555299258] [system.system_monitor.system_monitor.system_monitor_container]: Load Library: /autoware/install/system_monitor/lib/libmem_monitor_lib.so
[component_container_mt-20] [INFO] [1754820133.559603805] [system.system_monitor.system_monitor.system_monitor_container]: Found class: rclcpp_components::NodeFactoryTemplate<MemMonitor>
[component_container_mt-20] [INFO] [1754820133.559666379] [system.system_monitor.system_monitor.system_monitor_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<MemMonitor>
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/system/system_monitor/mem_monitor' in container '/system/system_monitor/system_monitor/system_monitor_container'
[component_container_mt-20] [INFO] [1754820133.575959618] [system.system_monitor.system_monitor.system_monitor_container]: Load Library: /autoware/install/system_monitor/lib/libprocess_monitor_lib.so
[component_container_mt-20] [INFO] [1754820133.577910778] [system.system_monitor.system_monitor.system_monitor_container]: Found class: rclcpp_components::NodeFactoryTemplate<ProcessMonitor>
[component_container_mt-20] [INFO] [1754820133.577986570] [system.system_monitor.system_monitor.system_monitor_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<ProcessMonitor>
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/system/system_monitor/process_monitor' in container '/system/system_monitor/system_monitor/system_monitor_container'
[component_container_mt-20] [INFO] [1754820133.592292009] [system.system_monitor.system_monitor.system_monitor_container]: Load Library: /autoware/install/system_monitor/lib/libvoltage_monitor_lib.so
[component_container_mt-20] [INFO] [1754820133.594036764] [system.system_monitor.system_monitor.system_monitor_container]: Found class: rclcpp_components::NodeFactoryTemplate<VoltageMonitor>
[component_container_mt-20] [INFO] [1754820133.595151695] [system.system_monitor.system_monitor.system_monitor_container]: Instantiate class: rclcpp_components::NodeFactoryTemplate<VoltageMonitor>
[INFO] [launch_ros.actions.load_composable_nodes]: Loaded node '/system/system_monitor/voltage_monitor' in container '/system/system_monitor/system_monitor/system_monitor_container'
[rviz2-15] QPainter::begin: Paint device returned engine == 0, type: 2
[rviz2-15] QPainter::setPen: Painter not active
[rviz2-15] [INFO] [1754820134.039236879] [rviz2]: Stereo is NOT SUPPORTED
[rviz2-15] [INFO] [1754820134.039799278] [rviz2]: OpenGl version: 4.6 (GLSL 4.6)
[rviz2-15] [INFO] [1754820134.142230478] [rviz2]: Stereo is NOT SUPPORTED
[simple_pure_pursuit-11] [INFO] [1754820134.364515700] [simple_pure_pursuit_node]: odometry is not available
[simple_trajectory_generator_node-9] [WARN] [1754820135.215640668] [planning.scenario_planning.simple_trajectory_generator]: New subscription discovered on topic '/planning/scenario_planning/trajectory', requesting incompatible QoS. No messages will be sent to it. Last incompatible policy: RELIABILITY_QOS_POLICY
[simple_trajectory_generator_node-9] [WARN] [1754820135.217840933] [planning.scenario_planning.simple_trajectory_generator]: New subscription discovered on topic '/planning/scenario_planning/trajectory', requesting incompatible QoS. No messages will be sent to it. Last incompatible policy: RELIABILITY_QOS_POLICY
[simple_pure_pursuit-11] [INFO] [1754820135.374264221] [simple_pure_pursuit_node]: odometry is not available
[ekf_localizer-7] [WARN] [1754820135.467989211] [localization.ekf_localizer]: The node is not activated. Provide initial pose to pose_initializer
[simple_pure_pursuit-11] [INFO] [1754820136.384656077] [simple_pure_pursuit_node]: odometry is not available
[rviz2-15] QObject::connect: No such signal QLineEdit::valueChanged(std::string)
[goal_pose_setter_node-14] [INFO] [1754820136.696571421] [goal_pose_setter]: Call localization trigger
[goal_pose_setter_node-14] [INFO] [1754820136.749926692] [goal_pose_setter]: Complete localization trigger
[interactive_server-10] [INFO] [1754820138.347662118] [editor_tool_server]: Loading CSV file: /aichallenge/workspace/src/aichallenge-trajectory-editor/csv/centerline_15km.csv
[interactive_server-10] [ERROR] [1754820138.348002428] [editor_tool_server]: Failed to open file: /aichallenge/workspace/src/aichallenge-trajectory-editor/csv/centerline_15km.csv
[relay-16] [WARN] [1754820154.977082939] [initial_pose_relay]: Some, but not all, publishers on topic /initialpose offer 'transient local' durability. Falling back to 'volatile' durability in orderto connect to all publishers.
[rviz2-15] [WARN] [1754820155.111691698] [rviz2]: Negative eigenvalue found for position. Is the covariance matrix correct (positive semidefinite)?
[ekf_localizer-7] [WARN] [1754820155.125562426] [localization.ekf_localizer]: The Mahalanobis distance 11962652488.8070 is over the limit 10000.0000.
[rviz2-15] OpenCV: FFMPEG: tag 0x34363268/'h264' is not supported with codec id 27 and format 'mp4 / MP4 (MPEG-4 Part 14)'
[rviz2-15] OpenCV: FFMPEG: fallback to use tag 0x31637661/'avc1'
[simple_trajectory_generator_node-9] [INFO] [1754820189.351788871] [planning.scenario_planning.simple_trajectory_generator]: Published trajectory with 121 points
[simple_trajectory_generator_node-9] [INFO] [1754820250.351888699] [planning.scenario_planning.simple_trajectory_generator]: Published trajectory with 121 points
[simple_trajectory_generator_node-9] [INFO] [1754820311.351903135] [planning.scenario_planning.simple_trajectory_generator]: Published trajectory with 121 points
[simple_trajectory_generator_node-9] [INFO] [1754820372.351530967] [planning.scenario_planning.simple_trajectory_generator]: Published trajectory with 121 points
[simple_trajectory_generator_node-9] [INFO] [1754820433.351547600] [planning.scenario_planning.simple_trajectory_generator]: Published trajectory with 121 points
[simple_trajectory_generator_node-9] [INFO] [1754820494.351877020] [planning.scenario_planning.simple_trajectory_generator]: Published trajectory with 121 points
"""
