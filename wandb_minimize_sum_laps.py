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


def _get_current_speed_mps(container_name: str) -> float | None:
    """コンテナ内で /localization/kinematic_state の合成速度を1回取得する。

    twist.twist.linear.{x,y} を取得し、sqrt(x^2 + y^2) を返す。取得失敗時は None。
    """
    try:
        cmd_x = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            "source /autoware/install/setup.bash >/dev/null 2>&1; ros2 topic echo -n 1 /localization/kinematic_state --field twist.twist.linear.x 2>/dev/null | tr -d '\r'",
        ]
        cmd_y = [
            "docker",
            "exec",
            container_name,
            "bash",
            "-lc",
            "source /autoware/install/setup.bash >/dev/null 2>&1; ros2 topic echo -n 1 /localization/kinematic_state --field twist.twist.linear.y 2>/dev/null | tr -d '\r'",
        ]
        res_x = subprocess.run(cmd_x, capture_output=True, text=True, timeout=5)
        res_y = subprocess.run(cmd_y, capture_output=True, text=True, timeout=5)
        if res_x.returncode != 0 or res_y.returncode != 0:
            return None
        sx = res_x.stdout.strip()
        sy = res_y.stdout.strip()
        if not sx or not sy:
            return None
        vx = float(sx)
        vy = float(sy)
        return math.hypot(vx, vy)
    except Exception:
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
        deterministic_name = f"aichallenge-2025-eval-{os.getpid()}"
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

        # 5. Save modified XML file to result folder
        result_xml_path = os.path.join(latest_folder, "reference.launch.xml")
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
        proc = subprocess.Popen(
            [
                "env",
                f"CONTAINER_NAME={deterministic_name}",
                "./docker_run.sh",
                "eval",
                "cpu",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def monitor_and_stop_on_slow_speed():
            # 合成速度（sqrt(vx^2 + vy^2)）が threshold 未満の状態が hold_seconds 継続したら停止
            threshold = float(os.environ.get("SPEED_STOP_THRESHOLD", "0.1"))
            hold_seconds = float(os.environ.get("SPEED_STOP_HOLD_SEC", "15"))
            consecutive = 0.0
            interval = 1.0
            while proc.poll() is None:
                speed = _get_current_speed_mps(deterministic_name)
                if speed is not None and abs(speed) < threshold:
                    consecutive += interval
                else:
                    consecutive = 0.0
                if consecutive >= hold_seconds:
                    subprocess.run(
                        ["docker", "stop", "-t", "5", deterministic_name],
                        capture_output=True,
                    )
                    break
                time.sleep(interval)

        monitor_thread = threading.Thread(
            target=monitor_and_stop_on_slow_speed, daemon=True
        )
        monitor_thread.start()

        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "stop", "-t", "5", deterministic_name], capture_output=True
            )
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
        result_xml_path = os.path.join(latest_folder, "reference.launch.xml")
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
