"""
Optunaを使用して、Autowareのターン走行における6周のラップタイムの合計を最小化するスクリプトです。
このスクリプトは、Dockerを使用してAutowareの評価を行い、最適なパラメータを探索します。
各試行では、Dockerイメージのビルド、コンテナの実行、および結果の解析を行います。

使い方:
$ python3 optuna_minimize_sum_laps.py

確認方法:
$ optuna-dashboard sqlite:///autoware-turning.db
"""

import optuna
import subprocess
import os
import json
import glob
import threading
import time
import sys
import xml.etree.ElementTree as ET
import shutil


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


def objective(trial):
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
        run_result = run_subprocess_with_spinner(
            ["./docker_run.sh", "eval", "cpu"],
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
            return total_time

    except subprocess.TimeoutExpired:
        print("Process timed out")
        return 225.0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 225.0
    finally:
        # Skip per-trial restoration - will be handled at script end
        pass


def main():
    # Create single backup at script start
    launch_file_path = "./aichallenge/workspace/src/aichallenge_submit/aichallenge_submit_launch/launch/reference.launch.xml"
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    backup_path = backup_launch_file(launch_file_path, output_dir)
    print(f"Initial backup created at: {backup_path}")

    try:
        study = optuna.create_study(
            direction="minimize",
            study_name="autoware-turning",
            storage="sqlite:///autoware-turning.db",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=100)

        print("=" * 50)
        print("Study finished")
        print("Number of trials:", len(study.trials))
        print("Best trial:")
        print("  Best parameters:", study.best_params)
        print("  Best score     :", study.best_value)
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"Error during optimization: {e}")
    finally:
        # Restore backup file at script end
        if os.path.exists(backup_path):
            if restore_launch_file(launch_file_path, backup_path):
                print(f"Launch file restored from backup: {backup_path}")
                # Clean up backup file
                os.remove(backup_path)
                print("Backup file cleaned up")
            else:
                print("Failed to restore launch file from backup")
        print("=" * 50)


if __name__ == "__main__":
    main()
