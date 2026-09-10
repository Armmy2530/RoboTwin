#!/usr/bin/env python3
"""
rerun_visualize.py

Stream and visualize RoboTwin / XPolicyLab HDF5 datasets in Rerun (https://rerun.io).

Features:
  - Wrist camera (cam_left_wrist, cam_right_wrist) and head camera video streams.
  - Interactive multi-line time-series plots for 6-DoF arm joint angles.
  - Gripper state & action curves.
  - 3D spatial tracking of robot end-effector positions and trajectory lines.
  - Synchronized timeline scrubber with millisecond precision.
  - Export to standalone .rrd recording files.

Usage:
  # View the latest collected episode in interactive Rerun desktop app:
  pixi run python scripts/rerun_visualize.py

  # View a specific HDF5 file:
  pixi run python scripts/rerun_visualize.py data/demo_clean_yam/adjust_bottle/yam_yam/data/episode_0000000.hdf5

  # Save as .rrd file without launching desktop GUI:
  pixi run python scripts/rerun_visualize.py -s episode_review.rrd

  # Filter by task name:
  pixi run python scripts/rerun_visualize.py --task adjust_bottle
"""

import sys
import os
import glob
import argparse
import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Ensure data/ directory is importable for decode_image_bit
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

from dataset_utils import resolve_dataset_file

try:
    from decode_image_bit import decode_image_bit
except ImportError:
    import cv2
    def decode_image_bit(buf):
        if isinstance(buf, (bytes, bytearray)):
            buf = buf.rstrip(b"\0")
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def find_latest_hdf5(task_filter=None, data_root=None):
    """Find the most recently modified episode_*.hdf5 file."""
    if data_root is None:
        data_root = os.path.join(REPO_ROOT, "data")

    pattern = os.path.join(data_root, "**", "episode_*.hdf5")
    files = glob.glob(pattern, recursive=True)

    if task_filter:
        files = [f for f in files if task_filter in f]

    if not files:
        return None

    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def decode_frame(raw_buf):
    """Decode encoded byte buffer into uint8 RGB numpy array."""
    if isinstance(raw_buf, np.ndarray) and raw_buf.dtype == np.uint8 and raw_buf.ndim >= 3:
        return raw_buf
    return decode_image_bit(raw_buf)


def build_rerun_blueprint(has_right_arm=False, active_cams=None):
    """Construct an optimized multi-view Rerun blueprint layout."""
    if active_cams is None:
        active_cams = ["cam_head", "cam_left_wrist"]

    # Camera panels
    cam_views = []
    for cam in active_cams:
        cam_title = cam.replace("cam_", "").replace("_", " ").title() + " View"
        cam_views.append(
            rrb.Spatial2DView(
                origin=f"world/cameras/{cam}",
                name=cam_title,
            )
        )

    # Time series plots
    ts_views = [
        rrb.TimeSeriesView(
            origin="robot/left_arm",
            name="Left Arm Joint States (rad) & Gripper",
        )
    ]
    if has_right_arm:
        ts_views.append(
            rrb.TimeSeriesView(
                origin="robot/right_arm",
                name="Right Arm Joint States (rad) & Gripper",
            )
        )

    # 3D spatial workspace view
    spatial_3d = rrb.Spatial3DView(
        origin="world",
        name="3D Workspace & End-Effector Paths",
    )

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*cam_views),
            rrb.Horizontal(*ts_views, spatial_3d),
        ),
        collapse_panels=False,
    )
    return blueprint


def find_free_port(start_port=9877):
    import socket
    for p in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


def visualize_in_rerun(hdf5_path, save_rrd=None, fps=25, spawn=True, web=False, web_port=9095):
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    filename = os.path.basename(hdf5_path)
    print("=" * 65)
    print(" RoboTwin Rerun Visualizer" + (" (Web Browser Mode)" if web else ""))
    print(f" Dataset : {hdf5_path}")
    print("=" * 65)

    with h5py.File(hdf5_path, "r") as f:
        # Inspect available vision and state groups
        if "vision" not in f:
            raise KeyError("HDF5 file missing 'vision' group.")

        cam_names = list(f["vision"].keys())
        print(f"Found camera streams : {cam_names}")

        has_left_joints = "state/left_arm_joint_states" in f
        has_right_joints = "state/right_arm_joint_states" in f
        has_left_ee = "state/left_ee_poses" in f
        has_right_ee = "state/right_ee_poses" in f

        sample_cam = cam_names[0]
        total_frames = len(f[f"vision/{sample_cam}/colors"])
        print(f"Total episode frames : {total_frames}")

        # Instruction metadata
        instruction = ""
        if "instructions" in f:
            raw_inst = f["instructions"][()]
            if isinstance(raw_inst, (bytes, bytearray)):
                raw_inst = raw_inst.decode("utf-8")
            instruction = str(raw_inst)
            if instruction.startswith("['") and instruction.endswith("']"):
                instruction = instruction[2:-2]
        print(f"Task Instruction     : {instruction}")

        # Read state arrays into memory
        left_joints = np.array(f["state/left_arm_joint_states"]) if has_left_joints else None
        right_joints = np.array(f["state/right_arm_joint_states"]) if has_right_joints else None
        left_grips = np.array(f["state/left_ee_joint_states"]).squeeze() if "state/left_ee_joint_states" in f else None
        right_grips = np.array(f["state/right_ee_joint_states"]).squeeze() if "state/right_ee_joint_states" in f else None
        left_ee_poses = np.array(f["state/left_ee_poses"]) if has_left_ee else None
        right_ee_poses = np.array(f["state/right_ee_poses"]) if has_right_ee else None

        # Check action arrays (optional comparison)
        has_action_left = "action/left_arm_joint_states" in f
        action_left_joints = np.array(f["action/left_arm_joint_states"]) if has_action_left else None

        # Initialize Rerun recording
        app_id = f"robotwin_{filename.replace('.hdf5', '')}"

        local_url = None
        if web:
            import urllib.parse
            actual_web_port = find_free_port(web_port)
            grpc_port = find_free_port(9877)
            rr.init(app_id, spawn=False)
            server_uri = rr.serve_grpc(grpc_port=grpc_port)
            rr.serve_web_viewer(connect_to=server_uri, web_port=actual_web_port, open_browser=True)
            local_url = f"http://localhost:{actual_web_port}/?url={urllib.parse.quote(server_uri, safe='')}"
            hosted_url = f"https://app.rerun.io/?url={urllib.parse.quote(server_uri, safe='')}"
            print("\n" + "=" * 65)
            print(" Rerun Web Viewer Active:")
            print(f" -> Local Browser URL : {local_url}")
            print(f" -> Cloud Web URL     : {hosted_url}")
            print("=" * 65 + "\n")
        else:
            rr.init(app_id, spawn=(spawn and save_rrd is None))

        if save_rrd:
            print(f"Saving Rerun recording to: {save_rrd}")
            rr.save(save_rrd)

        # Set coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Apply layout blueprint
        blueprint = build_rerun_blueprint(has_right_arm=has_right_joints, active_cams=cam_names)
        rr.send_blueprint(blueprint)

        # Log instruction
        if instruction:
            rr.log("info/instruction", rr.TextDocument(f"### Instruction:\n**{instruction}**"), static=True)

        left_ee_traj = []
        right_ee_traj = []

        print("Streaming dataset frames to Rerun...")
        for frame_idx in range(total_frames):
            sim_time = frame_idx / fps

            # Synchronize timeline
            rr.set_time("frame_idx", sequence=frame_idx)
            rr.set_time("sim_time", duration=sim_time)

            # 1. Cameras (Wrist & Head)
            for cam_name in cam_names:
                raw_buf = f[f"vision/{cam_name}/colors"][frame_idx]
                img_rgb = decode_frame(raw_buf)
                rr.log(f"world/cameras/{cam_name}", rr.Image(img_rgb))

            # 2. Left Arm Joint States & Gripper
            if left_joints is not None:
                for j in range(left_joints.shape[1]):
                    rr.log(f"robot/left_arm/joints/joint_{j}", rr.Scalars(float(left_joints[frame_idx, j])))
            if left_grips is not None:
                rr.log("robot/left_arm/gripper", rr.Scalars(float(left_grips[frame_idx])))

            # Left Arm Action Target (if present, for state vs action comparison)
            if action_left_joints is not None:
                for j in range(action_left_joints.shape[1]):
                    rr.log(f"action/left_arm/joints/joint_{j}", rr.Scalars(float(action_left_joints[frame_idx, j])))

            # 3. Right Arm Joint States & Gripper
            if right_joints is not None:
                for j in range(right_joints.shape[1]):
                    rr.log(f"robot/right_arm/joints/joint_{j}", rr.Scalars(float(right_joints[frame_idx, j])))
            if right_grips is not None:
                rr.log("robot/right_arm/gripper", rr.Scalars(float(right_grips[frame_idx])))

            # 4. End-Effector 3D Positions & Trajectory Paths
            if left_ee_poses is not None:
                pos = left_ee_poses[frame_idx, :3]
                left_ee_traj.append(pos)
                rr.log(
                    "world/robot/left_ee",
                    rr.Points3D([pos], radii=[0.015], colors=[[0, 200, 255]])
                )
                if len(left_ee_traj) > 1:
                    rr.log(
                        "world/robot/left_ee_trajectory",
                        rr.LineStrips3D([np.array(left_ee_traj)], colors=[[0, 150, 255]], radii=[0.003])
                    )

            if right_ee_poses is not None:
                pos = right_ee_poses[frame_idx, :3]
                right_ee_traj.append(pos)
                rr.log(
                    "world/robot/right_ee",
                    rr.Points3D([pos], radii=[0.015], colors=[[255, 140, 0]])
                )
                if len(right_ee_traj) > 1:
                    rr.log(
                        "world/robot/right_ee_trajectory",
                        rr.LineStrips3D([np.array(right_ee_traj)], colors=[[255, 100, 0]], radii=[0.003])
                    )

            if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                print(f"  Streamed {frame_idx + 1}/{total_frames} frames...", end="\r")

        print(f"\nAll {total_frames} frames successfully logged into Rerun!")
        if save_rrd:
            print(f"Recording file saved: {os.path.abspath(save_rrd)}")
        elif web:
            import time
            print("\n" + "=" * 65)
            print(" [INFO] Rerun Web Viewer is streaming in your browser!")
            print(f" [INFO] Open URL: {local_url}")
            print(" [INFO] Press Ctrl+C to stop the web viewer server.")
            print("=" * 65 + "\n")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[INFO] Rerun web viewer server stopped.")
        else:
            print("Interactive Rerun session is running.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize RoboTwin HDF5 datasets in Rerun (Wrist Camera + Joint States)."
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        nargs="?",
        default=None,
        help="Path or short filename of episode_*.hdf5 file. If omitted, auto-resolves.",
    )
    parser.add_argument(
        "-e",
        "--episode",
        type=int,
        default=None,
        help="Specify episode index to replay (e.g. -e 0)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task name (e.g. 'adjust_bottle' or 'beat_block_hammer')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Filter by task config name (e.g. 'demo_clean_yam')",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        default=False,
        help="Display an interactive table of all available episodes to select from.",
    )
    parser.add_argument(
        "-w",
        "--web",
        action="store_true",
        default=False,
        help="Open and stream Rerun visualization in a web browser (instead of native desktop window).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9095,
        help="Port to serve Rerun web viewer on (default: 9095 to avoid conflict with Cockpit 9090).",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Save to an .rrd recording file instead of opening interactive viewer",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Timeline playback rate (default: 25 FPS)",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        default=False,
        help="Do not spawn the Rerun viewer desktop app",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    target_file = resolve_dataset_file(
        file_arg=args.hdf5_file,
        episode=args.episode,
        task=args.task,
        config=args.config,
        list_mode=args.list,
    )

    if target_file is None:
        sys.exit(0 if args.list else 1)

    print(f"Selected dataset: {target_file}")
    visualize_in_rerun(
        hdf5_path=target_file,
        save_rrd=args.save,
        fps=args.fps,
        spawn=not args.no_spawn and not args.web,
        web=args.web,
        web_port=args.web_port,
    )
