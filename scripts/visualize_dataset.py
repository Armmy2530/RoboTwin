#!/usr/bin/env python3
"""
visualize_dataset.py

Interactive player and multi-camera video exporter for RoboTwin / XPolicyLab HDF5 datasets.

Features:
  - Interactive playback of all cameras (Head, Left Wrist, Right Wrist) side-by-side.
  - Telemetry HUD: Frame index, instruction, gripper states, and joint positions.
  - Interactive controls:
      [SPACE]       : Pause / Play
      [A] / [Left]  : Previous frame (when paused)
      [D] / [Right] : Next frame (when paused)
      [R]           : Restart episode from frame 0
      [Q] / [ESC]   : Exit
  - Export synchronized multi-camera MP4 video (--export-video / -s).

Usage:
  # Play the latest generated episode:
  pixi run python scripts/visualize_dataset.py

  # Play a specific HDF5 file:
  pixi run python scripts/visualize_dataset.py data/demo_clean_yam/adjust_bottle/yam_yam/data/episode_0000000.hdf5

  # Export a multi-camera side-by-side MP4 video:
  pixi run python scripts/visualize_dataset.py --export-video output.mp4

  # Filter by task name:
  pixi run python scripts/visualize_dataset.py --task adjust_bottle
"""

import sys
import os
import glob
import argparse
import h5py
import cv2
import numpy as np

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
    def decode_image_bit(buf):
        if isinstance(buf, (bytes, bytearray)):
            buf = buf.rstrip(b"\0")
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def find_latest_hdf5(task_filter=None, data_root=None):
    """Locate the most recently modified episode_*.hdf5 in data/."""
    if data_root is None:
        data_root = os.path.join(REPO_ROOT, "data")

    pattern = os.path.join(data_root, "**", "episode_*.hdf5")
    files = glob.glob(pattern, recursive=True)

    if task_filter:
        files = [f for f in files if task_filter in f]

    if not files:
        return None

    # Sort by file modification time (newest first)
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def decode_frame(raw_buf):
    """Safely decode one image buffer into RGB uint8 numpy array."""
    if isinstance(raw_buf, np.ndarray) and raw_buf.dtype == np.uint8 and raw_buf.ndim >= 3:
        return raw_buf
    return decode_image_bit(raw_buf)


def render_hud(frame_img, frame_idx, total_frames, instruction, left_grip, right_grip):
    """Overlay telemetry and instruction bar onto the composite frame."""
    h, w, _ = frame_img.shape
    bar_h = 50
    hud_bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

    # Frame info
    progress_text = f"Frame: {frame_idx + 1}/{total_frames}"
    cv2.putText(hud_bar, progress_text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Gripper states
    grip_text = f"Gripper L: {left_grip:.3f} | R: {right_grip:.3f}" if (left_grip is not None and right_grip is not None) else ""
    cv2.putText(hud_bar, grip_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)

    # Instruction
    if instruction:
        instr_text = f"Instruction: {instruction}"
        cv2.putText(hud_bar, instr_text, (w // 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 180), 1, cv2.LINE_AA)

    # Help text
    help_text = "[SPACE] Pause | [A/D] Step | [R] Reset | [Q] Quit"
    cv2.putText(hud_bar, help_text, (w - 380, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # Visual progress bar at bottom of HUD
    progress_ratio = (frame_idx + 1) / total_frames
    bar_width = int(w * progress_ratio)
    cv2.rectangle(hud_bar, (0, bar_h - 4), (bar_width, bar_h), (0, 165, 255), -1)

    return np.vstack([frame_img, hud_bar])


def visualize_dataset(hdf5_path, export_video=None, playback_fps=25, tile_height=360):
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    print("=" * 65)
    print(" RoboTwin Dataset Visualizer")
    print(f" File: {hdf5_path}")
    print("=" * 65)

    with h5py.File(hdf5_path, "r") as f:
        # Check vision group
        if "vision" not in f:
            raise KeyError("HDF5 file does not contain a 'vision' group.")

        cam_groups = list(f["vision"].keys())
        print(f"Found camera views: {cam_groups}")

        # Standard camera display order
        ordered_cams = []
        for pref in ["cam_head", "head_camera", "cam_left_wrist", "left_camera", "cam_right_wrist", "right_camera"]:
            if pref in cam_groups and pref not in ordered_cams:
                ordered_cams.append(pref)
        for c in cam_groups:
            if c not in ordered_cams:
                ordered_cams.append(c)

        # Total frames
        sample_cam = ordered_cams[0]
        total_frames = len(f[f"vision/{sample_cam}/colors"])
        print(f"Total frames: {total_frames}")

        # Retrieve instructions
        instruction = ""
        if "instructions" in f:
            raw_inst = f["instructions"][()]
            if isinstance(raw_inst, (bytes, bytearray)):
                raw_inst = raw_inst.decode("utf-8")
            instruction = str(raw_inst)
            if instruction.startswith("['") and instruction.endswith("']"):
                instruction = instruction[2:-2]

        # Read gripper states if present
        left_grips = None
        right_grips = None
        if "state/left_ee_joint_states" in f:
            left_grips = np.array(f["state/left_ee_joint_states"]).squeeze()
        if "state/right_ee_joint_states" in f:
            right_grips = np.array(f["state/right_ee_joint_states"]).squeeze()

        # Video writer setup
        video_writer = None

        frame_idx = 0
        paused = False
        window_name = f"RoboTwin Player - {os.path.basename(hdf5_path)}"

        if export_video is None:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        delay_ms = max(1, int(1000.0 / playback_fps))

        while 0 <= frame_idx < total_frames:
            tiles = []
            for cam_name in ordered_cams:
                raw_buf = f[f"vision/{cam_name}/colors"][frame_idx]
                img_rgb = decode_frame(raw_buf)

                # Resize to common height
                h, w = img_rgb.shape[:2]
                target_w = int(tile_height * (w / h))
                resized = cv2.resize(img_rgb, (target_w, tile_height), interpolation=cv2.INTER_LINEAR)

                # Convert RGB to BGR for OpenCV display/saving
                img_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

                # Camera title badge
                label = cam_name.replace("cam_", "").replace("_", " ").upper()
                cv2.rectangle(img_bgr, (0, 0), (140, 25), (30, 30, 30), -1)
                cv2.putText(img_bgr, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                tiles.append(img_bgr)

            # Combine tiles horizontally
            composite = np.hstack(tiles)

            # Gripper values
            lg = left_grips[frame_idx] if left_grips is not None else None
            rg = right_grips[frame_idx] if right_grips is not None else None

            # Add HUD
            full_frame = render_hud(composite, frame_idx, total_frames, instruction, lg, rg)

            # If exporting video
            if export_video:
                if video_writer is None:
                    fh, fw = full_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(export_video, fourcc, playback_fps, (fw, fh))
                    print(f"Exporting synchronized multi-camera video to: {export_video}")

                video_writer.write(full_frame)
                frame_idx += 1
                if frame_idx % 25 == 0 or frame_idx == total_frames:
                    print(f"  Processed {frame_idx}/{total_frames} frames...", end="\r")
                continue

            # Interactive GUI display
            cv2.imshow(window_name, full_frame)

            wait_time = 0 if paused else delay_ms
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord(" "):  # Space: Pause / Play
                paused = not paused
            elif key == ord("d") or key == 83:  # D or Right Arrow
                if paused:
                    frame_idx = min(total_frames - 1, frame_idx + 1)
            elif key == ord("a") or key == 81:  # A or Left Arrow
                if paused:
                    frame_idx = max(0, frame_idx - 1)
            elif key == ord("r"):  # R: Restart
                frame_idx = 0
            else:
                if not paused:
                    frame_idx += 1
                    if frame_idx >= total_frames:
                        # Loop playback
                        frame_idx = 0

        if video_writer:
            video_writer.release()
            print(f"\nSuccessfully exported multi-camera video to: {export_video}")

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize and play back RoboTwin / XPolicyLab HDF5 datasets."
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
        "-s",
        "--export-video",
        type=str,
        default=None,
        help="Export synchronized multi-camera MP4 video to specified file path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Playback / export frame rate (default: 25 FPS)",
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

    print(f"Selected dataset file: {target_file}")
    visualize_dataset(
        hdf5_path=target_file,
        export_video=args.export_video,
        playback_fps=args.fps,
    )
