# RoboTwin 2.0 Quick Command Cheat Sheet

This guide provides a comprehensive reference of the essential commands for **robot previews, camera checks, data collection, dataset replay, and Rerun telemetry inspection**.

---

## Table of Contents
1. [Camera Frame Checks & 3D Space Inspection](#1-camera-frame-checks--3d-space-inspection)
2. [Data Collection](#2-data-collection)
3. [Multimodal Dataset Visualization (Rerun)](#3-multimodal-dataset-visualization-rerun)
4. [Multi-Camera Video Player & Exporter (OpenCV)](#4-multi-camera-video-player--exporter-opencv)
5. [Physical In-Simulation Trajectory Replay (SAPIEN)](#5-physical-in-simulation-trajectory-replay-sapien)
6. [Task & Embodiment Reference](#6-task--embodiment-reference)

---

## 1. Camera Frame Checks & 3D Space Inspection

Use `scripts/capture_camera_frames.py` to quickly check camera angles, framing, and tabletop layout without executing full tasks.

### Capture Camera Overview Images (PNG)
```bash
# Default (YAM with ABC RealSense D405/D455 calibration):
pixi run python scripts/capture_camera_frames.py

# ARX-X5 with ABC calibration:
pixi run python scripts/capture_camera_frames.py --embodiment ARX-X5-abc

# Original default embodiments (for comparison):
pixi run python scripts/capture_camera_frames.py --embodiment yam
pixi run python scripts/capture_camera_frames.py --embodiment ARX-X5

# Specify custom output filename:
pixi run python scripts/capture_camera_frames.py -o my_camera_overview.png

# Also save separate PNGs for each individual camera:
pixi run python scripts/capture_camera_frames.py --save-individual
```

### Open Interactive 3D Viewer Paused at Start (`-v` / `--viewer`)
```bash
# Open 3D viewer paused at robot home pose:
pixi run python scripts/capture_camera_frames.py -v

# For ARX-X5:
pixi run python scripts/capture_camera_frames.py --embodiment ARX-X5-abc -v
```

#### SAPIEN 3D Viewer Navigation Controls:
| Action | Key / Mouse Binding |
| :--- | :--- |
| **Look around / Orbit** | Hold **Left Mouse Button (LMB)** + Drag |
| **Pan Camera** | Hold **Right Mouse Button (RMB)** + Drag (or **Middle Mouse Button**) |
| **Zoom In / Out** | **Mouse Scroll Wheel** |
| **Fly Mode (Free Cam)** | Hold **RMB** and use <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> (<kbd>E</kbd> = up, <kbd>Q</kbd> = down) |
| **Pause / Play** | Press <kbd>Spacebar</kbd> |
| **Close / Exit** | Press <kbd>Esc</kbd> or close the window |

---

## 2. Data Collection

Collect robot demonstration episodes for downstream imitation learning (ACT, Diffusion Policy).

### Direct Python Command
```bash
# Format: python scripts/collect_data.py <task_name> <task_config>

# Pick bottles and put in dustbin/trash can (YAM with ABC calibration):
pixi run python scripts/collect_data.py put_bottles_dustbin abc_camera_config_yam

# Adjust bottle task (YAM):
pixi run python scripts/collect_data.py adjust_bottle abc_camera_config_yam

# Adjust bottle task (ARX-X5):
pixi run python scripts/collect_data.py adjust_bottle abc_camera_config_arx5

# Default configs:
pixi run python scripts/collect_data.py adjust_bottle demo_clean_yam
```

### Using the Bash Script Wrapper (`collect_data.sh`)
```bash
# Format: bash collect_data.sh <task_name> <task_config> <gpu_id>
# Note: '0' specifies GPU device 0 (CUDA_VISIBLE_DEVICES=0)

bash collect_data.sh put_bottles_dustbin abc_camera_config_yam 0
bash collect_data.sh adjust_bottle abc_camera_config_yam 0
```

---

## 3. Multimodal Dataset Visualization (Rerun)

Use `scripts/rerun_visualize.py` to inspect **wrist camera frames, head camera, 6-DoF joint state angles (radians), gripper status, and 3D end-effector trails** simultaneously with a synchronized timeline.

```bash
# Replay the newest collected episode automatically (desktop GUI):
pixi run python scripts/rerun_visualize.py

# --- WEB BROWSER MODE (-w / --web) ---
# View in web browser (Chrome, Firefox, Edge) - no native GUI required!
pixi run python scripts/rerun_visualize.py --web
pixi run python scripts/rerun_visualize.py -e 0 --web
pixi run python scripts/rerun_visualize.py --task put_bottles_dustbin -e 0 --web

# Replay a specific task and episode index:
pixi run python scripts/rerun_visualize.py --task put_bottles_dustbin -e 0
pixi run python scripts/rerun_visualize.py --task adjust_bottle -e 0

# Interactive selection table (lists all recorded episodes):
pixi run python scripts/rerun_visualize.py --list

# Replay by direct path or short filename:
pixi run python scripts/rerun_visualize.py episode_0000000.hdf5
pixi run python scripts/rerun_visualize.py data/demo_clean_yam/adjust_bottle/yam_yam/data/episode_0000000.hdf5

# Export a standalone .rrd recording (no GUI window launched):
pixi run python scripts/rerun_visualize.py -s recording.rrd

# View any saved .rrd recording later:
pixi run rerun recording.rrd

# View any saved .rrd recording in the WEB BROWSER:
pixi run rerun recording.rrd --web-viewer --web-viewer-port 9095
```

---

## 4. Multi-Camera Video Player & Exporter (OpenCV)

Use `scripts/visualize_dataset.py` for lightweight side-by-side camera video playback with HUD overlays (frame counter, instruction, gripper values).

```bash
# Play newest episode:
pixi run python scripts/visualize_dataset.py

# Specify episode index & task:
pixi run python scripts/visualize_dataset.py --task put_bottles_dustbin -e 0

# Interactive episode selection menu:
pixi run python scripts/visualize_dataset.py --list

# Export synchronized multi-camera MP4 video:
pixi run python scripts/visualize_dataset.py -s multi_camera_review.mp4
```

#### Player Controls:
- <kbd>Spacebar</kbd>: Pause / Play
- <kbd>A</kbd> / <kbd>←</kbd>: Step backward 1 frame (when paused)
- <kbd>D</kbd> / <kbd>→</kbd>: Step forward 1 frame (when paused)
- <kbd>R</kbd>: Restart from frame 0
- <kbd>Q</kbd> / <kbd>Esc</kbd>: Exit

---

## 5. Physical In-Simulation Trajectory Replay (SAPIEN)

Use `scripts/replay_in_sim.py` to watch the simulated robot arm **physically execute the trajectory in SAPIEN 3D simulation** with real-time PhysX.

```bash
# Replay newest episode physically in SAPIEN:
pixi run python scripts/replay_in_sim.py

# Specify episode index & task:
pixi run python scripts/replay_in_sim.py --task put_bottles_dustbin -e 0
pixi run python scripts/replay_in_sim.py --task adjust_bottle -e 0

# Interactive episode selection menu:
pixi run python scripts/replay_in_sim.py --list

# Start motion immediately without pausing at the start:
pixi run python scripts/replay_in_sim.py --task put_bottles_dustbin -e 0 --no-pause
```

*By default, the simulation opens paused so you can orbit or fly around the table. Press <kbd>Spacebar</kbd> in the SAPIEN window to start the physical motion.*

---

## 6. Task & Embodiment Reference

### Key Tasks
| Task Name | Description | Objects Involved |
| :--- | :--- | :--- |
| **`put_bottles_dustbin`** | Pick 3 bottles and drop into dustbin/trash can | `114_bottle` (3x), `011_dustbin` |
| **`adjust_bottle`** | Pick up bottle and adjust to target orientation | `114_bottle` |
| **`beat_block_hammer`** | Grasp hammer and strike the wooden block | `020_hammer`, `004_fluted-block` |
| **`shake_bottle`** | Grasp and shake bottle vertically | `114_bottle` |
| **`shake_bottle_horizontally`** | Grasp and shake bottle horizontally | `114_bottle` |

### Key Embodiments
| Embodiment | Config Path | Description |
| :--- | :--- | :--- |
| **`yam-abc`** | `assets/embodiments/yam-abc/` | Dual YAM arms with ABC RealSense D405 wrist extrinsics (40° cant) |
| **`ARX-X5-abc`** | `assets/embodiments/ARX-X5-abc/` | Dual ARX-X5 arms with tuned 35° pitch RealSense D405 mount |
| **`yam`** | `assets/embodiments/yam/` | Default original YAM embodiment |
| **`ARX-X5`** | `assets/embodiments/ARX-X5/` | Default original ARX-X5 embodiment |

### Key Task Configurations
| Config Name | File Path | Settings |
| :--- | :--- | :--- |
| **`abc_camera_config_yam`** | `env_cfg/task_config/abc_camera_config_yam.yml` | `yam-abc`, D405/D455 wide cameras, `save_freq: 10` (25 Hz) |
| **`abc_camera_config_arx5`** | `env_cfg/task_config/abc_camera_config_arx5.yml` | `ARX-X5-abc`, D405/D455 wide cameras, `save_freq: 10` (25 Hz) |
| **`demo_clean_yam`** | `env_cfg/task_config/demo_clean_yam.yml` | YAM arms, `save_freq: 10` (25 Hz) |

### Frequency & Step Calculation
- SAPIEN Physics Step: **$4\,\text{ms}$** ($250\,\text{Hz}$)
- **`save_freq: 10`**: Records an observation every 10 steps $\to$ **$40\,\text{ms}$** interval = **$25\,\text{Hz}$** policy dataset.
- **`save_freq: 5`**: Records an observation every 5 steps $\to$ **$20\,\text{ms}$** interval = **$50\,\text{Hz}$** policy dataset.
