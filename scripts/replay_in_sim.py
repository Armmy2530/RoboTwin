#!/usr/bin/env python3
"""
replay_in_sim.py

Physically replay a recorded episode trajectory inside the SAPIEN 3D simulation environment.

Features:
  - Loads the task environment, seed, and table setup.
  - Replays the saved arm trajectories and gripper actuations in real-time PhysX.
  - Opens the interactive SAPIEN 3D viewer:
      - Pauses at the start so you can orbit/fly around and inspect the scene.
      - Press [SPACEBAR] in the viewer to play / pause the physical motion.
      - Fly camera: Hold RMB + W/A/S/D. Orbit: Hold LMB.
  - Select file by exact path, -e <idx>, --task <name>, or interactive menu (--list).

Usage:
  # Replay the latest episode:
  pixi run python scripts/replay_in_sim.py

  # Replay a specific episode index and task:
  pixi run python scripts/replay_in_sim.py --task adjust_bottle -e 0

  # Replay an exact HDF5 file:
  pixi run python scripts/replay_in_sim.py data/demo_clean_yam/adjust_bottle/yam_yam/data/episode_0000000.hdf5

  # Browse all available episodes interactively:
  pixi run python scripts/replay_in_sim.py --list
"""

import sys
import os
import re
import yaml
import json
import argparse
import importlib
import numpy as np

# Ensure repo root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset_utils import resolve_dataset_file, scan_all_episodes
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs import *


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.safe_load(f)
    return embodiment_args


def replay_episode_in_sim(hdf5_path, pause_at_start=True):
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Dataset file not found: {hdf5_path}")

    # Deduce task_config, task_name, embodiment from path
    # Expected: data/<config>/<task>/<embodiment>/data/episode_XXXXXXX.hdf5
    rel_path = os.path.relpath(hdf5_path, os.path.join(REPO_ROOT, "data"))
    parts = rel_path.split(os.sep)

    if len(parts) >= 4:
        task_config = parts[0]
        task_name = parts[1]
    else:
        raise ValueError(f"Could not infer task configuration from path: {hdf5_path}")

    match = re.search(r"episode_?(\d+)", os.path.basename(hdf5_path))
    episode_idx = int(match.group(1)) if match else 0

    print("=" * 65)
    print(" RoboTwin In-Simulation Physical Replay")
    print(f" Dataset     : {hdf5_path}")
    print(f" Task Name   : {task_name}")
    print(f" Config Name : {task_config}")
    print(f" Episode Idx : {episode_idx}")
    print("=" * 65)

    # 1. Load Task Config
    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["render_freq"] = 1  # Force viewer rendering enabled for replay
    args["save_data"] = False  # Do not overwrite dataset while replaying

    # 2. Embodiment config
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f)

    def get_embodiment_file(emb_type):
        return _embodiment_types[emb_type]["file_path"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    embodiment_dir = re.sub(r"[^A-Za-z0-9_]+", "_", str(embodiment_type[0])).lower()
    save_path = os.path.join(REPO_ROOT, "data", task_config, str(task_name), embodiment_dir)
    args["save_path"] = save_path

    # Read seed
    seed = episode_idx
    seed_file = os.path.join(save_path, "seed.txt")
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            seeds = f.read().split()
            if len(seeds) > episode_idx:
                seed = int(seeds[episode_idx])

    # 3. Instantiate task
    task_module = importlib.import_module(f"envs.{task_name}")
    task_class = getattr(task_module, task_name)
    env = task_class()

    print(f"Setting up scene with seed={seed}...")
    env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)

    # 4. Load trajectory data
    traj_data = env.load_tran_data(episode_idx)
    args["left_joint_path"] = traj_data["left_joint_path"]
    args["right_joint_path"] = traj_data["right_joint_path"]
    env.set_path_lst(args)

    print("\n" + "=" * 65)
    print(" SAPIEN 3D VIEWER OPENED")
    print("  • Look around : Hold RMB + W/A/S/D (fly) or LMB (orbit)")
    print("  • Zoom / Pan  : Scroll wheel / Hold RMB + Drag")
    print("  • Play / Pause: Press SPACEBAR in the viewer window")
    print("=" * 65 + "\n")

    if pause_at_start and hasattr(env, "viewer") and env.viewer is not None:
        env.viewer.paused = True
        print("Simulation is PAUSED at the start. Press SPACEBAR in viewer to start physical motion!")
        while not env.viewer.closed and env.viewer.paused:
            env.scene.update_render()
            env.viewer.render()

    if hasattr(env, "viewer") and not env.viewer.closed:
        print("Executing physical trajectory in SAPIEN...")
        info = env.play_once()
        print(f"Replay completed! Execution status: {info}")

        # Keep viewer open until user closes window
        print("Trajectory finished. Explore the final state or close the window to exit.")
        while not env.viewer.closed:
            env.scene.update_render()
            env.viewer.render()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Physically replay RoboTwin recorded trajectory in SAPIEN 3D simulation."
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
        help="Episode index to replay (e.g. -e 0)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task name (e.g. 'adjust_bottle')",
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
        "--no-pause",
        action="store_true",
        default=False,
        help="Do not pause at start; start motion immediately upon launch.",
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

    replay_episode_in_sim(
        hdf5_path=target_file,
        pause_at_start=not args.no_pause,
    )
