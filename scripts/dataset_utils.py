#!/usr/bin/env python3
"""
dataset_utils.py

Shared utilities to locate, list, filter, and resolve RoboTwin HDF5 dataset files.
Supports:
  - Exact file paths (relative or absolute)
  - Short filenames (e.g. 'episode_0000000.hdf5' or '0.hdf5')
  - Episode index filtering (-e / --episode 0)
  - Task and config filtering (--task adjust_bottle --config demo_clean_yam)
  - Interactive terminal table selection (--list / -l)
"""

import os
import re
import glob
import h5py
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")


def scan_all_episodes(data_root=None, task=None, config=None):
    """Scan data_root recursively for all episode_*.hdf5 files and return metadata list."""
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    pattern = os.path.join(data_root, "**", "episode_*.hdf5")
    files = glob.glob(pattern, recursive=True)

    results = []
    for filepath in files:
        rel_path = os.path.relpath(filepath, data_root)
        parts = rel_path.split(os.sep)

        # Expected layout: <config>/<task>/<embodiment>/data/episode_XXXXXXX.hdf5
        cfg_name = parts[0] if len(parts) >= 4 else "unknown"
        task_name = parts[1] if len(parts) >= 4 else "unknown"
        emb_name = parts[2] if len(parts) >= 4 else "unknown"

        if task and task.lower() not in task_name.lower():
            continue
        if config and config.lower() not in cfg_name.lower():
            continue

        filename = os.path.basename(filepath)
        match = re.search(r"episode_?(\d+)", filename)
        ep_idx = int(match.group(1)) if match else -1

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))

        frames = None
        try:
            with h5py.File(filepath, "r") as h:
                cams = list(h.get("vision", {}).keys())
                if cams:
                    frames = len(h[f"vision/{cams[0]}/colors"])
        except Exception:
            frames = "?"

        results.append({
            "index": ep_idx,
            "filename": filename,
            "path": filepath,
            "rel_path": rel_path,
            "task": task_name,
            "config": cfg_name,
            "embodiment": emb_name,
            "frames": frames,
            "size_mb": size_mb,
            "mtime": mtime,
        })

    # Sort by mtime descending (newest first)
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results


def print_episode_table(episodes):
    """Print a clean formatted table of available episodes."""
    if not episodes:
        print("\n  [!] No matching episode_*.hdf5 files found in data/.\n")
        return

    print("\n" + "=" * 92)
    print(f" {'#':<3} | {'Task':<22} | {'Embodiment':<16} | {'Episode':<10} | {'Frames':<7} | {'Size':<8} | {'Modified'}")
    print("-" * 92)
    for i, ep in enumerate(episodes):
        idx_str = f"#{i + 1}"
        frames_str = str(ep["frames"]) if ep["frames"] is not None else "?"
        size_str = f"{ep['size_mb']:.1f} MB"
        mtime_str = ep["mtime"].strftime("%Y-%m-%d %H:%M")
        print(f" {idx_str:<3} | {ep['task']:<22} | {ep['embodiment']:<16} | {ep['filename']:<10} | {frames_str:<7} | {size_str:<8} | {mtime_str}")
    print("=" * 92 + "\n")


def resolve_dataset_file(
    file_arg=None,
    episode=None,
    task=None,
    config=None,
    list_mode=False,
    data_root=None,
):
    """
    Resolve and return a single HDF5 dataset file path based on user specifications.
    
    Returns:
        str: Absolute path to the resolved HDF5 file, or None if cancelled/not found.
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    # Case 1: Direct path exists on disk
    if file_arg and os.path.isfile(file_arg):
        return os.path.abspath(file_arg)

    # Fetch all candidate episodes
    episodes = scan_all_episodes(data_root=data_root, task=task, config=config)

    # Case 2: Interactive list requested
    if list_mode:
        print_episode_table(episodes)
        if not episodes:
            return None
        try:
            choice = input(f"Select episode [1-{len(episodes)}] (or press Enter for newest, 'q' to quit): ").strip()
            if choice.lower() in ("q", "quit", "exit"):
                return None
            if not choice:
                return episodes[0]["path"]
            chosen_idx = int(choice) - 1
            if 0 <= chosen_idx < len(episodes):
                return episodes[chosen_idx]["path"]
            else:
                print("Invalid selection.")
                return None
        except (ValueError, KeyboardInterrupt):
            return None

    # Case 3: Match short filename or partial string (e.g. "episode_0000000.hdf5" or "0.hdf5")
    if file_arg:
        # Check by exact basename
        for ep in episodes:
            if ep["filename"] == file_arg or ep["filename"] == f"episode_{file_arg:0>7}.hdf5":
                return ep["path"]
        # Partial match
        for ep in episodes:
            if file_arg in ep["path"]:
                return ep["path"]
        print(f"Error: Specified file '{file_arg}' not found in {data_root}.")
        return None

    # Case 4: Match by integer episode number (-e / --episode)
    if episode is not None:
        matches = [ep for ep in episodes if ep["index"] == episode]
        if matches:
            return matches[0]["path"]
        print(f"Error: Episode index {episode} not found (task={task}, config={config}).")
        return None

    # Case 5: Default to newest matching episode
    if episodes:
        return episodes[0]["path"]

    print(f"Error: No matching episodes found in {data_root}.")
    return None
