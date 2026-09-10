#!/usr/bin/env python3
"""
capture_camera_frames.py

Quick utility to capture and output single image frames of head and wrist cameras
as PNG files into the RoboTwin scripts folder (or a custom directory).
Designed for fast camera frame verification without executing a full task.

Usage:
    # Default (yam-abc with ABC D405/D455 configs):
    pixi run python scripts/capture_camera_frames.py

    # ARX-X5 with ABC configs:
    pixi run python scripts/capture_camera_frames.py --embodiment ARX-X5-abc

    # Original default embodiments:
    pixi run python scripts/capture_camera_frames.py --embodiment yam
    pixi run python scripts/capture_camera_frames.py --embodiment ARX-X5

    # Specify custom prefix or output directory:
    pixi run python scripts/capture_camera_frames.py --prefix test_ --output-dir ./scripts
"""

import sys
import os
import argparse

# Ensure repo root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sapien.core as sapien
from sapien.render import set_global_config

from envs.robot import Robot
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.camera.camera import Camera


def get_default_task_config(embodiment_name):
    """Auto-select appropriate camera config for embodiment."""
    if "arx" in embodiment_name.lower() and "abc" in embodiment_name.lower():
        return "abc_camera_config_arx5"
    elif "abc" in embodiment_name.lower():
        return "abc_camera_config_yam"
    else:
        return "demo_clean"


def setup_quick_scene():
    """Setup a lightweight SAPIEN rendering scene with table and visual targets."""
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1.8], [1, 1, 1], shadow=True)

    # Standard manipulation table
    table_builder = scene.create_actor_builder()
    table_builder.add_box_visual(
        sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025], material=[0.75, 0.75, 0.75]
    )
    table_builder.add_box_collision(
        sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025]
    )
    table_builder.build_static("table")

    # Table center marker / target bottle
    marker_builder = scene.create_actor_builder()
    marker_builder.add_cylinder_visual(
        sapien.Pose([0, 0, 0.85]), radius=0.035, half_length=0.08, material=[0.15, 0.45, 0.9]
    )
    marker_builder.add_cylinder_collision(
        sapien.Pose([0, 0, 0.85]), radius=0.035, half_length=0.08
    )
    marker_builder.build_static("target_bottle")

    # Secondary reference targets (left and right table zones)
    left_target = scene.create_actor_builder()
    left_target.add_box_visual(sapien.Pose([-0.2, 0.0, 0.79]), half_size=[0.03, 0.03, 0.025], material=[0.9, 0.2, 0.2])
    left_target.build_static("marker_left")

    right_target = scene.create_actor_builder()
    right_target.add_box_visual(sapien.Pose([0.2, 0.0, 0.79]), half_size=[0.03, 0.03, 0.025], material=[0.2, 0.8, 0.3])
    right_target.build_static("marker_right")

    return engine, renderer, scene


def create_montage(camera_images, title_prefix=""):
    """Create a single side-by-side montage image of all cameras."""
    if not camera_images:
        return None

    tiles = []
    tile_h = 300
    for name, img_arr in camera_images.items():
        pil_img = Image.fromarray(img_arr)
        aspect = pil_img.width / pil_img.height
        tile_w = int(tile_h * aspect)
        pil_resized = pil_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        tiles.append((name, pil_resized, img_arr.shape))

    header_h = 50
    footer_h = 35
    padding = 15
    total_w = sum(t[1].width for t in tiles) + padding * (len(tiles) + 1)
    total_h = header_h + tile_h + footer_h + padding

    montage = Image.new("RGB", (total_w, total_h), (30, 32, 40))
    draw = ImageDraw.Draw(montage)

    # Title
    main_title = f"{title_prefix} Camera Frame Check" if title_prefix else "Camera Frame Check"
    draw.text((padding, 15), main_title, fill=(255, 255, 255))

    curr_x = padding
    for name, pil_img, shape in tiles:
        montage.paste(pil_img, (curr_x, header_h))
        # Border
        draw.rectangle(
            [curr_x, header_h, curr_x + pil_img.width, header_h + pil_img.height],
            outline=(100, 150, 240),
            width=2,
        )
        # Subtitle
        label = f"{name} ({shape[1]}x{shape[0]})"
        draw.text((curr_x + 5, header_h + pil_img.height + 8), label, fill=(200, 220, 240))
        curr_x += pil_img.width + padding

    return montage


def capture_frames(
    embodiment_name="yam-abc",
    task_config_name=None,
    output_dir=None,
    prefix="",
    output_name=None,
    save_individual=False,
    camera_type_override=None,
    show_viewer=False,
):
    if output_dir is None:
        output_dir = SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if task_config_name is None:
        task_config_name = get_default_task_config(embodiment_name)

    print(f"==================================================")
    print(f" RoboTwin Camera Frame Capture")
    print(f" Embodiment      : {embodiment_name}")
    print(f" Task Config     : {task_config_name}")
    print(f" Output Dir      : {output_dir}")
    print(f" Save Individual : {save_individual}")
    print(f" Interactive 3D  : {show_viewer}")
    print(f"==================================================")

    # Load task config
    task_config_file = os.path.join(CONFIGS_PATH, f"{task_config_name}.yml")
    if not os.path.exists(task_config_file):
        raise FileNotFoundError(f"Task config not found: {task_config_file}")
    with open(task_config_file, "r") as f:
        task_cfg = yaml.safe_load(f)

    # Load embodiment config registry
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r") as f:
        _embodiment_types = yaml.safe_load(f)

    if embodiment_name not in _embodiment_types:
        available = list(_embodiment_types.keys())
        raise ValueError(f"Unknown embodiment '{embodiment_name}'. Available: {available}")

    robot_rel_dir = _embodiment_types[embodiment_name]["file_path"]
    robot_file = os.path.join(REPO_ROOT, robot_rel_dir)
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r") as f:
        embodiment_args = yaml.safe_load(f)

    camera_cfg = task_cfg.get("camera", {}).copy()
    if camera_type_override:
        camera_cfg["head_camera_type"] = camera_type_override
        camera_cfg["wrist_camera_type"] = camera_type_override

    kwargs = {
        "left_embodiment_config": embodiment_args,
        "right_embodiment_config": embodiment_args,
        "left_robot_file": robot_file,
        "right_robot_file": robot_file,
        "dual_arm_embodied": embodiment_args.get("dual_arm", False),
        "embodiment_dis": 0.6,
        "camera": camera_cfg,
    }

    # Build scene
    engine, renderer, scene = setup_quick_scene()

    # Load Robot
    print("Loading robot model and moving to home state...")
    robot = Robot(scene, need_topp=False, **kwargs)
    robot.init_joints()
    robot.move_to_homestate()

    # Allow physics settling
    for _ in range(30):
        scene.step()

    # Initialize cameras
    print("Initializing camera system...")
    cam_system = Camera(bias=0, **kwargs)
    cam_system.load_camera(scene)
    cam_system.update_wrist_camera(robot.left_camera.get_pose(), robot.right_camera.get_pose())

    # If interactive viewer requested, open SAPIEN 3D viewer paused at start
    if show_viewer:
        from sapien.utils.viewer import Viewer

        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=0.4, y=0.22, z=1.5)
        viewer.set_camera_rpy(r=0, p=-0.8, y=2.45)
        viewer.paused = True

        print("\n" + "=" * 60)
        print(" SAPIEN 3D VIEWER OPENED (PAUSED AT START)")
        print("  • Look around: Hold RMB + W/A/S/D (fly) or LMB (orbit)")
        print("  • Pan        : Hold RMB + Drag or Middle Mouse Button")
        print("  • Zoom       : Mouse Scroll Wheel")
        print("  • Pause/Play : Press SPACEBAR to toggle simulation pause")
        print("  • Finish     : Close the window to save camera snapshot")
        print("=" * 60 + "\n")

        while not viewer.closed:
            if not viewer.paused:
                scene.step()
                cam_system.update_wrist_camera(robot.left_camera.get_pose(), robot.right_camera.get_pose())
            scene.update_render()
            viewer.render()

    # Step & render single frame
    scene.step()
    scene.update_render()
    cam_system.update_picture()

    rgbs = cam_system.get_rgb()
    saved_files = {}
    raw_images = {}

    pfx = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix

    for cam_name, cam_data in rgbs.items():
        img_arr = cam_data["rgb"]
        raw_images[cam_name] = img_arr

        if save_individual:
            filename = f"{pfx}{cam_name}.png" if pfx else f"{cam_name}.png"
            out_path = os.path.join(output_dir, filename)
            Image.fromarray(img_arr).save(out_path)
            saved_files[cam_name] = out_path
            h, w, c = img_arr.shape
            print(f" -> Saved individual {cam_name:12s} : {out_path} ({w}x{h})")

    # Generate and save overview montage (default)
    if len(raw_images) > 0:
        montage = create_montage(raw_images, title_prefix=f"[{embodiment_name}]")
        if montage:
            if output_name:
                overview_filename = output_name if output_name.endswith(".png") else f"{output_name}.png"
            elif pfx:
                overview_filename = f"{pfx}camera_frame_overview.png"
            else:
                overview_filename = f"{embodiment_name}_camera_frame_overview.png"

            overview_path = os.path.join(output_dir, overview_filename)
            montage.save(overview_path)
            saved_files["overview"] = overview_path
            print(f" -> Saved camera overview   : {overview_path}")

    print("Camera frame check completed successfully!\n")
    return saved_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture camera overview montage as PNG in RoboTwin (default: overview only)."
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default="yam-abc",
        help="Embodiment name (e.g. yam-abc, ARX-X5-abc, yam, ARX-X5, aloha-agilex). Default: yam-abc",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        help="Task config yml name (e.g. abc_camera_config_yam). Default: auto-selected based on embodiment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to save PNG file(s). Default: {SCRIPT_DIR}",
    )
    parser.add_argument(
        "-o",
        "--output-name",
        type=str,
        default=None,
        help="Explicit filename for the overview image (e.g. 'camera_overview.png')",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for output PNG filename(s) (e.g. 'yam_' or 'run1_')",
    )
    parser.add_argument(
        "--save-individual",
        "--individual",
        action="store_true",
        default=False,
        help="Also output individual PNGs for each camera in addition to the overview montage.",
    )
    parser.add_argument(
        "--camera-type",
        type=str,
        default=None,
        help="Override camera profile type (e.g. ABC_D405, ABC_D455, D435)",
    )
    parser.add_argument(
        "-v",
        "--viewer",
        action="store_true",
        default=False,
        help="Open interactive SAPIEN 3D viewer paused at start to look around 3D space.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    capture_frames(
        embodiment_name=args.embodiment,
        task_config_name=args.task_config,
        output_dir=args.output_dir,
        prefix=args.prefix,
        output_name=args.output_name,
        save_individual=args.save_individual,
        camera_type_override=args.camera_type,
        show_viewer=args.viewer,
    )
