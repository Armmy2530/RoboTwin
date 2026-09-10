import sys
import os
sys.path.append("./")

import yaml
import numpy as np
from PIL import Image
import sapien.core as sapien
from sapien.render import set_global_config

from envs.robot import Robot
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.camera.camera import Camera

def render_embodiment(embodiment_name, task_config_name, out_prefix):
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

    # Table
    builder = scene.create_actor_builder()
    builder.add_box_visual(sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025], material=[0.8, 0.8, 0.8])
    builder.add_box_collision(sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025])
    builder.build_static("table")

    # Table center marker / target bottle
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(sapien.Pose([0, 0, 0.85]), radius=0.03, half_length=0.08, material=[0.1, 0.45, 0.9])
    builder.add_cylinder_collision(sapien.Pose([0, 0, 0.85]), radius=0.03, half_length=0.08)
    builder.build_static("target_bottle")

    # Load task config
    task_config_file = os.path.join(CONFIGS_PATH, f"{task_config_name}.yml")
    with open(task_config_file, "r") as f:
        task_cfg = yaml.safe_load(f)

    # Load embodiment config
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r") as f:
        _embodiment_types = yaml.safe_load(f)

    robot_file = os.path.join("./", _embodiment_types[embodiment_name]["file_path"])
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r") as f:
        embodiment_args = yaml.safe_load(f)

    kwargs = {
        "left_embodiment_config": embodiment_args,
        "right_embodiment_config": embodiment_args,
        "left_robot_file": robot_file,
        "right_robot_file": robot_file,
        "dual_arm_embodied": embodiment_args.get("dual_arm", False),
        "embodiment_dis": 0.6,
        "camera": task_cfg["camera"]
    }

    robot = Robot(scene, need_topp=False, **kwargs)
    robot.init_joints()
    robot.move_to_homestate()

    for _ in range(50):
        scene.step()

    cam_system = Camera(bias=0, **kwargs)
    cam_system.load_camera(scene)
    cam_system.update_wrist_camera(robot.left_camera.get_pose(), robot.right_camera.get_pose())

    scene.step()
    scene.update_render()
    cam_system.update_picture()

    rgbs = cam_system.get_rgb()
    out_dir = "/home/armmy2530-fedora-pc/.gemini/antigravity-cli/brain/7739b1d3-b678-4a93-adf8-2dcd7f98c911"
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for cam_name, cam_data in rgbs.items():
        img = Image.fromarray(cam_data["rgb"])
        out_path = os.path.join(out_dir, f"{out_prefix}_{cam_name}.png")
        img.save(out_path)
        results[cam_name] = out_path
        print(f"[{out_prefix}] Saved {cam_name} -> {out_path} (shape: {cam_data['rgb'].shape})")
    
    return results

if __name__ == "__main__":
    print("=== Verifying YAM with ABC camera config ===")
    yam_res = render_embodiment("yam-abc", "abc_camera_config_yam", "yam_abc")
    
    print("\n=== Verifying ARX-X5 with ABC camera config ===")
    arx_res = render_embodiment("ARX-X5-abc", "abc_camera_config_arx5", "arx5_abc")
    
    print("\nVerification completed successfully!")
