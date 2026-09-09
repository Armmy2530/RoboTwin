"""Spawn the OpenArm robot using the RoboTwin pipeline's own loading code.

This replicates the exact same flow as collect_data.py → Base_Task → Robot,
so you can visually confirm the robot loads and looks correct.

Run from the RoboTwin root:
    python script/visual_check_openarm.py
"""

import sys
sys.path.append("./")

import os
import yaml
import sapien.core as sapien
from sapien.render import set_global_config
from sapien.utils.viewer import Viewer
import numpy as np

from envs.robot import Robot
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

# ── Load config (same as collect_data.py) ──────────────────────────────
embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
with open(embodiment_config_path, "r") as f:
    _embodiment_types = yaml.safe_load(f)

robot_file = _embodiment_types["openarm-v1"]["file_path"]
# robot_file = _embodiment_types["aloha-agilex"]["file_path"]
robot_config_file = os.path.join(robot_file, "config.yml")
with open(robot_config_file, "r") as f:
    embodiment_args = yaml.safe_load(f)

# ── Create SAPIEN scene (same as Base_Task.setup_scene) ────────────────
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

# ── Add a simple table ─────────────────────────────────────────────────
builder = scene.create_actor_builder()
builder.add_box_visual(sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025],
                       material=[0.8, 0.8, 0.8])
builder.add_box_collision(sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025])
builder.build_static("table")

# ── Load robot (same as Base_Task.load_robot) ──────────────────────────
kwargs = {
    "left_embodiment_config": embodiment_args,
    "right_embodiment_config": embodiment_args,
    "left_robot_file": robot_file,
    "right_robot_file": robot_file,
    "dual_arm_embodied": embodiment_args.get("dual_arm", False),
}

robot = Robot(scene, need_topp=False, **kwargs)
robot.init_joints()
robot.move_to_homestate()

# ── Step physics to settle ─────────────────────────────────────────────
for _ in range(100):
    scene.step()

# ── Print diagnostics ─────────────────────────────────────────────────
print("\n=== Robot Loaded via RoboTwin Pipeline ===")
print(f"  Left entity root pose:  p={robot.left_entity.get_root_pose().p}  q={robot.left_entity.get_root_pose().q}")
print(f"  Right entity root pose: p={robot.right_entity.get_root_pose().p}  q={robot.right_entity.get_root_pose().q}")

left_ee_pose = robot.left_ee.global_pose if robot.left_ee else None
right_ee_pose = robot.right_ee.global_pose if robot.right_ee else None
print(f"  Left EE global_pose:  p={left_ee_pose.p if left_ee_pose else 'None'}  q={left_ee_pose.q if left_ee_pose else 'None'}")
print(f"  Right EE global_pose: p={right_ee_pose.p if right_ee_pose else 'None'}  q={right_ee_pose.q if right_ee_pose else 'None'}")

has_nan = (left_ee_pose is not None and np.any(np.isnan(left_ee_pose.p))) or \
          (right_ee_pose is not None and np.any(np.isnan(right_ee_pose.p)))
if has_nan:
    print("\n  ❌ NaN detected in ee_pose!")
else:
    print("\n  ✅ ee_pose values are valid (no NaN)")

left_pose = robot.get_left_ee_pose()
right_pose = robot.get_right_ee_pose()
print(f"\n  get_left_ee_pose():  {left_pose}")
print(f"  get_right_ee_pose(): {right_pose}")

print(f"\n  Active joints: {[j.get_name() for j in robot.left_entity.get_active_joints()]}")
print(f"  All links:     {[l.get_name() for l in robot.left_entity.get_links()]}")

# ── Open viewer ────────────────────────────────────────────────────────
viewer = Viewer(renderer)
viewer.set_scene(scene)
viewer.set_camera_xyz(x=0.4, y=0.22, z=1.5)
viewer.set_camera_rpy(r=0, p=-0.8, y=2.45)

print("\nViewer opened. Close the window to exit.")
while not viewer.closed:
    scene.step()
    scene.update_render()
    viewer.render()
