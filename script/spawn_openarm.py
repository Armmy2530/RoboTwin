"""Spawn the OpenArm embodiment in a SAPIEN scene with a viewer.

No cuRobo / mplib / manipulation pipeline is involved - this only loads the
URDF into SAPIEN, steps the simulation and renders.

Run from inside the `robotwin_ws` pixi shell:

    uv run python script/spawn_openarm.py
"""

import os
import re
import sys
import tempfile

import sapien.core as sapien
from sapien.render import set_global_config
from sapien.utils.viewer import Viewer

import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# URDF_PATH = os.path.join(ROOT, "assets/embodiments/openarm-v1/urdf/openarm_v10.urdf")
# SRDF_PATH = os.path.join(ROOT, "assets/embodiments/openarm-v1/srdf/openarm_v10.srdf")
URDF_PATH = os.path.join(ROOT, "assets/embodiments/yam/v1/yam_linear_4310_d405_correctColor.urdf")
SRDF_PATH = os.path.join(ROOT, "assets/embodiments/yam/v1/yam_d405.srdf")
# URDF_PATH = os.path.join(ROOT, "assets/embodiments/franka-panda/panda.urdf")
# SRDF_PATH = os.path.join(ROOT, "assets/embodiments/franka-panda/panda.srdf")

def load_openarm(scene, urdf_path=URDF_PATH, srdf_path=SRDF_PATH):
    """Load the OpenArm URDF.

    SAPIEN 3.0.0b1 segfaults in PhysX when a collision mesh carries a negative
    scale (the openarm URDF mirrors the left arm via `scale="0.001 -0.001
    0.001"`).  As a workaround, a copy of the URDF with all collision-mesh
    scales made positive is loaded on the fly.  Visual meshes keep their
    original scale so the robot looks correct.
    """
    # tree = ET.parse(urdf_path)
    # root = tree.getroot()

    # for collision in root.iter("collision"):
    #     mesh = collision.find("geometry/mesh")
    #     if mesh is None or mesh.get("scale") is None:
    #         continue
    #     vals = [abs(float(v)) for v in mesh.get("scale").replace(",", " ").split()]
    #     mesh.set("scale", " ".join(f"{v:g}" for v in vals))

    # fd, tmp_path = tempfile.mkstemp(suffix=".urdf", dir=os.path.dirname(urdf_path))
    # os.close(fd)

    try:
        # tree.write(tmp_path, xml_declaration=True, encoding="utf-8")
        loader = scene.create_urdf_loader()
        # loader.fix_root_link = True
        # return loader.load(tmp_path, srdf_file=srdf_path)
        return loader.load(urdf_path, srdf_file=srdf_path)
    finally:
      pass
      # os.remove(tmp_path)


def create_table(scene):
    builder = scene.create_actor_builder()
    builder.add_box_visual(
        sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025], material=[0.8, 0.8, 0.8]
    )
    builder.add_box_collision(
        sapien.Pose([0, 0, 0.74]), half_size=[0.6, 0.35, 0.025]
    )
    builder.build_static("table")


def main():
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene(sapien.SceneConfig())
    scene.set_timestep(1 / 250)
    # scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5])
    scene.add_point_light([1, 0, 1.8], [1, 1, 1])
    scene.add_point_light([-1, 0, 1.8], [1, 1, 1])

    # create_table(scene)

    robot = load_openarm(scene)
    print("OpenArm loaded")
    print("  active joints:", [j.get_name() for j in robot.get_active_joints()])
    print("  links:", [l.get_name() for l in robot.get_links()])

    if "--headless" not in sys.argv:
        print("mambo")
        viewer = Viewer(renderer)
        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=0.4, y=0.22, z=1.5)
        viewer.set_camera_rpy(r=0, p=-0.8, y=2.45)

        while not viewer.closed:
            scene.update_render()
            scene.step()
            viewer.render()
    else:
        for _ in range(2000):
            scene.step()
        print("headless step OK")


if __name__ == "__main__":
    main()
