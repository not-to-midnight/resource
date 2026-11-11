"""全局常量与路径。"""

import os

WIDTH, HEIGHT, FPS = 640, 480, 30
URDF = (
    "/home/lab416/resource/act_project/clones/mycobot_ros"
    "/mycobot_description/urdf/mycobot_pi/mycobot_urdf.urdf"
)
MESH_DIR = os.path.dirname(URDF)
JOINT_LIMIT = 3.14
