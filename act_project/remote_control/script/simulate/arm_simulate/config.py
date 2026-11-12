"""全局常量与路径。"""

import os

WIDTH, HEIGHT, FPS = 640, 480, 30
URDF = (
    "/home/lab416/resource/act_project/clones/mycobot_ros"
    "/mycobot_description/urdf/mycobot_pi/mycobot_urdf.urdf"
)
MESH_DIR = os.path.dirname(URDF)
JOINT_LIMIT = 3.14

# Cube 初始状态
CUBE_INITIAL_POS = [0.18, -0.08, 0.012]  # 世界坐标，米
CUBE_INITIAL_SIZE_MM = 24.0  # 边长，毫米

# Arm 初始控制状态
ARM_INITIAL_ANGLES_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ARM_INITIAL_GRIPPER_MM = 0.0

# 网页 IK 默认值
IK_INITIAL_POS = [0.18, -0.08, 0.14]
IK_INITIAL_GRIPPER_MM = 10.0
IK_INITIAL_DOWN = True
