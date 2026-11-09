import pybullet as p
import pybullet_data
import os
import sys
import math
import imageio

print(f"当前解释器: {sys.executable}")

# 服务器没有显示器，用 DIRECT 离屏渲染，再录成 mp4
physicsClient = p.connect(p.DIRECT)
if physicsClient < 0:
    print("❌ 无法连接 PyBullet 物理引擎")
    sys.exit(1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 加载地面
planeId = p.loadURDF("plane.urdf")

robot_urdf = "/home/lab416/resource/act_project/clones/mycobot_ros/mycobot_description/urdf/mycobot_pi/mycobot_urdf.urdf"
p.setAdditionalSearchPath("/home/lab416/resource/act_project/clones/mycobot_ros")

print(f"🔍 正在尝试加载: {robot_urdf}")

# 相机参数：斜上方看向机械臂
WIDTH, HEIGHT = 640, 480
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0, 0, 0.15],
    distance=0.7,
    yaw=45,
    pitch=-30,
    roll=0,
    upAxisIndex=2,
)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=WIDTH / HEIGHT, nearVal=0.01, farVal=3
)

try:
    if not os.path.exists(robot_urdf):
        print(f"❌ 找不到文件，请检查路径。")
    else:
        robot_id = p.loadURDF(robot_urdf, basePosition=[0, 0, 0], useFixedBase=True)
        print(f"✅ 机械臂加载成功！")

        num_joints = p.getNumJoints(robot_id)
        print(f"✅ 机械臂共有 {num_joints} 个关节。")

        # 打印关节名（方便你确认哪个是夹爪）
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            print(f"   - 关节 {i}: {joint_name}")

        print("🎬 开始让机械臂运动并录制视频...")

        frames = []
        for i in range(1200):
            # 让关节 1 来回转动
            target_angle = 1.0 * math.sin(i * 0.02)
            p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=target_angle)
            p.stepSimulation()

            # 每 4 步取一帧，最终约 30fps
            if i % 4 == 0:
                img = p.getCameraImage(
                    WIDTH, HEIGHT, view_matrix, proj_matrix,
                    renderer=p.ER_TINY_RENDERER,
                )
                frames.append(img[2][:, :, :3])  # RGB

        out_path = "/home/lab416/resource/act_project/remote_control/mycobot_sim.mp4"
        imageio.mimsave(out_path, frames, fps=30, quality=8)
        print(f"✅ 仿真结束，视频已保存: {out_path}")

except Exception as e:
    print("❌ 加载失败，报错信息：", e)

p.disconnect()
