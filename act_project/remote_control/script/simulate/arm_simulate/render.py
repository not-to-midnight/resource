"""仿真步进 + 三视角渲染线程：持续把 JPEG 帧写入 ArmActor。"""

import time

import cv2
import mujoco

from .arm_actor import ArmActor
from .config import FPS, HEIGHT, MESH_DIR, URDF, WIDTH
from .mjcf import build_mjcf


def render_loop(actor: ArmActor) -> None:
    try:
        mjcf = build_mjcf(URDF, MESH_DIR)
        with open("/tmp/mycobot_mjcf.xml", "w") as f:
            f.write(mjcf)  # 留一份便于排查
        model = mujoco.MjModel.from_xml_string(mjcf)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        wrist_renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        chase_renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    except Exception as e:
        # 模型加载失败不能静默——线程直接死会让三路推流全黑且无任何报错
        import traceback
        print(f"[render] 模型初始化失败: {e}")
        traceback.print_exc()
        return

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance = 1.15
    cam.azimuth = 135
    cam.elevation = -22

    substeps = max(1, round((1.0 / FPS) / model.opt.timestep))

    while not actor.stop_event.is_set():
        state = actor.get_state()
        data.ctrl[:6] = state["angles_rad"]
        data.ctrl[6:] = state["gripper_m"]
        for _ in range(substeps):
            mujoco.mj_step(model, data)  # 仿真步进/物理模拟计算

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpg_bytes = buffer.tobytes()

        wrist_renderer.update_scene(data, camera="wrist_cam")
        rgb = wrist_renderer.render()
        bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        wrist_jpg_bytes = buffer.tobytes()

        chase_renderer.update_scene(data, camera="chase_cam")
        rgb = chase_renderer.render()
        bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        chase_jpg_bytes = buffer.tobytes()

        actor.set_frames(jpg_bytes, wrist_jpg_bytes, chase_jpg_bytes)

        time.sleep(1.0 / FPS)
