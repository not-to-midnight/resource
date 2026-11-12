"""仿真步进 + 三视角渲染线程：持续把 JPEG 帧写入 ArmActor。"""

import time

import cv2
import mujoco

from .arm_actor import ArmActor
from .config import FPS, HEIGHT, WIDTH


def render_loop(actor: ArmActor, stop_event, cube_actor=None) -> None:
    try:
        model, data = actor.ensure_sim()  # 模型归 actor 所有，这里只取用
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        wrist_renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        chase_renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    except Exception as e:
        # 初始化失败不能静默——线程直接死会让三路推流全黑且无任何报错
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

    # 接入 CubeManager 中已有的 CubeData：初始化时读取当前仿真状态，之后只同步控制修改
    cube_qpos_adr = cube_dof_adr = cube_geom_id = None
    last_cube_pos = last_cube_size = None
    if cube_actor is not None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        joint_id = model.body_jntadr[body_id]
        cube_qpos_adr = int(model.jnt_qposadr[joint_id])
        cube_dof_adr = int(model.jnt_dofadr[joint_id])
        cube_geom_id = int(model.body_geomadr[body_id])
        cube_actor.data.pos = [
            float(v) for v in data.qpos[cube_qpos_adr:cube_qpos_adr + 3]
        ]
        cube_actor.data.size_mm = float(model.geom_size[cube_geom_id, 0] * 2000)
        last_cube_pos = tuple(cube_actor.data.pos)
        last_cube_size = cube_actor.data.size_mm

    while not stop_event.is_set():
        state = actor.get_state()
        data.ctrl[:6] = state["angles_rad"]
        data.ctrl[6:] = state["gripper_m"]
        with actor.sim_lock:
            if cube_actor is not None:
                pos = tuple(float(v) for v in cube_actor.data.pos)
                size_mm = float(cube_actor.data.size_mm)
                if pos != last_cube_pos:
                    data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = pos
                    data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0
                    last_cube_pos = pos
                if size_mm != last_cube_size:
                    model.geom_size[cube_geom_id, :] = size_mm / 2000.0
                    last_cube_size = size_mm
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
