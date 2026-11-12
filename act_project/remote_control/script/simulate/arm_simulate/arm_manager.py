"""ArmSimulateManager：myCobot 280 Pi 仿真业务入口。

职责：创建并持有 ArmActor，启动渲染线程与 HTTP 服务，对外提供控制接口。
机械臂的全部状态都在 ArmActor 里，本类只做编排。
"""

import math
import os
import sys
import threading
import time

import numpy as np

# 把 script/ 加进 sys.path，这样 simulate/ 和 function/ 都可见
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from function.LogicSystem import BaseService

# 延迟 import：避免 init 前就触发 MUJOCO_GL 设置
_render_module = None
_server_module = None
_actor_module = None
_motion_ik = None


def _lazy_modules():
    global _render_module, _server_module, _actor_module, _motion_ik
    if _render_module is None:
        from simulate.arm_simulate import arm_actor as _actor_module
        from simulate.arm_simulate import render as _render_module
        from simulate.arm_simulate import server as _server_module
        from simulate.motion.ik import dls_ik as _motion_ik
    return _render_module, _server_module, _actor_module, _motion_ik


class ArmSimulateManager(BaseService):
    """myCobot 280 Pi 仿真业务管理器（由 LogicSys 驱动生命周期）。"""

    def __init__(self):
        self._actor = None  # ArmActor：机械臂实体
        self._render_thread = None  # 渲染线程
        self._httpd = None  # HTTP 服务对象
        self._httpd_thread = None  # HTTP 服务线程
        self._running = False  # 是否正在运行
        self.stop_event = threading.Event()  # 渲染与轨迹线程退出信号
        self._port = 8080  # HTTP 服务端口
        self._traj_thread = None  # 轨迹插值线程
        self._traj_cancel = None  # 轨迹取消信号

    # ---------- 实体 ----------
    @property
    def actor(self):
        """机械臂实体；未启动时按需创建，便于提前下发指令。"""
        if self._actor is None:
            _, _, actor_module, _ = _lazy_modules()
            self._actor = actor_module.ArmActor()
        return self._actor

    # ---------- BaseService 生命周期 ----------
    def init(self, *args, **kwargs):
        self.start()

    def unload(self):
        self.stop()

    # ---------- 手动启停（调试，幂等） ----------
    def start(self, port: int = None) -> None:
        if self._running:
            return
        render, server, _, _ = _lazy_modules()
        self._port = port or self._port
        actor = self.actor

        # 只接入已有 CubeActor，不改变 CubeManager / CubeActor 的业务逻辑
        from function.LogicSystem import LogicSys
        from simulate.cube_simulate.cube_simulate_manager import CubeSimulateManager

        cube_actor = None
        if LogicSys.has(CubeSimulateManager):
            cube_actor = LogicSys.get(CubeSimulateManager).list_cubes().get(0)

        self.stop_event.clear()
        actor.ensure_sim()  # 同步建模型：失败会直接抛给 LogicSys，不会静默黑屏
        self._render_thread = threading.Thread(
            target=render.render_loop,
            args=(actor, self.stop_event, cube_actor),
            name="arm-render",
            daemon=True,
        )
        self._render_thread.start()
        time.sleep(1.0)  # 等渲染线程建好 renderer 并出新帧

        from http.server import ThreadingHTTPServer

        # 把实体注入 HTTP 层：每个请求处理器都通过同一个 actor 读写
        server.StreamHandler.actor = actor
        # IK 走 LogicSys.get(ArmSimulateManager) 获取，不做方法注入
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self._port), server.StreamHandler)
        # serve_forever 是阻塞调用，必须放独立线程；主流程只负责启动和登记
        self._httpd_thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="arm-http",
            daemon=True,
        )
        self._httpd_thread.start()
        self._running = True
        print(f"[ArmSimulate] 已启动 http://127.0.0.1:{self._port}")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        actor = self.actor
        if self._traj_cancel is not None:
            self._traj_cancel.set()  # 中断进行中的轨迹，释放运动控制权
        if self._traj_thread is not None:
            self._traj_thread.join(timeout=1.0)
            self._traj_thread = None
        self.stop_event.set()  # 让渲染与轨迹线程退出
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None
        if self._httpd is not None:
            self._httpd.shutdown()  # 停 serve_forever
            self._httpd.server_close()  # 释放端口
            self._httpd = None
        if self._httpd_thread is not None:
            self._httpd_thread.join(timeout=2.0)
            self._httpd_thread = None
        actor.reset()
        print("[ArmSimulate] 已停止")

    # ---------- 业务 API ----------
    def set_angles(self, *angles_deg) -> None:
        self.actor.set_angles(*angles_deg)

    def set_gripper(self, mm: float) -> None:
        self.actor.set_gripper_mm(mm)

    def get_state(self) -> dict:
        state = self.actor.get_state()
        state["motion_busy"] = self.actor.motion_busy
        return state

    def reset(self) -> None:
        self.set_angles(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.set_gripper(0.0)

        from function.LogicSystem import LogicSys
        from simulate.cube_simulate.cube_simulate_manager import CubeSimulateManager

        if LogicSys.has(CubeSimulateManager):
            LogicSys.get(CubeSimulateManager).reset()

    def move_to_pose(self, pos, rot=None, gripper_mm: float = None, max_iter: int = 200) -> dict:
        actor = self.actor
        if not actor.begin_motion():
            return {
                "ok": False,
                "busy": True,
                "reason": "已有运动在执行，禁止并发 IK / 手动控制",
            }
        try:
            result = self._solve_to_pose(actor, pos, rot, gripper_mm, max_iter)
        except Exception:
            actor.end_motion()
            raise
        if not result.get("ok"):
            actor.end_motion()
            return result
        # 解算成功：不直接跳到终点角，交给轨迹线程平滑插值过去，
        # 运动控制权由轨迹线程在结束时释放（期间手动/并发 IK 仍被禁止）
        q_start = result.pop("_q_start")
        q_goal = result.pop("_q_goal")
        result["duration_s"] = self._start_trajectory(actor, q_start, q_goal)
        return result

    def _start_trajectory(self, actor, q_start, q_goal) -> float:
        """从当前关节角平滑插值到 IK 目标角，返回预计用时（秒）。

        smoothstep 速度曲线（两端速度为 0），单关节限速 60°/s，30Hz 下发。
        """
        delta = np.abs(q_goal - q_start)
        duration = min(6.0, max(0.8, float(delta.max()) / math.radians(60)))
        self._traj_cancel = threading.Event()
        cancel = self._traj_cancel

        def run():
            try:
                steps = max(1, int(duration * 30))
                for i in range(1, steps + 1):
                    if cancel.is_set() or self.stop_event.is_set():
                        return
                    t = i / steps
                    s = t * t * (3 - 2 * t)  # smoothstep，起止无冲击
                    q = q_start + (q_goal - q_start) * s
                    actor.set_angles(*[math.degrees(v) for v in q], _from_ik=True)
                    time.sleep(duration / steps)
            finally:
                actor.end_motion()

        self._traj_thread = threading.Thread(target=run, name="arm-traj", daemon=True)
        self._traj_thread.start()
        return duration

    def _solve_to_pose(self, actor, pos, rot, gripper_mm, max_iter) -> dict:
        _, _, actor_module, dls_ik = _lazy_modules()
        import mujoco  # MUJOCO_GL 已在包初始化时设好，这里再导入才安全

        model, _ = actor.ensure_sim()  # 未启动仿真也能单独解算
        ik_data = mujoco.MjData(model)  # 本次 IK 的独立数据，不干扰渲染线程
        qpos_adr = actor.arm_qpos_adr  # 臂关节在 qpos / qvel 中的实际索引
        dof_adr = actor.arm_dof_adr
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        if site_id < 0:
            raise RuntimeError("模型里找不到 tcp site")

        def fk(q):
            """正解：关节角(弧度) → 末端位置[3] 与姿态[3,3]。"""
            ik_data.qpos[qpos_adr] = q
            mujoco.mj_forward(model, ik_data)
            return (
                ik_data.site_xpos[site_id].copy(),
                ik_data.site_xmat[site_id].reshape(3, 3).copy(),
            )

        def jac(q):
            """雅可比：(6, 6)，前 3 行位置、后 3 行姿态。"""
            ik_data.qpos[qpos_adr] = q
            mujoco.mj_forward(model, ik_data)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
            return np.vstack([jacp, jacr])[:, dof_adr]

        target_pos = np.asarray(pos, dtype=float).reshape(-1)
        if target_pos.shape != (3,):
            raise ValueError(f"pos 需要 3 个分量，收到 {target_pos.size} 个")

        q0 = np.array(actor.get_arm_qpos(), dtype=float)
        target_rot = fk(q0)[1] if rot is None else np.asarray(rot, dtype=float).reshape(3, 3)

        # 多起点求解：从零位出发时机械臂完全伸直（奇异位形），单一起点容易解到
        # 肘部翻转 180° 的反向姿态，所以额外试两个自然弯曲的初值，取最优解。
        seeds = [
            q0,
            np.array([0.0, -0.5, 1.0, 0.0, 0.5, 0.0]),
            np.array([0.0, 0.5, -1.0, 0.0, 0.5, 0.0]),
        ]
        best = None
        for seed in seeds:
            q_try, ok_try, it_try = dls_ik(
                fk,
                jac,
                seed,
                target_pos,
                target_rot,
                max_iter=max_iter,
                q_min=-actor_module.JOINT_LIMIT_RAD,
                q_max=actor_module.JOINT_LIMIT_RAD,
            )
            err = float(np.linalg.norm(target_pos - fk(q_try)[0]))
            # 先比是否收敛，再比位置误差，最后偏好关节角小的自然姿态
            score = (0 if ok_try else 1, round(err, 6), float(np.linalg.norm(q_try)))
            if best is None or score < best[0]:
                best = (score, q_try, ok_try, it_try)

        _, q, ok, iters = best

        if ok and gripper_mm is not None:
            self.set_gripper(gripper_mm)  # 夹爪不参与轨迹，直接到位

        result = {
            "ok": ok,
            "iters": iters,
            "q_deg": [round(math.degrees(v), 1) for v in q[:6]],
            "pos_err_m": float(np.linalg.norm(target_pos - fk(q)[0])),
        }
        if ok:
            # 轨迹端点（内部键，move_to_pose 里弹出，不进 HTTP 响应）
            result["_q_start"] = q0.copy()
            result["_q_goal"] = q[:6].copy()
        return result
