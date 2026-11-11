"""ArmSimulateManager：myCobot 280 Pi 仿真业务入口。

职责：创建并持有 ArmActor，启动渲染线程与 HTTP 服务，对外提供控制接口。
机械臂的全部状态都在 ArmActor 里，本类只做编排。
"""

import os
import sys
import threading
import time

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
        self._port = 8080  # HTTP 服务端口

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

        actor.stop_event.clear()
        self._render_thread = threading.Thread(
            target=render.render_loop, args=(actor,), name="arm-render", daemon=True
        )
        self._render_thread.start()
        time.sleep(1.0)  # 等 MuJoCo 模型加载出第一帧

        from http.server import ThreadingHTTPServer

        # 把实体注入 HTTP 层：每个请求处理器都通过同一个 actor 读写
        server.StreamHandler.actor = actor
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
        actor.stop_event.set()  # 让渲染线程退出循环
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
        return self.actor.get_state()

    def capture_camera(self, name: str = "main") -> bytes:
        return self.actor.get_frame(name)

    def move_to_pose(self, pos, rot, gripper_mm: float = None) -> dict:
        """给定末端目标位姿 (pos[3], rot[3,3])，解 IK 并下发。

        这是高层入口——真机控制和仿真渲染都可以复用同一运动算法。
        """
        # TODO: 等 render_loop 把 model/data 暴露出来后补上
        # q0 从 data.qpos 读，forward/jacobian 用 mujoco API 封装
        # q_solved, ok, _ = _motion_ik(forward, jacobian, q0, pos, rot)
        # if ok: self.set_angles(*[math.degrees(q) for q in q_solved[:6]])
        raise NotImplementedError("IK 入口待接入——需要 render_loop 暴露 model/data")
