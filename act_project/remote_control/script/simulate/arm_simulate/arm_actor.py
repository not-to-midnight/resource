"""ArmActor：机械臂实体。

集中持有机械臂的全部可变状态——关节目标、夹爪开度、最新渲染帧、停机信号。
由 ArmSimulateManager 创建并驱动；渲染线程与 HTTP 线程只通过它读写数据，
不再直接碰模块级全局变量。
"""

import math
import threading

JOINT_LIMIT_RAD = 2.9  # 关节角限幅（弧度）
GRIPPER_MAX_M = 0.015  # 夹爪最大开度（米）
CAMERAS = ("main", "wrist", "chase")


class ArmActor:
    """myCobot 280 Pi 的状态载体（不含渲染，不含业务编排）。"""

    def __init__(self) -> None:
        self._ctrl_lock = threading.Lock()  # 保护关节目标 / 夹爪开度
        self._targets = [0.0] * 6  # 关节目标角，弧度
        self._gripper = 0.0  # 夹爪开度，米；0 = 闭合

        self._frame_lock = threading.Lock()  # 保护渲染帧
        self._frames = {name: b"" for name in CAMERAS}

        self.stop_event = threading.Event()  # 渲染线程退出信号

    # ---------- 控制目标（manager / server 写，render 读） ----------
    def set_angles(self, *angles_deg) -> None:
        """下发 6 个关节角（度），超限自动夹紧。"""
        if len(angles_deg) != 6:
            raise ValueError(f"需要 6 个关节角，收到 {len(angles_deg)} 个")
        rads = [
            max(-JOINT_LIMIT_RAD, min(JOINT_LIMIT_RAD, math.radians(d)))
            for d in angles_deg
        ]
        with self._ctrl_lock:
            self._targets[:] = rads

    def set_gripper_mm(self, mm: float) -> None:
        """下发夹爪开度（毫米，0 ~ 15）。"""
        with self._ctrl_lock:
            self._gripper = max(0.0, min(GRIPPER_MAX_M, mm / 1000.0))

    #用于获取当前状态
    def get_state(self) -> dict:
        with self._ctrl_lock:
            targets = list(self._targets)
            grip = self._gripper
        return {
            "angles_deg": [round(math.degrees(a), 1) for a in targets],
            "angles_rad": targets,
            "gripper_m": grip,
            "gripper_mm": round(grip * 1000.0, 1),
        }

    def reset(self) -> None:
        """回到零位（停止仿真时调用）。"""
        with self._ctrl_lock:
            self._targets[:] = [0.0] * 6
            self._gripper = 0.0

    # ---------- 渲染帧（render 写，server 读） ----------
    def set_frames(self, main: bytes, wrist: bytes, chase: bytes) -> None:
        with self._frame_lock:
            self._frames["main"] = main
            self._frames["wrist"] = wrist
            self._frames["chase"] = chase

    def get_frame(self, name: str = "main") -> bytes:
        name = name.lower()
        if name not in self._frames:
            raise ValueError(f"未知相机 {name}，可选 {'/'.join(CAMERAS)}")
        with self._frame_lock:
            return self._frames[name]
