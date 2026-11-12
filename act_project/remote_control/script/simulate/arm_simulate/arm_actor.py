"""ArmActor：机械臂实体。

集中持有机械臂的全部可变状态——仿真模型、关节目标、夹爪开度、最新渲染帧。
由 ArmSimulateManager 创建并驱动；渲染线程与 HTTP 线程只通过它读写数据，
不再直接碰模块级全局变量。
"""

import math
import threading

from .config import (
    ARM_INITIAL_ANGLES_DEG,
    ARM_INITIAL_GRIPPER_MM,
    MESH_DIR,
    URDF,
)
from .mjcf import build_mjcf

JOINT_LIMIT_RAD = 2.9  # 关节角限幅（弧度）
GRIPPER_MAX_M = 0.015  # 夹爪最大开度（米）
CAMERAS = ("main", "wrist", "chase")


class ArmActor:
    """myCobot 280 Pi 的状态载体（不含渲染，不含业务编排）。"""

    def __init__(self) -> None:

        self._targets = [math.radians(v) for v in ARM_INITIAL_ANGLES_DEG]  # 关节目标角，弧度
        self._gripper = ARM_INITIAL_GRIPPER_MM / 1000.0  # 夹爪开度，米；0 = 闭合

        self._frames = {name: b"" for name in CAMERAS}  # 转换为二进制字节流
        self.model = None  # 由 ensure_sim 构建后持有
        self.data = None
        self.arm_qpos_adr = list(range(6))
        self.arm_dof_adr = list(range(6))

        self._frame_lock = threading.Lock()  # 保护渲染帧
        self._ctrl_lock = threading.Lock()  # 保护关节目标 / 夹爪开度
        self.sim_lock = threading.Lock()  # 保护 model / data 的步进与读取
        self._motion_lock = threading.Lock()

    # ---------- 仿真模型（actor 自己持有，render / IK 只读） ----------
    def ensure_sim(self):
        with self.sim_lock:
            if self.model is not None:
                return self.model, self.data

            import mujoco

            mjcf = build_mjcf(URDF, MESH_DIR)
            with open("/tmp/mycobot_mjcf.xml", "w") as f:
                f.write(mjcf)  # 留一份便于排查
            model = mujoco.MjModel.from_xml_string(mjcf)
            data = mujoco.MjData(model)

            # 前 6 个位置执行器对应 6 个臂关节，据此反查它们的 qpos/dof 索引
            joint_ids = [int(model.actuator_trnid[i, 0]) for i in range(6)]
            self.arm_qpos_adr = [
                int(model.jnt_qposadr[j]) for j in joint_ids
            ]  # 关节位置索引
            self.arm_dof_adr = [
                int(model.jnt_dofadr[j]) for j in joint_ids
            ]  # 关节速度索引

            self.model, self.data = model, data
            return model, data

    # ---------- 运动控制权（IK 与手动互斥） ----------
    def begin_motion(self) -> bool:
        return self._motion_lock.acquire(blocking=False)

    def end_motion(self) -> None:
        if self._motion_lock.locked():
            self._motion_lock.release()

    @property
    def motion_busy(self) -> bool:
        return self._motion_lock.locked()

    # ---------- 控制目标（manager / server 写，render 读） ----------
    def set_angles(self, *angles_deg, _from_ik: bool = False) -> None:
        if not _from_ik and self._motion_lock.locked():
            raise RuntimeError("IK 正在解算，已禁止手动控制")
        if len(angles_deg) != 6:
            raise ValueError(f"需要 6 个关节角，收到 {len(angles_deg)} 个")
        rads = [
            max(-JOINT_LIMIT_RAD, min(JOINT_LIMIT_RAD, math.radians(d)))
            for d in angles_deg
        ]
        with self._ctrl_lock:
            self._targets[:] = rads

    def set_gripper_mm(self, mm: float) -> None:
        with self._ctrl_lock:
            self._gripper = max(0.0, min(GRIPPER_MAX_M, mm / 1000.0))

    def reset(self) -> None:
        with self._ctrl_lock:
            self._targets[:] = [math.radians(v) for v in ARM_INITIAL_ANGLES_DEG]
            self._gripper = ARM_INITIAL_GRIPPER_MM / 1000.0

    # 用于获取当前状态
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

    # 获取机械臂的位置信息
    def get_arm_qpos(self) -> list:
        with self.sim_lock:
            return [float(self.data.qpos[a]) for a in self.arm_qpos_adr]
            
    # ---------- 渲染帧（render 写，server 读） ----------
    def set_frames(self, main: bytes, wrist: bytes, chase: bytes) -> None:
        with self._frame_lock:
            self._frames["main"] = main
            self._frames["wrist"] = wrist
            self._frames["chase"] = chase

    def get_frame(self, name: str = "main") -> bytes:
        name = name.lower()
        with self._frame_lock:
            return self._frames[name]
