"""myCobot 280 Pi 的 MuJoCo 实时仿真画面（HTTP MJPEG 推流）。

页面：固定视角 / 第一人称（腕部相机）/ 第三人称（跟随夹爪）三个画面
     + 6 关节滑块 + 夹爪滑块。
实现位于 script/simulate/arm_simulate，此文件仅作为独立启动入口，
与 startboot 走同一套 ArmSimulateManager 启动逻辑。

用法（可在任意目录运行）：
    MUJOCO_GL=egl python3 remote_control/arm_viewer_mj.py --port 8080
"""

import argparse
import os
import signal
import sys

_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from function.LogicSystem import LogicSys
from simulate.arm_simulate.arm_manager import ArmSimulateManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="myCobot MuJoCo 实时画面")
    parser.add_argument("--port", type=int, default=8080, help="HTTP 端口")
    # 独立入口也注册进 LogicSys，/move 等服务定位才能拿到同一个实例
    LogicSys.register(ArmSimulateManager)
    manager = LogicSys.get(ArmSimulateManager)
    manager.start(parser.parse_args().port)
    try:
        signal.pause()  # 主线程挂起，等工作线程推流
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
