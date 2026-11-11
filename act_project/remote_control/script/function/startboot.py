"""StartBoot：系统启动入口。

职责：
  1. 读 default_services（相当于 config）——声明有哪些情景/服务
  2. LogicSys.init() 批量注册 + 依次初始化
  3. （可选）等服务器返回玩家/关卡数据后，调 LogicSys.loadPost(...) 等钩子
  4. 退出时 LogicSys.unload() 统一清理

本文件只做最小骨架，实际 import 在 run() 里按环境延迟加载。
"""

import os
import sys

# 让 startboot 在任意 cwd 下都能找到 function/ 和 simulate/ 包
_FUNCTION_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPT_DIR = os.path.dirname(_FUNCTION_DIR)
for _p in (_SCRIPT_DIR, _FUNCTION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from LogicSystem import LogicSystem, LogicSys


class StartBoot:
    def __init__(self):
        self.ctl: LogicSystem = LogicSys

    def start(self) -> None:
        """启动全部服务。"""
        print("[StartBoot] 初始化系统...")
        self.ctl.init()  # 读 default_services 配置，批量 register + init
        print("[StartBoot] 当前状态:", self.ctl.status())

    def stop(self) -> None:
        """停止全部服务。"""
        print("[StartBoot] 卸载全部服务...")
        self.ctl.unload()
        print("[StartBoot] 已卸载")

    # --- 预留钩子：服务器返回数据时再调 ---
    def load_post(self, *args, **kwargs):
        """加载关卡/帖子数据后触发。"""
        self.ctl.load_post(*args, **kwargs)


if __name__ == "__main__":
    boot = StartBoot()
    try:
        boot.start()
        # 等待 Ctrl+C
        import signal
        signal.pause()
    except KeyboardInterrupt:
        boot.stop()
