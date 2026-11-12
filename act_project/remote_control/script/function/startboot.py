import os
import sys

# 只把 script/ 放入 sys.path：LogicSystem 统一以 function.LogicSystem 导入，
# 避免同一文件被当成两个模块（LogicSystem / function.LogicSystem）而产生两个单例。
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from function.LogicSystem import LogicSystem, LogicSys


class StartBoot:
    def __init__(self):
        self.ctl: LogicSystem = LogicSys

    def start(self) -> None:
        print("[StartBoot] 初始化系统...")
        self.ctl.init()  # 读 default_services 配置，批量 register + init
        print("[StartBoot] 当前状态:", self.ctl.status())

    def stop(self) -> None:
        print("[StartBoot] 卸载全部服务...")
        self.ctl.unload()
        print("[StartBoot] 已卸载")

    # --- 预留钩子：服务器返回数据时再调 ---
    def load_post(self, *args, **kwargs):
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
