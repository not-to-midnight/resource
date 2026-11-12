

import threading
from abc import ABC, abstractmethod
from typing import List, Type


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------
class BaseService(ABC):
    """所有 manager / service 必须实现的接口。"""

    @abstractmethod
    def init(self, *args, **kwargs):
        """初始化加载。args/kwargs 可按具体场景传入（如服务器返回的 role 信息）。"""

    @abstractmethod
    def unload(self):
        """卸载释放资源（停止线程、关闭连接等）。"""


# ---------------------------------------------------------------------------
# 单例主体
# ---------------------------------------------------------------------------
class LogicSystem:

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._services = {}    # Type[BaseService] -> BaseService
                inst._inited = {}      # Type[BaseService] -> bool
                inst._lock = threading.RLock()
                inst._booted = False
                cls._instance = inst
        return cls._instance

    def init(self, services_to_register: List[Type[BaseService]] = None) -> None:
        with self._lock:
            if self._booted:
                self._clear_all()

            classes = services_to_register if services_to_register is not None else default_services()
            for cls in classes:
                self.register(cls)

            for cls, svc in list(self._services.items()):
                try:
                    svc.init()
                    self._inited[cls] = True
                except Exception as e:
                    print(f"[LogicSys] init {cls.__name__} 失败: {e}")
            self._booted = True

    def register(self, cls: Type[BaseService]) -> "LogicSystem":
        with self._lock:
            if cls in self._services and self._inited.get(cls):
                raise RuntimeError(f"{cls.__name__} 已在运行")
            instance = cls()
            self._services[cls] = instance
            return self  # 链式调用

    def get(self, cls: Type[BaseService]) -> BaseService:
        with self._lock:
            if cls not in self._services:
                raise KeyError(f"{cls.__name__} 未注册")
            return self._services[cls]

    def has(self, cls: Type[BaseService]) -> bool:
        with self._lock:
            return cls in self._services

    def unload(self, cls: Type[BaseService] = None) -> None:
        with self._lock:
            targets = [cls] if cls is not None else list(self._services)
            for c in targets:
                svc = self._services.get(c)
                if svc is None:
                    continue
                try:
                    svc.unload()
                except Exception as e:
                    print(f"[LogicSys] unload {c.__name__} 异常: {e}")
                self._inited.pop(c, None)
            if cls is None:
                self._booted = False

    def status(self) -> dict:
        with self._lock:
            return {c.__name__: self._inited.get(c, False) for c in self._services}


    # ---------- 内部 ----------
    def _clear_all(self) -> None:
        for svc in list(self._services.values()):
            try:
                svc.unload()
            except Exception:
                pass
        self._services.clear()
        self._inited.clear()
        self._booted = False



def default_services() -> List[Type[BaseService]]:
    from simulate.arm_simulate.arm_manager import ArmSimulateManager
    from simulate.cube_simulate.cube_simulate_manager import CubeSimulateManager

    services: List[Type[BaseService]] = [
        CubeSimulateManager,
        ArmSimulateManager,
    ]
    return services


LogicSys = LogicSystem()
