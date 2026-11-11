"""CubeSimulateManager：被抓取方块的业务入口。

实现 BaseService，职责：
- 维护一个或多个方块的状态（CubeData：位置、姿态、是否被夹住）
- 传入的 cube 数据必须是 CubeData 类型（isinstance 强制检查）

等 ACT 模型上线后，识别到方块 → 构造 CubeData → 调 set_pose → 广播给视觉/仿真。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from function.LogicSystem import BaseService
from simulate.cube_simulate.cube_actor import CubeData, CubeActor


class CubeSimulateManager(BaseService):
    def __init__(self):
        self._cubes: dict[int, CubeActor] = {}  # cube_id -> CubeActor
        self._idCount: int = 0

    # ---------- BaseService ----------
    def init(self, config: CubeData = None):
        if config is None:
            self.create_actor(CubeData(id=0, pos=[0.0, 0.0, 0.0]))
        else:
            self.create_actor(config)

    def unload(self):
        self._cubes.clear()

    def get_cube_byID(self, cube_id: int) -> CubeActor:
        if cube_id not in self._cubes:
            raise KeyError(cube_id)
        return self._cubes[cube_id]

    def list_cubes(self) -> dict[int, CubeActor]:
        return self._cubes

    def create_actor(self, config: CubeData) -> None:
        config.id = self._idCount
        self._cubes[self._idCount] = CubeActor(config)
        self._idCount += 1
