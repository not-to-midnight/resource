from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class CubeData:
    """方块数据。字段规范：

    id      : 非空字符串，唯一标识一个方块
    pos     : [x, y, z]，单位米，世界坐标系
    rot     : None = 保持当前姿态；3x3 旋转矩阵；或四元数 (w,x,y,z)（自动转矩阵）
    size_mm : 边长，毫米
    grasped : 是否被夹爪抓住
    """

    id: int
    pos: List[float]
    rot: List[float] = None
    size_mm: float = 24.0
    grasped: bool = False


class CubeActor:
    """方块智能体。字段规范：

    id      : 非空字符串，唯一标识一个方块智能体
    """

    def __init__(self, config: CubeData):
        self.data = config


    def get_id(self) -> str:
        return self.data.id

    def set_pose(self, position: list[float]):
        self.data.pos = position

    def set_rotation(self, rotation: List[float]):
        self.data.rot = rotation