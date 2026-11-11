"""myCobot 280 Pi MuJoCo 实时仿真推流包。

模块划分：
- config : 常量与路径（分辨率、URDF 位置、关节限位）
- state  : 渲染线程与 HTTP 线程共享的运行时状态
- mjcf   : URDF → MJCF 模型生成（含夹爪与三视角相机）
- render : 仿真步进 + 三视角渲染线程
- server : HTTP 服务（MJPEG 推流、关节角下发）
- main   : 命令行入口
"""

import os

# 必须在 import mujoco 之前指定渲染后端；包内任何子模块都会先执行这里
os.environ.setdefault("MUJOCO_GL", "egl")
