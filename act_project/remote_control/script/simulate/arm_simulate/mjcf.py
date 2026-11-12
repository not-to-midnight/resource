"""URDF → MJCF 模型生成：本体 + 平行夹爪 + 三视角相机 + 位置执行器。"""

import math
import os
import xml.etree.ElementTree as ET

from .config import CUBE_INITIAL_POS, CUBE_INITIAL_SIZE_MM, JOINT_LIMIT


def rpy_to_quat(roll: float, pitch: float, yaw: float):
    """URDF 的固定轴 rpy（Rz*Ry*Rx）转四元数 (w, x, y, z)。"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _parse_urdf(urdf_path: str, mesh_dir: str):
    """提取 visual 网格与关节树，返回 (links, joints, root_link)。"""
    root = ET.parse(urdf_path).getroot()

    links = {}
    for link in root.findall("link"):
        vis = link.find("visual")
        mesh = vis.find("geometry/mesh") if vis is not None else None
        if mesh is None:
            continue
        origin = vis.find("origin")
        xyz = [float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy") if origin is not None else "0 0 0").split()]
        name = mesh.get("filename").rsplit("/", 1)[-1]
        name = os.path.splitext(name)[0] + ".obj"  # MuJoCo 不支持 dae
        links[link.get("name")] = (os.path.join(mesh_dir, name), xyz, rpy)

    joints, children = [], set()
    for j in root.findall("joint"):
        child = j.find("child").get("link")
        o = j.find("origin")
        joints.append(
            dict(
                name=j.get("name"),
                parent=j.find("parent").get("link"),
                child=child,
                xyz=[float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()],
                rpy=[float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()],
                axis=[float(v) for v in j.find("axis").get("xyz").split()],
            )
        )
        children.add(child)
    root_link = next(link for link in links if link not in children)
    return links, joints, root_link


def _emit_gripper(out: list, pad: str) -> None:
    """法兰末端挂一个平行两指夹爪 + 侧装腕部相机（eye-in-hand）。"""
    out.append(f'{pad}<body name="gripper_base" pos="0 0 0.012">')
    out.append(f'{pad}  <inertial pos="0 0 0" mass="0.06" diaginertia="1e-5 1e-5 1e-5"/>')
    out.append(
        f'{pad}  <geom type="box" size="0.015 0.023 0.009" rgba="0.30 0.33 0.38 1"/>'
    )
    # 工具中心点：两指尖之间，IK 的末端目标就解到这里
    out.append(f'{pad}  <site name="tcp" pos="0 0 0.039" size="0.002" rgba="0 0 0 0"/>')
    # 第一人称：挂在掌部侧后方（侧装+略微后移），前视+朝夹爪内倾，视场较宽，能看到手指
    out.append(
        f'{pad}  <camera name="wrist_cam" pos="0.045 0 -0.015" '
        f'xyaxes="0.985 0 0.174 0 -1 0" fovy="85"/>'
    )
    # 第三人称：与第一人称同方向（沿工具轴朝前），只是后退并偏到侧面，能看到夹爪本体
    out.append(
        f'{pad}  <camera name="chase_cam" pos="0.055 -0.01 -0.23" '
        f'xyaxes="1 0 0 0 -1 0" fovy="60"/>'
    )
    for side, sign in (("l", 1), ("r", -1)):
        out.append(
            f'{pad}  <body name="finger_{side}" pos="0 {sign * 0.005:.4f} 0.009">'
        )
        out.append(
            f'{pad}    <joint name="grip_{side}" type="slide" axis="0 {sign} 0" '
            f'range="0 0.015" damping="0.3"/>'
        )
        out.append(
            f'{pad}    <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>'
        )
        out.append(
            f'{pad}    <geom type="box" size="0.005 0.004 0.015" pos="0 0 0.015" '
            f'rgba="0.22 0.24 0.28 1"/>'
        )
        out.append(f'{pad}  </body>')
    out.append(f'{pad}</body>')


def build_mjcf(urdf_path: str, mesh_dir: str) -> str:
    """把 mycobot 的 URDF 转成带地面、相机、位置执行器的 MJCF。"""
    links, joints, root_link = _parse_urdf(urdf_path, mesh_dir)
    out = ['<mujoco model="mycobot_280_pi">',
           '  <compiler angle="radian" autolimits="true"/>',
           '  <option timestep="0.002" integrator="implicitfast"/>',
           '  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7"/>'
           '  <quality shadowsize="8192"/>'
           '  <global offwidth="640" offheight="480"/></visual>',
           "  <asset>"]
    for name, (path, _, _) in links.items():
        out.append(f'    <mesh name="{name}" file="{path}"/>')
    out += ["  </asset>", "  <worldbody>",
            '    <light pos="0.25 -0.25 1.4" dir="-0.18 0.18 -1" directional="true" castshadow="false"/>',
            '    <geom name="floor" type="plane" size="6 6 0.02" rgba="0.55 0.58 0.62 1"/>',
            f'    <body name="cube" pos="{fmt(CUBE_INITIAL_POS)}">',
            '      <freejoint/>',
            f'      <geom type="box" size="{fmt([CUBE_INITIAL_SIZE_MM / 2000.0] * 3)}" rgba="0.85 0.35 0.25 1"/>',
            '    </body>']

    def emit(link_name: str, indent: int, parent_joint=None) -> None:
        pad = " " * indent
        if parent_joint is None:
            out.append(f'{pad}<body name="{link_name}">')
        else:
            w, x, y, z = rpy_to_quat(*parent_joint["rpy"])
            out.append(
                f'{pad}<body name="{link_name}" pos="{fmt(parent_joint["xyz"])}" '
                f'quat="{w:.6g} {x:.6g} {y:.6g} {z:.6g}">'
            )
            out.append(
                f'{pad}  <joint name="{parent_joint["name"]}" type="hinge" '
                f'axis="{fmt(parent_joint["axis"])}" range="-{JOINT_LIMIT} {JOINT_LIMIT}"/>'
            )
        path, xyz, rpy = links[link_name]
        w, x, y, z = rpy_to_quat(*rpy)
        out.append(
            f'{pad}  <geom type="mesh" mesh="{link_name}" pos="{fmt(xyz)}" '
            f'quat="{w:.6g} {x:.6g} {y:.6g} {z:.6g}" rgba="0.88 0.89 0.92 1" '
            f'contype="0" conaffinity="0"/>'  # 只用于显示，关掉碰撞避免自碰撞发散
        )
        out.append(f'{pad}  <inertial pos="0 0 0" mass="0.35" diaginertia="0.0004 0.0004 0.0004"/>')
        if link_name == "joint6_flange":
            _emit_gripper(out, f"{pad}  ")
        for j in joints:
            if j["parent"] == link_name:
                emit(j["child"], indent + 2, j)
        out.append(f"{pad}</body>")

    emit(root_link, 4)
    out += ["  </worldbody>", "  <actuator>"]
    for j in joints:
        out.append(
            f'    <position name="act_{j["name"]}" joint="{j["name"]}" kp="200" kv="10" '
            f'ctrlrange="-{JOINT_LIMIT} {JOINT_LIMIT}"/>'
        )
    for side in ("l", "r"):
        out.append(
            f'    <position name="act_grip_{side}" joint="grip_{side}" kp="1000" kv="20" '
            f'ctrlrange="0 0.015"/>'
        )
    out += ["  </actuator>", "</mujoco>"]
    return "\n".join(out)
