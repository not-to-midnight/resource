#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0 (OpenPI VLA) + MuJoCo 桌面抓取仿真主程序
============================================
流程：
    1. 加载 MuJoCo 场景，渲染相机 RGB 图像
    2. 读取机器人本体关节状态
    3. 将【图像 + 关节状态 + 语言指令 prompt】送入 π0 VLA 策略
    4. π0 输出 action chunk（连续动作序列）
    5. 循环下发动作到 mujoco，步进仿真，实时可视化
    6. 动作裁剪：限制关节动作增量，防止抖动 / 大幅度跳变
    7. 简单抓取成功判定：夹爪接触物体并抬升高度

使用：
    python run_pi0_grasp.py
    python run_pi0_grasp.py --config config.yaml
    python run_pi0_grasp.py --prompt "pick up the red block"
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# 配置加载
# ----------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """加载 yaml 配置文件"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ----------------------------------------------------------------------------
# π0 策略加载（适配 openpi 官方 API）
# ----------------------------------------------------------------------------
def load_pi0_policy(cfg: dict):
    """
    按照 openpi 官方方式加载预训练 pi0_base 策略。

    官方 API:
        from openpi.training import config as _config
        from openpi.policies import policy_config
        from openpi.shared import download
        config = _config.get_config("pi0_base")
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
    """
    print("[Pi0] 正在加载 π0 策略 ...")
    from openpi.training import config as _config
    from openpi.policies import policy_config
    from openpi.shared import download

    config_name = cfg["model"]["config_name"]
    checkpoint = cfg["model"]["checkpoint"]

    # 加载模型配置
    pi0_config = _config.get_config(config_name)

    # 下载 / 定位 checkpoint
    checkpoint_dir = download.maybe_download(checkpoint)
    print(f"[Pi0] config_name = {config_name}")
    print(f"[Pi0] checkpoint  = {checkpoint_dir}")

    # 创建已训练策略
    policy = policy_config.create_trained_policy(pi0_config, checkpoint_dir)
    print("[Pi0] 策略加载完成。")
    return policy


# ----------------------------------------------------------------------------
# 图像处理：将 MuJoCo 渲染的图像转为 π0 输入格式 [3, 224, 224] float32
# ----------------------------------------------------------------------------
def preprocess_image(rgb: np.ndarray) -> np.ndarray:
    """
    MuJoCo 渲染输出: (H, W, 3) uint8, RGB
    π0 输入要求:    (3, 224, 224) float32, 归一化到 [0, 1]
    """
    img = rgb.astype(np.float32) / 255.0
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    return img


# ----------------------------------------------------------------------------
# 末端执行器 Jacobian 逆运动学（将 EE 速度转为关节速度）
# ----------------------------------------------------------------------------
def solve_joint_velocity_from_ee(
    data, model, ee_site_id, ee_body_id,
    ee_vel: np.ndarray, damping: float = 1e-3
) -> np.ndarray:
    """
    使用雅可比伪逆将末端执行器速度映射为关节速度。

    参数:
        ee_vel: [6,] 末端速度 [vx, vy, vz, wx, wy, wz]
    返回:
        qvel:   [nq,] 关节速度（仅前 6 个关节有效）
    """
    import mujoco

    # 获取末端执行器的 6xN 雅可比矩阵
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_body_id)
    jac = np.vstack([jacp, jacr])  # (6, nv)

    # 取前 6 个关节（机械臂自由度，不含夹爪）
    jac_arm = jac[:, :6]

    # 阻尼最小二乘伪逆 (J^T (J J^T + λI)^{-1})
    JJT = jac_arm @ jac_arm.T
    dq = jac_arm.T @ np.linalg.solve(JJT + damping * np.eye(6), ee_vel)
    return dq


# ----------------------------------------------------------------------------
# 动作裁剪：限制关节角度增量，防止抖动和大幅跳变
# ----------------------------------------------------------------------------
def clip_joint_delta(
    target_qpos: np.ndarray,
    current_qpos: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    """将目标关节位置裁剪到当前位置 ± max_delta 范围内"""
    delta = target_qpos - current_qpos
    delta = np.clip(delta, -max_delta, max_delta)
    return current_qpos + delta


# ----------------------------------------------------------------------------
# 抓取成功判定
# ----------------------------------------------------------------------------
def check_grasp_success(data, model, cfg) -> dict:
    """
    简单抓取成功判定：
        1. 夹爪闭合（两指间距 < 阈值）
        2. 目标物体被抬升（z 高度 - 桌面高度 > 阈值）
    返回 dict: {success, gripper_closed, lifted, info}
    """
    # 夹爪状态：两指间距
    grip_l = data.joint("finger_left").qpos[0]
    grip_r = data.joint("finger_right").qpos[0]
    gripper_gap = abs(grip_l - grip_r) + abs(grip_l + grip_r)
    # 两指都向中心收拢时 gap 小
    gripper_closed = (grip_l < cfg["task"]["gripper_closed_thresh"]) and \
                     (grip_r < cfg["task"]["gripper_closed_thresh"])

    # 目标物体高度
    target_name = cfg["task"]["target_body"]
    try:
        target_body_id = data.body(target_name).id
        target_z = data.xpos[target_body_id][2]
    except Exception:
        target_z = 0.0
    # 桌面高度 0.4
    table_z = 0.4
    lifted = (target_z - table_z) > cfg["task"]["lift_threshold"]

    success = gripper_closed and lifted
    return {
        "success": success,
        "gripper_closed": gripper_closed,
        "lifted": lifted,
        "gap": float(gripper_gap),
        "target_z": float(target_z),
        "lift_height": float(target_z - table_z),
    }


# ----------------------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pi0 + MuJoCo 桌面抓取仿真")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--prompt", type=str, default=None, help="覆盖配置中的任务指令")
    parser.add_argument("--no-viewer", action="store_true", help="不显示可视化窗口")
    args = parser.parse_args()

    # ---- 加载配置 ----
    cfg = load_config(args.config)
    if args.prompt:
        cfg["task"]["prompt"] = args.prompt
    if args.no_viewer:
        cfg["runtime"]["show_viewer"] = False

    scene_xml = cfg["scene"]["xml_path"]
    prompt = cfg["task"]["prompt"]
    print("=" * 70)
    print(f"[配置] 场景文件: {scene_xml}")
    print(f"[配置] 任务指令: {prompt}")
    print(f"[配置] 目标物体: {cfg['task']['target_body']}")
    print("=" * 70)

    # ---- 加载 MuJoCo 模型 ----
    import mujoco
    import mujoco_viewer

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 末端执行器 body id
    ee_body_id = data.body("ee").id

    # ---- 初始化可视化窗口 ----
    viewer = None
    if cfg["runtime"]["show_viewer"]:
        viewer = mujoco_viewer.MujocoViewer(model, data, width=960, height=720)

    # ---- 加载 π0 策略 ----
    policy = load_pi0_policy(cfg)

    # ---- 控制参数 ----
    control_freq = cfg["control"]["control_freq"]
    dt = cfg["control"]["dt"]
    steps_per_control = max(1, int(1.0 / (control_freq * dt)))  # 每控制周期的仿真步数
    max_joint_delta = cfg["control"]["max_joint_delta"]
    action_chunk_steps = cfg["control"]["action_chunk_steps"]
    gripper_open = cfg["control"]["gripper_open"]
    gripper_close = cfg["control"]["gripper_close"]
    kp = cfg["control"]["kp"]
    max_steps = cfg["runtime"]["max_steps"]

    # 渲染参数
    cam_name = cfg["scene"]["camera"]
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    render_w = cfg["scene"]["render_width"]
    render_h = cfg["scene"]["render_height"]
    renderer = mujoco.Renderer(model, height=render_h, width=render_w)

    # ---- 主循环 ----
    print(f"\n[启动] 控制频率 {control_freq}Hz, 每控制周期 {steps_per_control} 个仿真步")
    print(f"[启动] 最大步数 {max_steps}\n")

    step = 0
    grasp_result = None
    last_action = np.zeros(cfg["model"]["action_dim"], dtype=np.float32)

    while step < max_steps:
        # --------------- 推理阶段（每个控制周期一次）---------------
        if step % steps_per_control == 0:
            # 1) 渲染相机 RGB 图像
            renderer.update_scene(data, camera=cam_id)
            rgb = renderer.render()  # (H, W, 3) uint8
            img_chw = preprocess_image(rgb)

            # 2) 读取关节状态 (前6个关节 + 夹爪平均 = 7维状态)
            joint_qpos = np.array([data.joint(f"joint{i}").qpos[0] for i in range(1, 7)])
            grip_l = data.joint("finger_left").qpos[0]
            grip_r = data.joint("finger_right").qpos[0]
            gripper_state = (grip_l + grip_r) / 2.0
            robot_state = np.concatenate([joint_qpos, [gripper_state]])

            # 3) 构造 π0 输入 observation dict
            #    π0 要求: image=[3,H,W], state=关节状态, prompt=语言指令
            example = {
                "observation/exterior_image_1_left": img_chw,
                "observation/state": robot_state.astype(np.float32),
                "prompt": prompt,
            }

            # 4) 调用 π0 推理，得到 action chunk
            t0 = time.time()
            try:
                output = policy.infer(example)
                action_chunk = np.array(output["actions"], dtype=np.float32)
            except Exception as e:
                print(f"[警告] π0 推理失败: {e}，使用上一次动作")
                action_chunk = last_action[None, :]
            infer_time = (time.time() - t0) * 1000

            # 取 chunk 的第一步作为当前动作 (shape: [7,])
            action = action_chunk[0] if action_chunk.ndim > 1 else action_chunk
            last_action = action.copy()

            # 解析动作: [x, y, z, roll, pitch, yaw, gripper]
            # pi0_base 单臂动作: 末端 xyz + rpy + 夹爪
            ee_pos_delta = action[:3].copy()      # 末端位置增量
            ee_rot_delta = action[3:6].copy()     # 末端旋转增量
            gripper_action = action[6].copy()     # 夹爪 (0=开, 1=合 或 连续值)

            # 裁剪末端位移，防止大跳变
            ee_pos_delta = np.clip(ee_pos_delta, -cfg["control"]["max_ee_delta"],
                                   cfg["control"]["max_ee_delta"])

            print(f"[step {step:4d}] 推理耗时 {infer_time:6.1f}ms | "
                  f"动作 xyz={ee_pos_delta.round(4)} | "
                  f"rpy={ee_rot_delta.round(4)} | "
                  f"gripper={gripper_action:.3f}")

            # 末端执行器速度 = 位移增量 / 控制周期
            control_period = steps_per_control * dt
            ee_vel = np.concatenate([ee_pos_delta / control_period,
                                     ee_rot_delta / control_period])

            # 5) 用 Jacobian 伪逆解出关节速度，再积分得到目标关节位置
            dq_vel = solve_joint_velocity_from_ee(data, model, None, ee_body_id, ee_vel)
            target_qpos = joint_qpos + dq_vel[:6] * control_period

            # 6) 关节动作裁剪（限制单步增量，防抖动）
            target_qpos = clip_joint_delta(target_qpos, joint_qpos, max_joint_delta)

            # 7) 夹爪控制: 将 gripper_action 映射到开合范围
            #    假设 gripper_action 越大越闭合
            grip_norm = float(np.clip(gripper_action, 0.0, 1.0))
            grip_target = gripper_open * (1.0 - grip_norm) + gripper_close * grip_norm

        # --------------- 执行阶段（每个仿真步）---------------
        # 关节位置控制: ctrl = kp * (target - current)
        for i in range(6):
            cur = data.joint(f"joint{i+1}").qpos[0]
            data.ctrl[i] = kp * (target_qpos[i] - cur)

        # 夹爪控制
        data.ctrl[6] = kp * (grip_target - grip_l)
        data.ctrl[7] = kp * (grip_target - grip_r)

        # 步进仿真
        mujoco.mj_step(model, data)
        step += 1

        # 可视化
        if viewer is not None and viewer.is_alive():
            viewer.render()

        # --------------- 抓取成功判定（每个控制周期检查一次）---------------
        if step % steps_per_control == 0:
            grasp_result = check_grasp_success(data, model, cfg)
            status_str = "✔ 成功" if grasp_result["success"] else "进行中"
            print(f"            [判定] 夹爪闭合={grasp_result['gripper_closed']} | "
                  f"抬升={grasp_result['lifted']} (高度={grasp_result['lift_height']:.4f}m) | "
                  f"结果: {status_str}")

            if grasp_result["success"]:
                print(f"\n🎉 抓取成功！目标物体「{cfg['task']['target_body']}」已被夹爪拾取并抬升。")
                break

    # ---- 结束 ----
    if grasp_result is not None and not grasp_result["success"]:
        print(f"\n⏹ 达到最大步数 {max_steps}，未判定到成功抓取。")
        print(f"   最后状态: 夹爪闭合={grasp_result['gripper_closed']}, "
              f"抬升={grasp_result['lifted']}, "
              f"抬升高度={grasp_result['lift_height']:.4f}m")

    if viewer is not None:
        viewer.close()
    print("[结束] 仿真结束。")


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
