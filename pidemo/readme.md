# π0 (OpenPI VLA) + MuJoCo 桌面抓取仿真

本工程将 OpenPI 官方预训练的 **π0** 视觉-语言-动作 (VLA) 模型接入 **MuJoCo** 仿真环境，实现单 6 自由度机械臂的桌面抓取闭环控制。

```
相机RGB图像 + 关节状态 + 语言指令
            │
            ▼
       π0 VLA 策略
            │
            ▼
      action chunk (xyz + rpy + gripper)
            │
            ▼
   动作裁剪 → 雅可比IK → 关节控制 → MuJoCo步进
            │
            ▼
        实时可视化 + 抓取成功判定
```

---

## 1. 工程结构

```
.
├── mujoco_scene.xml      # 机器人、桌面、杯子、方块仿真场景 (MuJoCo XML)
├── run_pi0_grasp.py      # 主运行脚本 (中文注释)
├── config.yaml           # 实验配置：模型名称、控制频率、动作参数、任务prompt
├── readme.md             # 本文档
├── assets/               # 预览图、运行视频等素材
├── 实验日志/             # 实验过程记录
├── finetune_demo/        # LoRA 微调示例工程
└── 学习资料/             # 参考资料
```

---

## 2. 环境要求

- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA GPU (8GB+ 显存，如 RTX 4080/4090)
- **Python**: 3.11 (openpi 官方推荐)
- **已就绪**: openpi 官方仓库 + uv 环境 + π0_base 预训练权重

---

## 3. 环境部署步骤

### 3.1 安装 openpi 及依赖

进入 openpi 官方仓库目录，使用 uv 同步依赖并安装：

```bash
cd /path/to/openpi

# 同步依赖（GIT_LFS_SKIP_SMUDGE=1 为 LeRobot 兼容所必须）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 验证 transformers 版本 (需 4.53.2)
uv pip show transformers
```

### 3.2 安装仿真依赖

在 openpi 的虚拟环境中安装 MuJoCo 与可视化：

```bash
uv pip install mujoco mujoco-python-viewer pyyaml numpy
```

> `mujoco-python-viewer` 用于实时窗口可视化；若在无显示环境运行，用 `--no-viewer`。

### 3.3 安装 gsutil（用于可靠下载权重）

> ⚠️ openpi 通过 `gsutil` 下载 `openpi-assets` 权重。**若缺少 gsutil，会回退到 gcsfs，极易下载出损坏的 checkpoint**（表现为加载时报 `OUT_OF_RANGE` 或缓存目录出现嵌套的 `pi0_droid/pi0_droid`）。

```bash
# 在 openpi 虚拟环境中安装 gsutil
uv pip install gsutil

# 运行时确保 .venv/bin 在 PATH 中，openpi 才能调用到 gsutil：
export PATH=/path/to/openpi/.venv/bin:$PATH
```

### 3.4 下载 π0 权重

> 📌 **关于模型选择**：openpi 未提供 `pi0_base` 的推理配置（base 权重仅用于微调，无法直接推理）。因此本工程默认使用 **`pi0_droid`**——在 DROID 桌面操作数据上微调、支持 0-shot 推理的模型，其动作空间正是桌面抓取所需的 8 维。

权重在首次运行时自动下载到 `~/.cache/openpi`（约 11 GiB），也可手动下载：

```bash
# gs://openpi-assets/checkpoints/pi0_droid
PATH=/path/to/openpi/.venv/bin:$PATH python -c \
  "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi0_droid'))"
```

设置缓存目录（可选）：

```bash
export OPENPI_DATA_HOME=/path/to/your/cache
```

### 3.5 克隆 / 进入本工程

将本工程四个文件放在同一目录（建议放在 openpi 仓库内，便于直接复用其虚拟环境）：

```bash
cd /path/to/openpi
mkdir -p projects/pi0_grasp && cd projects/pi0_grasp
# 将 mujoco_scene.xml / run_pi0_grasp.py / config.yaml / readme.md 放入此目录
```

---

## 4. 运行命令

### 4.1 使用默认配置运行

```bash
uv run python run_pi0_grasp.py
```

### 4.2 指定配置文件

```bash
uv run python run_pi0_grasp.py --config config.yaml
```

### 4.3 覆盖任务指令（无需改 yaml）

```bash
uv run python run_pi0_grasp.py --prompt "pick up the red block"
```

### 4.4 无窗口模式（远程 / headless）

```bash
uv run python run_pi0_grasp.py --no-viewer
```

---

## 5. 配置参数说明 (`config.yaml`)

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `model.config_name` | openpi 模型配置名 | `pi0_droid` |
| `model.checkpoint` | checkpoint 路径 / gs 地址 | `gs://openpi-assets/checkpoints/pi0_droid` |
| `model.action_dim` | 动作维度（pi0_droid 为 8） | `8` |
| `scene.camera` | 全局观测相机名 | `global_cam` |
| `scene.wrist_camera` | 腕部观测相机名 | `wrist_cam` |
| `scene.table_height` | 桌面高度（抓取判定用） | `0.4` |
| `task.prompt` | 抓取语言指令 | `pick up the blue cup` |
| `task.target_body` | 目标物体 body 名 | `cup` |
| `task.lift_threshold` | 判定抬升成功高度 (m) | `0.06` |
| `control.control_freq` | π0 推理频率 (Hz) | `10` |
| `control.action_chunk_steps` | 每次推理下发的 chunk 步数 | `5` |
| `control.max_joint_delta` | 单步关节角度最大增量 (rad) | `0.05` |
| `control.max_ee_delta` | 单步末端最大位移 (m) | `0.01` |
| `control.kp` | 关节位置控制增益 | `50.0` |
| `runtime.max_steps` | 最大仿真步数 | `500` |

---

## 6. 场景与物体修改

编辑 `mujoco_scene.xml` 即可调整：

- **物体位置**：修改 `<body name="cup" pos="...">` 与 `<body name="block" pos="...">`
- **物体尺寸**：修改对应 `<geom size="...">`
- **桌面尺寸/颜色**：修改 `table` geom 的 `size` 和 `material`
- **相机视角**：修改 `global_cam` 的 `pos` 与 `xyaxes`

---

## 7. 动作裁剪与安全限制

为防止 π0 输出导致机械臂抖动或大幅跳变，主程序实现了两级裁剪：

1. **末端位移裁剪**：`ee_pos_delta` 被限制在 `±max_ee_delta` 内。
2. **关节增量裁剪**：目标关节位置被限制在当前位置 `±max_joint_delta` 内。

关节控制采用位置伺服 `ctrl = kp * (target - current)`，并在每个仿真步平滑执行。

---

## 8. 抓取成功判定

当同时满足以下两个条件时判定为抓取成功：

1. **夹爪闭合**：左右两指位移均 < `gripper_closed_thresh`（默认 0.015m）。
2. **物体抬升**：目标物体 z 坐标 - 桌面高度 > `lift_threshold`（默认 0.06m）。

---

## 9. 给 π0 使用的英文测试指令列表

以下指令可直接通过 `--prompt "..."` 或修改 `config.yaml` 中 `task.prompt` 使用：

### 9.1 抓取类

```
pick up the blue cup
pick up the red block
grab the cup
grab the red cube
lift the cup off the table
pick up the block and hold it
```

### 9.2 放置 / 移动类

```
pick up the cup and move it to the left
pick up the block and place it on the table
move the red block to the right
push the cup forward
slide the block toward the robot
```

### 9.3 多物体选择类

```
pick up the cup, not the block
grasp the blue object
pick up the red block only
lift the cup and ignore the block
```

### 9.4 方向 / 姿态类

```
pick up the cup from the side
grasp the block from above
pick up the cup and turn it upside down
lift the block straight up
```

### 9.5 复合指令类

```
pick up the cup and put it down on the right
grasp the block, lift it, and place it next to the cup
pick up the red block then pick up the cup
move the block away from the cup
```

> **提示**：π0_base 为预训练基础模型，对特定机器人/场景的指令跟随能力有限。
> 若抓取效果不佳，可：(1) 微调模型到本场景数据；(2) 使用 `pi0_droid` / `pi05_droid` 等已微调专家模型；
> (3) 调小 `max_joint_delta` / `max_ee_delta` 以提高控制平滑性。

---

## 10. 常见问题

| 问题 | 解决 |
|------|------|
| `ModuleNotFoundError: openpi` | 确认已在 openpi 仓库内 `uv pip install -e .` |
| 推理 OOM | 使用更小 batch / 单卡；设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` |
| 窗口无法显示 | 确认有显示器 / X11 转发；或加 `--no-viewer` |
| 机械臂抖动 | 减小 `max_joint_delta` 和 `max_ee_delta`，增大 `control_freq` |
| 抓取判定不触发 | 降低 `lift_threshold` 或 `gripper_closed_thresh` |
<<<<<<<< HEAD:demolab/readme.md
| 动作维度不匹配 | 确认 `action_dim` 与所用模型一致（pi0_droid=8） |
| 加载 checkpoint 报 `OUT_OF_RANGE` | checkpoint 下载损坏：删除 `~/.cache/openpi/openpi-assets/checkpoints/pi0_droid*`，装好 gsutil 后重下 |


## 11. π0_droid 观测 / 动作接口说明

`pi0_droid`（DROID 微调模型）的输入输出如下，主程序 `run_pi0_grasp.py` 已按此适配：

**输入 observation（dict）**

| 键 | 形状 / 类型 | 说明 |
|----|------------|------|
| `observation/exterior_image_1_left` | (224,224,3) uint8 | 全局相机 RGB |
| `observation/wrist_image_left` | (224,224,3) uint8 | 腕部相机 RGB |
| `observation/joint_position` | (6,) float32 | 机械臂 6 关节角 |
| `observation/gripper_position` | (1,) float32 | 夹爪位置 |
| `prompt` | str | 语言指令 |

> 状态与动作会被 openpi 的 `PadStatesAndActions` 自动填充到 32 维，因此 6 自由度机械臂可直接使用。

**输出**

```python
result = policy.infer(example)      # -> {"actions": np.ndarray}
actions = result["actions"]          # shape: (chunk_len, 8)
```

动作 8 维语义：`[:3]`=末端位置增量，`[3:6]`=末端旋转增量，`[6]`=夹爪，`[7]`=保留未用。

> ⚠️ **重要说明**：`pi0_droid` 是在真实 Franka/DROID 平台数据上训练的，与本案的自定义 6 自由度机械臂、MuJoCo 场景存在域差异。因此**零样本抓取成功率有限**；本工程的价值在于提供完整可跑通的 VLA 仿真闭环骨架（渲染 → 推理 → action chunk → IK → 裁剪 → 抓取判定）。若需稳定抓取，建议用本场景数据微调模型。


lab test
========
| 动作维度不匹配 | 确认 `action_dim` 与所用模型一致 (pi0_base=7) |


test
>>>>>>>> 32e60e03babd62f8eba1c695dbb39e1951706a72:pidemo/readme.md
