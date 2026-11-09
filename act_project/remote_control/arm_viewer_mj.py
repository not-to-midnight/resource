"""myCobot 280 Pi 的 MuJoCo 实时仿真画面（HTTP MJPEG 推流）。

与 arm_viewer.py 的页面完全一致：机械臂画面 + 6 个关节滑块 + 训练进度面板。
区别是物理与渲染改用 MuJoCo（EGL 无头渲染），模型由 mycobot 的 URDF 现场转成 MJCF。

用法：
    MUJOCO_GL=egl python3 arm_viewer_mj.py --port 8080
"""

import os

# 必须在 import mujoco 之前指定渲染后端
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import io
import json
import math
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
from PIL import Image

WIDTH, HEIGHT, FPS = 640, 480, 30
URDF = (
    "/home/lab416/resource/act_project/clones/mycobot_ros"
    "/mycobot_description/urdf/mycobot_pi/mycobot_urdf.urdf"
)
MESH_DIR = os.path.dirname(URDF)
JOINT_LIMIT = 3.14

latest_frame = b""
latest_wrist = b""
latest_chase = b""
frame_lock = threading.Lock()

# 6 个关节的目标角度（弧度），由网页滑块下发
targets = [0.0] * 6
targets_lock = threading.Lock()

# 夹爪开度（米），0 = 闭合，0.02 = 全开
gripper = 0.0


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


def build_mjcf(urdf_path: str, mesh_dir: str) -> str:
    """把 mycobot 的 URDF 转成带地面、相机、位置执行器的 MJCF。"""
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
            '    <body name="cube" pos="0.33 -0.10 0.016">',
            '      <freejoint/>',
            '      <geom type="box" size="0.016 0.016 0.016" rgba="0.85 0.35 0.25 1"/>',
            '    </body>']

    def emit_gripper(pad: str) -> None:
        """法兰末端挂一个平行两指夹爪 + 侧装腕部相机（eye-in-hand）。"""
        out.append(f'{pad}<body name="gripper_base" pos="0 0 0.012">')
        out.append(f'{pad}  <inertial pos="0 0 0" mass="0.06" diaginertia="1e-5 1e-5 1e-5"/>')
        out.append(
            f'{pad}  <geom type="box" size="0.015 0.023 0.009" rgba="0.30 0.33 0.38 1"/>'
        )
        # 第一人称：装在掌部侧面，严格朝向工具轴正前方
        out.append(
            f'{pad}  <camera name="wrist_cam" pos="0.035 0 0.018" '
            f'xyaxes="1 0 0 0 -1 0" fovy="60"/>'
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
            emit_gripper(f"{pad}  ")
        for j in joints:
            if j["parent"] == link_name:
                emit(j["child"], indent + 2, j)
        out.append(f"{pad}</body>")

    emit(root_link, 4)
    out += ["  </worldbody>", "  <actuator>"]
    for j in joints:
        out.append(
            f'    <position name="act_{j["name"]}" joint="{j["name"]}" kp="200" kv="20" '
            f'ctrlrange="-{JOINT_LIMIT} {JOINT_LIMIT}"/>'
        )
    for side in ("l", "r"):
        out.append(
            f'    <position name="act_grip_{side}" joint="grip_{side}" kp="1000" kv="20" '
            f'ctrlrange="0 0.015"/>'
        )
    out += ["  </actuator>", "</mujoco>"]
    return "\n".join(out)


def render_loop() -> None:
    """独立的仿真 + 渲染线程，持续把 JPEG 写进 latest_frame。"""
    global latest_frame, latest_wrist, latest_chase

    mjcf = build_mjcf(URDF, MESH_DIR)
    with open("/tmp/mycobot_mjcf.xml", "w") as f:
        f.write(mjcf)  # 留一份便于排查
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    wrist_renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    chase_renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.20]
    cam.distance = 0.75
    cam.azimuth = 135
    cam.elevation = -22
    print(f"MuJoCo 模型加载成功，nq={model.nq}，执行器={model.nu}")
    substeps = max(1, round((1.0 / FPS) / model.opt.timestep))

    while True:
        with targets_lock:
            current = list(targets)
            grip = gripper
        data.ctrl[:6] = current
        data.ctrl[6:] = grip
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
        main_jpg = buf.getvalue()

        wrist_renderer.update_scene(data, camera="wrist_cam")
        buf = io.BytesIO()
        Image.fromarray(wrist_renderer.render()).save(buf, format="JPEG", quality=80)
        wrist_jpg = buf.getvalue()

        chase_renderer.update_scene(data, camera="chase_cam")
        buf = io.BytesIO()
        Image.fromarray(chase_renderer.render()).save(buf, format="JPEG", quality=80)
        chase_jpg = buf.getvalue()

        with frame_lock:
            latest_frame = main_jpg
            latest_wrist = wrist_jpg
            latest_chase = chase_jpg

        time.sleep(1.0 / FPS)


class StreamHandler(BaseHTTPRequestHandler):
    train_log = ""  # 由 main() 注入的训练日志路径

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path.endswith("/stream_cam"):
            self._stream("wrist")
        elif path.endswith("/stream_chase"):
            self._stream("chase")
        elif path.endswith("/stream"):
            self._stream("main")
        elif path.endswith("/set"):
            self._set_joints()
        elif path.endswith("/training"):
            self._training()
        elif path.endswith("/favicon.ico"):
            self.send_response(204)
            self.end_headers()
        else:
            self._index()

    def _index(self) -> None:
        html = """<!DOCTYPE html>
<html><head><title>myCobot 仿真 + 训练进度</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:16px}
.wrap{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
img{border:1px solid #444;max-width:100%}
.panel{background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:16px;min-width:240px}
.panel h3{margin:0 0 12px;font-size:15px;color:#9cdcfe}
.row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #2a2a2a;font-size:14px}
.row span:last-child{font-family:monospace;color:#b5cea8}
.muted{color:#888;font-size:12px;margin-top:8px;line-height:1.6}
.slider{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:13px}
.slider label{width:26px;color:#9cdcfe}
.slider input[type=range]{flex:1;accent-color:#4ea1ff}
.slider output{width:52px;text-align:right;font-family:monospace;color:#b5cea8}
button{background:#2d2d2d;color:#eee;border:1px solid #444;border-radius:5px;padding:6px 12px;cursor:pointer;font-size:13px;margin-top:8px}
button:hover{background:#3a3a3a}
</style></head>
<body>
<div class="wrap">
  <div><h3 style="margin-top:0">固定视角</h3><img src="/stream"></div>
  <div><h3 style="margin-top:0">第一人称（腕部相机）</h3><img src="/stream_cam"></div>
  <div><h3 style="margin-top:0">第三人称（跟随夹爪）</h3><img src="/stream_chase"></div>
  <div class="panel">
    <h3>关节控制</h3>
    <div id="sliders"></div>
    <button onclick="resetJoints()">全部回零</button>
  </div>
  <div class="panel">
    <h3>训练进度</h3>
    <div class="row"><span>Epoch</span><span id="epoch">-</span></div>
    <div class="row"><span>Train loss</span><span id="train">-</span></div>
    <div class="row"><span>Val loss</span><span id="val">-</span></div>
    <div class="row"><span>状态</span><span id="status">-</span></div>
    <div class="muted" id="src"></div>
  </div>
</div>
<script>
const box = document.getElementById('sliders');
for (let i = 0; i < 6; i++) {
  const row = document.createElement('div');
  row.className = 'slider';
  row.innerHTML = `<label>J${i+1}</label>
    <input type="range" min="-160" max="160" step="1" value="0" id="j${i}">
    <output id="o${i}">0°</output>`;
  box.appendChild(row);
  row.querySelector('input').addEventListener('input', sendJoints);
}
{
  const row = document.createElement('div');
  row.className = 'slider';
  row.innerHTML = `<label>G</label>
    <input type="range" min="0" max="15" step="1" value="0" id="j6">
    <output id="o6">0mm</output>`;
  box.appendChild(row);
  row.querySelector('input').addEventListener('input', sendJoints);
}

function currentJoints() {
  const vals = [];
  for (let i = 0; i < 6; i++) {
    const v = document.getElementById('j'+i).value;
    document.getElementById('o'+i).textContent = v + '°';
    vals.push(v);
  }
  const g = document.getElementById('j6').value;
  document.getElementById('o6').textContent = g + 'mm';
  vals.push(g);
  return vals;
}

function sendJoints() {
  fetch('/set?angles=' + currentJoints().join(',')).catch(() => {});
}

function resetJoints() {
  for (let i = 0; i < 7; i++) document.getElementById('j'+i).value = 0;
  sendJoints();
}

async function tick(){
  try{
    const d = await (await fetch('/training')).json();
    document.getElementById('epoch').textContent = d.epoch ?? '-';
    document.getElementById('train').textContent = d.train_loss ?? '-';
    document.getElementById('val').textContent = d.val_loss ?? '-';
    document.getElementById('status').textContent = d.status;
    document.getElementById('src').textContent = d.log;
  }catch(e){ document.getElementById('status').textContent = '连接中断'; }
}
tick(); setInterval(tick, 1000);
</script>
</body></html>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _stream(self, kind: str = "main") -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    jpg = {"wrist": latest_wrist, "chase": latest_chase}.get(
                        kind, latest_frame
                    )
                if not jpg:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg + b"\r\n")
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _set_joints(self) -> None:
        """接收 ?angles=度,度,...（6 个关节，可选第 7 个为夹爪开度 mm）。"""
        global gripper
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        raw = query.get("angles", [""])[0]
        ok = False
        try:
            vals = [float(v) for v in raw.split(",")]
            if len(vals) in (6, 7):
                rads = [max(-2.9, min(2.9, math.radians(d))) for d in vals[:6]]
                with targets_lock:
                    targets[:] = rads
                    if len(vals) == 7:
                        gripper = max(0.0, min(0.015, vals[6] / 1000.0))
                ok = True
        except ValueError:
            pass

        with targets_lock:
            grip_mm = round(gripper * 1000, 1)
            body = json.dumps(
                {
                    "ok": ok,
                    "angles": [round(math.degrees(a), 1) for a in targets],
                    "gripper_mm": grip_mm,
                }
            ).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _training(self) -> None:
        """解析训练日志，返回最近一次的 epoch 和 loss。"""
        info = {"epoch": None, "train_loss": None, "val_loss": None,
                "status": "未开始", "log": self.train_log}
        if self.train_log and os.path.exists(self.train_log):
            try:
                with open(self.train_log, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    f.seek(max(0, f.tell() - 200_000))
                    text = f.read().decode("utf-8", errors="ignore")
            except OSError:
                text = ""

            for line in text.splitlines():
                m = re.search(r"Epoch\s+(\d+)", line)
                if m:
                    info["epoch"] = int(m.group(1))
                    info["status"] = "训练中/最近一次"
                m = re.search(r"Train loss:\s*([0-9.eE+-]+)", line)
                if m:
                    info["train_loss"] = round(float(m.group(1)), 6)
                m = re.search(r"Val loss:\s*([0-9.eE+-]+)", line)
                if m:
                    info["val_loss"] = round(float(m.group(1)), 6)
                if "Training finished" in line:
                    info["status"] = "已结束"

        body = json.dumps(info).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.requestline}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="myCobot MuJoCo 实时画面")
    parser.add_argument("--port", type=int, default=8080, help="HTTP 端口")
    parser.add_argument(
        "--log",
        default="/home/lab416/resource/act_project/train.log",
        help="训练日志路径（ACT 脚本的 stdout 重定向到这里）",
    )
    args = parser.parse_args()

    StreamHandler.train_log = args.log
    threading.Thread(target=render_loop, daemon=True).start()
    time.sleep(1.0)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), StreamHandler)
    print(f"推流已启动: http://127.0.0.1:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
