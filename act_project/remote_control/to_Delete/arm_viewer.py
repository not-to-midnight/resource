"""在 Trae 里实时查看 myCobot 仿真画面（无需显卡、无需 X server）。

原理：PyBullet 用 DIRECT 模式离屏渲染，把每帧 JPEG 通过 HTTP MJPEG 推流，
在 Trae 内置浏览器里打开 http://127.0.0.1:8080/ 即可看到实时画面。

用法：
    python3 arm_viewer.py                  # 默认端口 8080
    python3 arm_viewer.py --port 8081

查看：Trae 命令面板 -> Simple Browser: Show -> 输入 http://127.0.0.1:8080/
"""

import argparse
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import io
import json
import os
import re
import urllib.parse

import pybullet as p
import pybullet_data
from PIL import Image

WIDTH, HEIGHT = 640, 480
FPS = 30
ROBOT_URDF = (
    "/home/lab416/resource/act_project/clones/mycobot_ros"
    "/mycobot_description/urdf/mycobot_pi/mycobot_urdf.urdf"
)
PKG_ROOT = "/home/lab416/resource/act_project/clones/mycobot_ros"

latest_frame = b""
frame_lock = threading.Lock()

# 6 个关节的目标角度（弧度），由网页滑块下发
targets = [0.0] * 6
targets_lock = threading.Lock()


def render_loop() -> None:
    """在独立线程里跑仿真并持续渲染，结果写入 latest_frame。"""
    global latest_frame

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.setAdditionalSearchPath(PKG_ROOT)

    robot_id = p.loadURDF(ROBOT_URDF, basePosition=[0, 0, 0], useFixedBase=True)
    print(f"机械臂加载成功，关节数: {p.getNumJoints(robot_id)}")

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.15],
        distance=0.7,
        yaw=45,
        pitch=-30,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=WIDTH / HEIGHT, nearVal=0.01, farVal=3
    )

    while True:
        with targets_lock:
            current = list(targets)
        for j, angle in enumerate(current):
            p.setJointMotorControl2(
                robot_id, j, p.POSITION_CONTROL, targetPosition=angle, force=200
            )
        p.stepSimulation()

        img = p.getCameraImage(
            WIDTH, HEIGHT, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER
        )
        rgb = img[2][:, :, :3]
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
        with frame_lock:
            latest_frame = buf.getvalue()

        time.sleep(1.0 / FPS)


class StreamHandler(BaseHTTPRequestHandler):
    train_log = ""  # 由 main() 注入的训练日志路径

    def do_GET(self) -> None:
        # 忽略查询参数，Simple Browser 可能带上额外前缀
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path.endswith("/stream"):
            self._stream()
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
  <div><h3 style="margin-top:0">myCobot 280 Pi 实时仿真</h3><img src="/stream"></div>
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
// 生成 6 个关节滑块
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

function currentJoints() {
  return [...Array(6).keys()].map(i => {
    const v = document.getElementById('j'+i).value;
    document.getElementById('o'+i).textContent = v + '°';
    return v;
  });
}

function sendJoints() {
  const a = currentJoints();
  fetch('/set?angles=' + a.join(',')).catch(() => {});
}

function resetJoints() {
  for (let i = 0; i < 6; i++) document.getElementById('j'+i).value = 0;
  sendJoints();
}

async function tick(){
  try{
    const r = await fetch('/training');
    const d = await r.json();
    document.getElementById('epoch').textContent = d.epoch ?? '-';
    document.getElementById('train').textContent = d.train_loss ?? '-';
    document.getElementById('val').textContent = d.val_loss ?? '-';
    document.getElementById('status').textContent = d.status;
    document.getElementById('src').textContent = d.log;
  }catch(e){
    document.getElementById('status').textContent = '连接中断';
  }
}
tick(); setInterval(tick, 1000);
</script>
</body></html>""".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _set_joints(self) -> None:
        """接收 ?angles=度,度,...（6 个），转成弧度写入 targets。"""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        raw = query.get("angles", [""])[0]
        ok = False
        try:
            degs = [float(v) for v in raw.split(",")]
            if len(degs) == 6:
                rads = [max(-2.9, min(2.9, math.radians(d))) for d in degs]
                with targets_lock:
                    targets[:] = rads
                ok = True
        except ValueError:
            pass

        body = json.dumps({"ok": ok, "angles": [round(math.degrees(a), 1) for a in targets]}).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _training(self) -> None:
        """解析训练日志，返回最近一次的 epoch 和 loss。"""
        info = {
            "epoch": None,
            "train_loss": None,
            "val_loss": None,
            "status": "未开始",
            "log": self.train_log,
        }
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
                    info["status"] = "训练中"
                m = re.search(r"Train loss:\s*([0-9.eE+-]+)", line)
                if m:
                    info["train_loss"] = round(float(m.group(1)), 6)
                m = re.search(r"Val loss:\s*([0-9.eE+-]+)", line)
                if m:
                    info["val_loss"] = round(float(m.group(1)), 6)
                if "Training finished" in line:
                    info["status"] = "已结束"
            if info["epoch"] is not None and info["status"] == "训练中":
                info["status"] = "训练中/最近一次"

        body = json.dumps(info).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    jpg = latest_frame
                if not jpg:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg + b"\r\n")
                time.sleep(1.0 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass  # 浏览器关闭页面

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.requestline}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="myCobot 仿真实时画面（HTTP MJPEG）")
    parser.add_argument("--port", type=int, default=8080, help="HTTP 端口")
    parser.add_argument(
        "--log",
        default="/home/lab416/resource/act_project/train.log",
        help="训练日志路径（ACT 脚本的 stdout 重定向到这里）",
    )
    args = parser.parse_args()

    StreamHandler.train_log = args.log

    threading.Thread(target=render_loop, daemon=True).start()
    # 等第一帧渲染出来再启动服务
    while not latest_frame:
        time.sleep(0.1)

    print(f"推流已启动: http://127.0.0.1:{args.port}/")
    print("在 Trae 命令面板执行 Simple Browser: Show，输入上面的地址")
    ThreadingHTTPServer(("0.0.0.0", args.port), StreamHandler).serve_forever()


if __name__ == "__main__":
    main()
