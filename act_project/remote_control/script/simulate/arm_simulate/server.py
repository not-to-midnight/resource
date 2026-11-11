"""HTTP 服务：控制页 + 三路 MJPEG 推流 + 关节角下发。"""

import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler

from .arm_actor import ArmActor
from .config import FPS


class StreamHandler(BaseHTTPRequestHandler):
    # 由 ArmSimulateManager 启动时注入，所有请求共用同一个机械臂实体
    actor: ArmActor = None

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
        elif path.endswith("/favicon.ico"):
            self.send_response(204)
            self.end_headers()
        else:
            self._index()

    def _index(self) -> None:
        html = """<!DOCTYPE html>
<html><head><title>myCobot MuJoCo 仿真</title>
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
.slider .step{width:24px;height:24px;padding:0;margin:0;font-size:15px;line-height:1;border-radius:4px}
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
</div>
<script>
const box = document.getElementById('sliders');
for (let i = 0; i < 6; i++) {
  const row = document.createElement('div');
  row.className = 'slider';
  row.innerHTML = `<label>J${i+1}</label>
    <button class="step" onclick="nudge(${i},-1)">−</button>
    <input type="range" min="-160" max="160" step="1" value="0" id="j${i}">
    <button class="step" onclick="nudge(${i},1)">+</button>
    <output id="o${i}">0°</output>`;
  box.appendChild(row);
  row.querySelector('input').addEventListener('input', sendJoints);
}
{
  const row = document.createElement('div');
  row.className = 'slider';
  row.innerHTML = `<label>G</label>
    <button class="step" onclick="nudge(6,-1)">−</button>
    <input type="range" min="0" max="15" step="1" value="0" id="j6">
    <button class="step" onclick="nudge(6,1)">+</button>
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

function nudge(id, d) {
  const el = document.getElementById('j' + id);
  el.value = Math.min(Number(el.max), Math.max(Number(el.min), Number(el.value) + d));
  sendJoints();
}

function resetJoints() {
  for (let i = 0; i < 7; i++) document.getElementById('j'+i).value = 0;
  sendJoints();
}
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
                jpg = self.actor.get_frame(kind)
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
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        raw = query.get("angles", [""])[0]
        ok = False
        try:
            vals = [float(v) for v in raw.split(",")]
            if len(vals) in (6, 7):
                self.actor.set_angles(*vals[:6])
                if len(vals) == 7:
                    self.actor.set_gripper_mm(vals[6])
                ok = True
        except ValueError:
            pass

        current = self.actor.get_state()
        body = json.dumps(
            {
                "ok": ok,
                "angles": current["angles_deg"],
                "gripper_mm": current["gripper_mm"],
            }
        ).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass  # 静默 HTTP 请求日志：拖滑块时 /set 洪水会拖慢调试控制台
