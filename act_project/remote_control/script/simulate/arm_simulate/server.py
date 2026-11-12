"""HTTP 服务：控制页 + 三路 MJPEG 推流 + 关节角下发。"""

import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from . import config
from .arm_actor import ArmActor


class StreamHandler(BaseHTTPRequestHandler):
    actor: ArmActor = None

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path.endswith("/stream_cam"):
            self._stream("wrist")
        elif path.endswith("/stream_chase"):
            self._stream("chase")
        elif path.endswith("/stream"):
            self._stream("main")
        elif path.endswith("/config"):
            self._get_config()
        elif path.endswith("/state"):
            self._get_arm_state()
        elif path.endswith("/reset"):
            self._reset()
        elif path.endswith("/set"):
            self._set_joints()
        elif path.endswith("/move"):
            self._move_to_pose()
        elif path.endswith("/favicon.ico"):
            self.send_response(204)
            self.end_headers()
        else:
            self._index()

    def _index(self) -> None:
        html = Path(__file__).with_name("index.html").read_bytes()
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
                time.sleep(1.0 / config.FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _get_config(self) -> None:
        body = json.dumps({
            "arm": {
                "angles_deg": config.ARM_INITIAL_ANGLES_DEG,
                "gripper_mm": config.ARM_INITIAL_GRIPPER_MM,
            },
            "ik": {
                "pos": config.IK_INITIAL_POS,
                "gripper_mm": config.IK_INITIAL_GRIPPER_MM,
                "down": config.IK_INITIAL_DOWN,
            },
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_arm_state(self) -> None:
        from function.LogicSystem import LogicSys
        from .arm_manager import ArmSimulateManager

        manager = LogicSys.get(ArmSimulateManager)
        body = json.dumps({"ok": True, "state": manager.get_state()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reset(self) -> None:
        from function.LogicSystem import LogicSys
        from .arm_manager import ArmSimulateManager

        status = 200
        try:
            manager = LogicSys.get(ArmSimulateManager)
            manager.reset()
            body = {"ok": True, "state": manager.get_state()}
        except RuntimeError as e:
            status = 400
            body = {"ok": False, "reason": str(e)}
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _set_joints(self) -> None:
        """接收 ?angles=度,度,...（6 个关节，可选第 7 个为夹爪开度 mm）。"""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        raw = query.get("angles", [""])[0]
        ok = False
        reason = ""
        try:
            vals = [float(v) for v in raw.split(",")]
            if len(vals) in (6, 7):
                self.actor.set_angles(*vals[:6])
                if len(vals) == 7:
                    self.actor.set_gripper_mm(vals[6])
                ok = True
        except ValueError:
            reason = "参数格式错误"
        except RuntimeError as e:  # IK 解算中，手动控制被禁止
            reason = str(e)

        current = self.actor.get_state()
        body = json.dumps(
            {
                "ok": ok,
                "reason": reason,
                "angles": current["angles_deg"],
                "gripper_mm": current["gripper_mm"],
            }
        ).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _move_to_pose(self) -> None:
        """/move?pos=x,y,z&down=1&gripper=10 或 &rot=9个数（行主序旋转矩阵）。

        down=1 等价于工具轴竖直朝下 diag(1,-1,-1)；都不传则保持当前姿态。
        """
        import numpy as np

        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        body = {"ok": False, "reason": ""}
        try:
            raw = query.get("pos", [""])[0]
            pos = [float(v) for v in raw.split(",")]
            if len(pos) != 3:
                raise ValueError("pos 需要 x,y,z 三个数")

            rot = None
            if "rot" in query:
                vals = [float(v) for v in query["rot"][0].split(",")]
                if len(vals) != 9:
                    raise ValueError("rot 需要 9 个数（行主序 3x3）")
                rot = np.array(vals).reshape(3, 3)
            elif query.get("down", ["0"])[0] == "1":
                rot = np.diag([1.0, -1.0, -1.0])

            grip = query.get("gripper", [""])[0]
            gripper_mm = float(grip) if grip else None

            # 通过 LogicSys 服务定位拿 manager（局部 import 避免循环依赖）
            from function.LogicSystem import LogicSys
            from .arm_manager import ArmSimulateManager

            mgr = LogicSys.get(ArmSimulateManager)
            if mgr is None:
                body["reason"] = "ArmSimulateManager 未注册到 LogicSys"
            else:
                result = mgr.move_to_pose(pos, rot, gripper_mm)
                body = {"ok": result.get("ok", False), "reason": result.get("reason", ""),
                        "pos": pos, "ik": result}
        except (ValueError, RuntimeError) as e:
            body["reason"] = str(e)

        data = json.dumps(body).encode()
        self.send_response(200 if body["ok"] else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        pass  # 静默 HTTP 请求日志：拖滑块时 /set 洪水会拖慢调试控制台
