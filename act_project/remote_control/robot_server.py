"""机器人执行端（Server）——运行在机械臂本体（树莓派/控制盒）上。

职责：打开本地串口控制 myCobot，监听 TCP 端口，接收 4080 服务器下发的
6 个关节角并执行，然后回传 ACK。

用法：
    python3 robot_server.py                 # 连接真实机械臂
    python3 robot_server.py --dry-run       # 不连硬件，只打印（单机联调网络）
    python3 robot_server.py --port 8888 --speed 20
"""

import argparse
import json
import socket
from typing import List

DEFAULT_PORT = 8888
JOINT_COUNT = 6


class Arm:
    """机械臂执行器；dry_run 时只打印，便于在没有硬件的机器上验证链路。"""

    def __init__(self, dry_run: bool, speed: int) -> None:
        self.dry_run = dry_run
        self.speed = speed
        self.mc = None

        if dry_run:
            return

        try:
            from pymycobot.mycobot import MyCobot
        except ImportError:
            raise SystemExit(
                "未安装 pymycobot，请先在机器人端执行：pip install pymycobot\n"
                "（若只想验证网络链路，可加 --dry-run）"
            )

        self.mc = MyCobot("/dev/arm", 115200)
        self.mc.power_on()

    def move(self, angles: List[float]) -> None:
        if self.dry_run:
            print(f"[dry-run] send_angles({angles}, {self.speed})")
        else:
            self.mc.send_angles(angles, self.speed)


def parse_angles(payload: str) -> List[float]:
    """解析一行 JSON，校验并转换为 6 个浮点角度。"""
    data = json.loads(payload)
    if not isinstance(data, list) or len(data) != JOINT_COUNT:
        raise ValueError(f"期望 {JOINT_COUNT} 个关节角的 JSON 数组，收到: {data!r}")
    return [float(v) for v in data]


def serve(port: int, arm: Arm) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    print(f"机器人执行端已启动，监听 0.0.0.0:{port}，等待服务器连接...")

    while True:
        conn, addr = server.accept()
        print(f"服务器已连接: {addr}")
        try:
            buf = b""
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
                # 按换行符切分，最后一段可能是不完整的半条消息，留到下次
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        angles = parse_angles(line.decode("utf-8"))
                        print(f"收到指令，运动到: {angles}")
                        arm.move(angles)
                        reply = {"ok": True, "angles": angles}
                    except Exception as exc:  # 单条指令出错不应断开连接
                        print(f"指令无效: {exc}")
                        reply = {"ok": False, "error": str(exc)}
                    conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
        except Exception as exc:
            print(f"连接断开: {exc}")
        finally:
            conn.close()
            print("等待下一次连接...")


def main() -> None:
    parser = argparse.ArgumentParser(description="机器人执行端（TCP Server）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口")
    parser.add_argument(
        "--speed", type=int, default=20, help="send_angles 的速度参数（0~100）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="不连接硬件，仅打印收到的角度"
    )
    args = parser.parse_args()

    arm = Arm(args.dry_run, args.speed)
    try:
        serve(args.port, arm)
    except KeyboardInterrupt:
        print("\n执行端已退出")


if __name__ == "__main__":
    main()
