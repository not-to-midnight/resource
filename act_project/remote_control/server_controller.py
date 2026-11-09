"""控制端（Client）——运行在 4080 服务器上，通过键盘远程控制机械臂。

用法：
    python3 server_controller.py                        # 连 192.168.0.41:8888
    python3 server_controller.py --ip 192.168.0.41 --port 8888

按键：
    1~6   选择关节
    w/s   当前关节角度 +5° / -5°
    q     退出
"""

import argparse
import json
import socket
import sys
import termios
import tty

DEFAULT_IP = "192.168.0.41"
DEFAULT_PORT = 8888
JOINT_COUNT = 6
STEP = 5

"""
def setraw(fd, when=TCSAFLUSH):
    mode = tcgetattr(fd)
    mode[IFLAG] = mode[IFLAG] & ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON)
    mode[OFLAG] = mode[OFLAG] & ~(OPOST)
    mode[CFLAG] = mode[CFLAG] & ~(CSIZE | PARENB)
    mode[CFLAG] = mode[CFLAG] | CS8
    mode[LFLAG] = mode[LFLAG] & ~(ECHO | ICANON | IEXTEN | ISIG)
    mode[CC][VMIN] = 1
    mode[CC][VTIME] = 0
    tcsetattr(fd, when, mode)
"""
def get_key() -> str:
    """读取一个按键（raw 模式，无需回车）。"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)# 进入 raw 模式，无需回车,这里函数内部也是调用 tcsetattr
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def send_angles(client: socket.socket, angles: list[float]) -> dict:
    """按行发送 JSON 并同步等待执行端回执。"""
    client.sendall((json.dumps(angles) + "\n").encode("utf-8"))
    data = client.recv(1024).decode("utf-8").strip()
    return json.loads(data) if data else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="远程键盘控制机械臂（TCP Client）")
    parser.add_argument("--ip", default=DEFAULT_IP, help="机器人 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="机器人端口")
    args = parser.parse_args()

    try:
        client = socket.create_connection((args.ip, args.port), timeout=5)
    except OSError as exc:
        print(f"连接失败 {args.ip}:{args.port} -> {exc}")
        print("请确认：机器人端 robot_server.py 已启动、IP 正确、端口未被防火墙拦截")
        return

    client.settimeout(2.0)
    print(f"已连接机器人执行端 {args.ip}:{args.port}")
    print("=== 远程键盘控制 ===")
    print("1~6 选择关节 | w/s 当前关节 ±5° | q 退出")

    angles = [0.0] * JOINT_COUNT
    current_joint = 1

    try:
        while True:
            print(f"\n当前控制: {current_joint}号关节 | 角度: {angles}")
            key = get_key()

            if not key or key == "q" or key == "\x03":  # EOF / q / Ctrl+C
                break
            if key in "123456":
                current_joint = int(key)
                continue
            if key == "w":
                angles[current_joint - 1] += STEP
            elif key == "s":
                angles[current_joint - 1] -= STEP
            else:
                continue

            try:
                reply = send_angles(client, angles)
                if reply.get("ok", False):
                    print(f"已执行: {angles}")
                else:
                    print(f"执行端返回错误: {reply.get('error', reply)}")
            except (OSError, ValueError) as exc:
                print(f"下发失败: {exc}")
                break
    finally:
        client.close()
        print("\n已断开连接")


if __name__ == "__main__":
    main()
