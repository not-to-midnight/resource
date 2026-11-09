# 远程控制联调说明

机器人只负责执行，4080（lab416）负责发指令，两者走 TCP 通信。

```
4080 (lab416)  server_controller.py  ──TCP──▶  robot_server.py  ──串口 /dev/arm──▶  机械臂
   发 6 个关节角                              192.168.0.41:8888               myCobot
```

## 1. 从 lab416 连接机器人

lab416 已配置免密 SSH，别名 `robot`（见 `~/.ssh/config`）：

```bash
ssh robot          # 等价于 ssh robuster@192.168.0.41
```

验证连接确实是从 lab416 发起的：

```bash
ssh robot 'echo "来源: $SSH_CLIENT"'
# 来源: 192.168.0.181 51044 22   ← 192.168.0.181 即 lab416
```

SSH 配置（如换机器需重新配）：

```sshconfig
Host robot
    HostName 192.168.0.41
    User robuster
    Port 22
    IdentityFile ~/.ssh/id_ed25519
```

**不要用 Trae 的 Remote-SSH 连接机器人**：机器人是 Ubuntu 18.04 / glibc 2.27，
Trae 远程服务端要求 glibc ≥ 2.28，会报 GLIBC 版本错误。用 Trae 终端即可，
远程窗口里的终端本来就运行在 lab416 上。

## 2. 运行

机器人端（另开一个终端）：

```bash
ssh robot
cd ~/beetle_ai/scripts
python3 robot_server.py                    # 真机，默认端口 8888、速度 20
python3 robot_server.py --dry-run          # 不连硬件，只打印收到的角度
python3 robot_server.py --port 8888 --speed 20
```

4080 端（在 lab416 的另一个终端）：

```bash
cd /home/lab416/resource/act_project/remote_control
python3 server_controller.py --ip 192.168.0.41 --port 8888
```

按键：`1~6` 选关节，`w` / `s` 当前关节 ±5°，`q` 退出。

第一次真机运行请把手放在急停附近，先用默认速度 20 试。

## 3. 通信协议

每条消息是一行 JSON 加换行符，避免粘包/半包：

- 请求：`[0, 0, 0, 0, 0, 0]\n`，必须是 6 个数值
- 回执：`{"ok": true, "angles": [...]}\n`；失败为 `{"ok": false, "error": "..."}`

单条指令出错不会断开连接。后续接 ACT 模型时，只需把客户端键盘循环换成
「策略推理 → 6 维动作 → 发送」，通信层不用改。

## 4. 常见问题

| 现象 | 原因 |
| --- | --- |
| `Connection refused` | 机器人端 `robot_server.py` 没启动，或端口不一致 |
| `Connection timed out` | IP 不对 / 机器人没联网 / 防火墙拦截 8888 |
| `GLIBC_2.28 not found` | 用了 Trae Remote-SSH，改用 `ssh robot` |
| `No such file: /dev/arm` | 机械臂串口没识别，检查 `ls -l /dev/arm` |
| `未安装 pymycobot` | 机器人端执行 `pip install pymycobot` |
