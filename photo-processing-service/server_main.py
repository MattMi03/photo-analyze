"""
证件照图像处理服务启动入口（PhotoIDServer.exe）

源码运行：python server_main.py
打包后  ：双击 PhotoIDServer.exe（控制台窗口保持开启即服务运行中）
默认监听：http://0.0.0.0:10006
"""

import multiprocessing
import os
import sys
import webbrowser
import threading

# PyInstaller onefile + paddle/onnxruntime 多进程必须在最早处调用
if __name__ == "__main__":
    multiprocessing.freeze_support()

# noinspection PyUnresolvedReferences
from app import app, DEFAULT_BG_IMAGE, ensure_paddle_models  # 导入 FastAPI 应用（触发 startup 钩子）

import uvicorn

HOST = "0.0.0.0"
PORT = 10006


def _print_banner():
    print("=" * 64)
    print("  证件照图像处理服务 (PhotoIDServer)")
    print(f"  监听地址  : http://{HOST}:{PORT}")
    print(f"  本机访问  : http://127.0.0.1:{PORT}")
    print(f"  健康检查  : http://127.0.0.1:{PORT}/health")
    print(f"  接口文档  : http://127.0.0.1:{PORT}/docs")
    print(f"  内置背景图: {DEFAULT_BG_IMAGE} (存在={os.path.exists(DEFAULT_BG_IMAGE)})")
    print("  请勿关闭本窗口，关闭即停止服务。")
    print("=" * 64)


if __name__ == "__main__":
    ensure_paddle_models()
    _print_banner()

    # 2 秒后尝试打开接口文档页面（打包后双击运行更友好）
    if getattr(sys, "frozen", False):
        threading.Timer(
            2.0,
            lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/docs"),
        ).start()

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
