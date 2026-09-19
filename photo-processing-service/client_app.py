"""
证件照制作客户端 (PhotoIDClient.exe)
====================================
GUI 客户端，图像 ML 处理（抠图/OCR/人脸比对）全部调用 Java 后端 student-affair-service：
  - 人像抠图 → 客户端本地合成证件照，支持鼠标上下左右拖动微调 + 微调按钮 + 复位
  - 身份证 OCR 识别（Java 转发 Python 服务）
  - 人脸相似度比对（Java 转发 Python 服务）
  - 云端提交（登录 / 上传身份证信息 / 提交人脸采集）

调用链路：PhotoIDClient → Java(student-affair-service) → Python(photo-processing-service, Docker)

源码运行：
    1) 先启动 Java 后端（student-affair-service，默认端口 10005/5006）
    2) 确保 Java 后端已连接 Python 图像处理服务（photo-processing-service，Docker）
    3) python client_app.py
"""

import os
import re
import sys
import json
import time
import queue
import base64
import threading
import logging

import numpy as np
import cv2
import requests
import urllib3
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ========== 修复 PyInstaller --windowed 模式下 sys.stdout/stderr 为 None ==========
if getattr(sys, 'frozen', False):
    _log_dir = os.path.join(os.path.expanduser("~"), ".photo_id_tool")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, "client_runtime.log")
    if sys.stdout is None:
        sys.stdout = open(_log_path, 'a', encoding='utf-8', buffering=1)
    if sys.stderr is None:
        sys.stderr = open(_log_path, 'a', encoding='utf-8', buffering=1)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PhotoIDClient")
logging.getLogger("urllib3").setLevel(logging.WARNING)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== 配置持久化 ==========
CONFIG_DIR = os.path.expanduser("~/.photo_id_tool")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
COLLECTION_LOG_FILE = os.path.join(CONFIG_DIR, "collection_log.jsonl")


def resource_path(*parts):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)


# 内置背景图（随客户端打包，用于本地合成证件照）
DEFAULT_BG_IMAGE = resource_path("assets", "default_bg.jpg")

# ========== 云端接口（Java 后端，全部走 /admin 路径） ==========
DEFAULT_API_BASE = "https://111.12.149.164"
PATH_AUTH_LOGIN = "/admin/apiauth/auth/login"
PATH_FACE_IDENTITY = "/admin/apistudentaffair/admin/face/identity"
PATH_FACE_VERIFY = "/admin/apistudentaffair/admin/face/verify"
PATH_FACE_EXTRACT = "/admin/apistudentaffair/admin/face/extract"

# Java 后端地址（图像处理接口：抠图/OCR/人脸比对均经 Java 转发 Python 服务）
# 生产环境经 nginx 反代：/admin/apistudentaffair/** -> Java /api/**
DEFAULT_SERVER_URL = DEFAULT_API_BASE

# ========== 标准证件照尺寸（300 DPI，单位：像素） ==========
STANDARD_SIZES = {
    "一寸   (25×35mm)":  (295, 413),
    "小一寸 (22×32mm)":  (260, 378),
    "大一寸 (33×48mm)":  (390, 567),
    "二寸   (35×49mm)":  (413, 579),
    "小二寸 (35×45mm)":  (413, 531),
    "大二寸 (35×53mm)":  (413, 626),
}

# ========== 构图参数（与服务端 calc_geometry 保持一致） ==========
HEAD_TOP_MARGIN = 0.10
MAX_HEAD_HEIGHT_RATIO = 0.75
MAX_HEAD_WIDTH_RATIO = 0.90
HAIR_WIDTH_FACTOR = 1.15
HAIR_HEIGHT_FACTOR = 1.55

# ========== 预览框尺寸 ==========
PREVIEW_W, PREVIEW_H = 280, 300
PREVIEW_IMG_W, PREVIEW_IMG_H = 270, 290

# ========== 身份证号校验表 ==========
ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
ID_CHECK_MAP = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

# ========== 人脸相似度阈值 ==========
SIMILARITY_PASS = 0.50
SIMILARITY_WARN = 0.35

# 拖动微调步长（像素）
NUDGE_STEP = 2


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """读取图片：兼容 Windows 下含中文/空格等非 ASCII 字符的路径"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, FileNotFoundError):
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    if img is not None:
        return img
    if os.path.exists(path):
        return cv2.imread(path, flags)
    return None


def encode_file_base64(image_path, quality=92):
    """图片文件 -> JPEG Base64"""
    img = imread_unicode(image_path)
    if img is None:
        raise RuntimeError(f"无法读取图片：{image_path}")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("照片编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_pil_base64(pil_rgb, quality=92):
    """PIL RGB 图 -> JPEG Base64"""
    from io import BytesIO
    buf = BytesIO()
    pil_rgb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ==================== 图像处理服务客户端（调用 Java 后端） ====================
class ServerClient:
    def __init__(self, base_url):
        self.base = (base_url or DEFAULT_SERVER_URL).strip().rstrip("/")
        self.token = ""                      # ← 新增，登录成功后同步
        lowered = self.base.lower()
        if "localhost" in lowered or "127.0.0.1" in lowered:
            self._prefix = "/api/admin/photo"
        else:
            self._prefix = "/admin/apistudentaffair/admin/photo"

    def _headers(self):                      # ← 新增
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _post(self, path, payload, timeout=120):
        url = self.base + self._prefix + path
        log.info(f"[Java] POST {url} (timeout={timeout}s)")
        resp = requests.post(url, json=payload, headers=self._headers(),   # ← 加 headers
                             timeout=timeout, verify=False)
        ...
        body = resp.json()
        msg = body.get("msg") or f"服务调用失败(HTTP {resp.status_code})"
        log.warning(f"[Java] 业务失败: code={body.get('code')}, msg={msg}")
        if "未登录" in msg or "Token" in msg:
            raise RuntimeError("登录已失效，请重新登录")
        raise RuntimeError(msg)

    def ping(self):
        url = self.base + self._prefix + "/health"
        log.info(f"[Java] GET {url}")
        try:
            resp = requests.get(url, headers=self._headers(), timeout=5, verify=False)  # ← 加 headers
            body = resp.json()
            if resp.status_code == 200 and str(body.get("code")) == "200":
                return True, "已连接"
            msg = body.get("msg") or "服务异常"
            if "未登录" in msg or "Token" in msg:
                return False, "需登录"        # ← 区分"未登录"和"未连接"
            return False, msg
        except Exception as e:
            log.warning(f"[Java] 健康检查异常: {e}")
            return False, "未连接"

    def segment(self, face_b64):
        """人像抠图：返回透明 PNG + 定位元数据"""
        return self._post("/id-photo/segment",
                          {"facePhotoBase64": face_b64}, timeout=60)

    def ocr_id_card(self, id_b64):
        return self._post("/id-card/ocr",
                          {"idCardPhotoBase64": id_b64}, timeout=120)

    def compare_faces(self, id_b64, face_b64):
        return self._post("/face/compare",
                          {"idCardPhotoBase64": id_b64, "facePhotoBase64": face_b64},
                          timeout=60)


# ==================== 云端接口客户端 ====================
class ApiClient:
    """学生事务服务接口客户端：登录 / 身份证信息 / 人脸照片 / 特征提取"""

    def __init__(self):
        self.base_url = ""
        self.token = ""
        self.username = ""

    @property
    def logged_in(self):
        return bool(self.token)

    def _headers(self, with_auth=False):
        headers = {"Content-Type": "application/json"}
        if with_auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def login(self, base_url, username, password):
        self.base_url = base_url.strip().rstrip("/")
        url = f"{self.base_url}{PATH_AUTH_LOGIN}"
        payload = {
            "identifier": username,
            "password": password,
            "loginType": "WORK_CODE",
            "userType": 1,
        }
        log.info(f"[登录] 请求地址: {url}")
        log.info(f"[登录] 请求账号: {username}")
        log.info(f"[登录] 开始发送请求 (timeout=30s)...")
        try:
            resp = requests.post(url, json=payload, verify=False, timeout=30)
            log.info(f"[登录] 收到响应 HTTP {resp.status_code} (耗时 {resp.elapsed.total_seconds():.2f}s)")
        except requests.exceptions.ConnectTimeout as e:
            log.error(f"[登录] 连接超时：{e}")
            return False, f"连接超时（30秒）：{e}"
        except requests.exceptions.ReadTimeout as e:
            log.error(f"[登录] 读取超时：{e}")
            return False, f"服务器响应超时（30秒）：{e}"
        except requests.exceptions.ConnectionError as e:
            log.error(f"[登录] 连接失败：{e}")
            return False, f"无法连接服务器：{e}"
        except requests.exceptions.RequestException as e:
            log.error(f"[登录] 请求异常：{e}")
            return False, f"网络异常：{e}"
        try:
            body = resp.json()
        except ValueError:
            log.error(f"[登录] 响应非 JSON，内容前200字符: {resp.text[:200]}")
            return False, f"登录响应无法解析(HTTP {resp.status_code})"
        log.info(f"[登录] 响应体: code={body.get('code')}, msg={body.get('msg')}")
        if resp.status_code == 200 and str(body.get("code")) == "200" and body.get("data"):
            token = body["data"].get("token")
            if not token:
                log.error("[登录] 响应 data 中缺少 token")
                return False, "登录响应中缺少 token"
            self.token = token
            self.username = username
            log.info("[登录] 成功")
            return True, "登录成功"
        log.warning(f"[登录] 失败: code={body.get('code')}, msg={body.get('msg')}")
        return False, body.get("msg") or f"登录失败(HTTP {resp.status_code})"

    def save_identity(self, sfzjh, xm, xb, mz, id_card_b64):
        url = f"{self.base_url}{PATH_FACE_IDENTITY}"
        resp = requests.post(
            url, headers=self._headers(),
            json={
                "sfzjh": sfzjh, "xm": xm, "xb": xb, "mz": mz,
                "idCardPhotoBase64": id_card_b64,
                "idCardPhotoContentType": "image/jpeg",
            },
            verify=False, timeout=60,
        )
        return self._parse_result(resp, "身份证信息上传失败")

    def verify_face(self, sfzjh, photo_b64):
        url = f"{self.base_url}{PATH_FACE_VERIFY}"
        resp = requests.post(
            url, headers=self._headers(),
            json={"sfzjh": sfzjh, "photoBase64": photo_b64},
            verify=False, timeout=60,
        )
        return self._parse_result(resp, "人脸照片上传失败")

    def extract_feature(self, sfzjh):
        url = f"{self.base_url}{PATH_FACE_EXTRACT}"
        resp = requests.post(
            url, headers=self._headers(with_auth=True), params={"sfzjh": sfzjh},
            verify=False, timeout=30,
        )
        try:
            body = resp.json()
        except ValueError:
            resp.raise_for_status()
            return ""
        if resp.status_code == 200 and str(body.get("code")) == "200":
            return body.get("msg", "任务已提交")
        raise RuntimeError(body.get("msg") or f"特征提取任务提交失败(HTTP {resp.status_code})")

    @staticmethod
    def _parse_result(resp, err_prefix):
        try:
            body = resp.json()
        except ValueError:
            resp.raise_for_status()
            return {}
        if resp.status_code == 200 and str(body.get("code")) == "200":
            return body.get("data") or {}
        msg = body.get("msg") or f"接口调用失败(HTTP {resp.status_code})"
        raise RuntimeError(f"{err_prefix}：{msg}")


# ==================== 摄像头采集对话框 ====================
class CameraDialog(tk.Toplevel):
    def __init__(self, app, title, on_capture, hint="", on_close=None):
        super().__init__(app.root)
        self.app = app
        self.on_capture = on_capture
        self.on_close = on_close
        self.cap = None
        self.running = True
        self._last_frame = None
        self._camera_index = 0

        self.title(title)
        self.resizable(False, False)
        self.transient(app.root)
        self.grab_set()

        if hint:
            ttk.Label(self, text=hint, foreground="#333",
                      font=("Helvetica", 11), justify="center"
                      ).pack(pady=(12, 4), padx=12)

        self.video_label = tk.Label(self, bg="black", width=80, height=30)
        self.video_label.pack(padx=12, pady=6)

        self.status_label = ttk.Label(self, text="正在打开摄像头...", foreground="gray")
        self.status_label.pack(pady=2)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        self.capture_btn = ttk.Button(btn_frame, text="拍照（空格）", command=self._capture,
                                      width=16, state="disabled")
        self.capture_btn.pack(side="left", padx=6)
        ttk.Button(btn_frame, text="取消（Esc）", command=self._close, width=14
                   ).pack(side="left", padx=6)

        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<space>", lambda e: self._capture())
        self.bind("<Return>", lambda e: self._capture())
        self.bind("<Escape>", lambda e: self._close())

        self.update_idletasks()
        mx, my = app.root.winfo_rootx(), app.root.winfo_rooty()
        mw, mh = app.root.winfo_width(), app.root.winfo_height()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{mx + max(0, (mw - w) // 2)}+{my + max(0, (mh - h) // 2)}")
        self.after(50, self._open_camera)

    def _open_camera(self):
        if not self.running:
            return
        last_err = None
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            try:
                cap = cv2.VideoCapture(self._camera_index, backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        self.cap = cap
                        break
                    cap.release()
            except Exception as e:
                last_err = e
                continue

        if self.cap is None:
            msg = "无法打开摄像头，请检查设备连接或驱动"
            if last_err:
                msg += f"\n（{last_err}）"
            log.error(f"[CameraDialog] {msg}")
            self.status_label.config(text=msg, foreground="red")
            messagebox.showerror("摄像头错误", msg, parent=self)
            return

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.status_label.config(text="摄像头就绪，按空格或点击“拍照”", foreground="green")
        self.capture_btn.config(state="normal")
        self._update_frame()

    def _update_frame(self):
        if not self.running or self.cap is None:
            return
        try:
            ret, frame = self.cap.read()
        except Exception:
            ret, frame = False, None
        if ret and frame is not None:
            self._last_frame = frame.copy()
            preview_frame = cv2.flip(frame, 1)
            h, w = preview_frame.shape[:2]
            max_w, max_h = 720, 540
            scale = min(max_w / w, max_h / h, 1.0)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            preview = cv2.resize(preview_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.config(image=photo, width=new_w, height=new_h)
            self.video_label.image = photo
        self.after(33, self._update_frame)

    def _capture(self):
        if self._last_frame is None or self.cap is None:
            return
        frame = self._last_frame.copy()
        callback = self.on_capture
        self._close()
        try:
            callback(frame)
        except Exception as e:
            log.exception("[CameraDialog] 拍照回调异常")
            messagebox.showerror("错误", f"处理拍照结果失败：{e}")

    def _close(self):
        if not self.running:
            return
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        cb = self.on_close
        try:
            self.destroy()
        except Exception:
            pass
        if cb:
            try:
                cb()
            except Exception:
                pass


# ==================== 登录弹窗 ====================
class LoginDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("登录")
        self.resizable(False, False)
        self.transient(app.root)
        self.grab_set()

        ttk.Label(self, text="云端账号登录", font=("Helvetica", 13, "bold")
                  ).pack(pady=(16, 8), padx=18, anchor="w")

        frame = ttk.Frame(self, padding=(18, 0, 18, 18))
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="API地址：").grid(row=0, column=0, sticky="e", pady=4)
        self.base_entry = ttk.Entry(frame, width=30)
        self.base_entry.grid(row=0, column=1, pady=4)
        self.base_entry.insert(0, app.config.get("api_base_url", DEFAULT_API_BASE))

        ttk.Label(frame, text="账号：").grid(row=1, column=0, sticky="e", pady=4)
        self.user_entry = ttk.Entry(frame, width=30)
        self.user_entry.grid(row=1, column=1, pady=4)
        self.user_entry.insert(0, app.config.get("api_username", "admin"))

        ttk.Label(frame, text="密码：").grid(row=2, column=0, sticky="e", pady=4)
        self.pwd_entry = ttk.Entry(frame, width=30, show="*")
        self.pwd_entry.grid(row=2, column=1, pady=4)
        self.pwd_entry.insert(0, app.config.get("api_password", ""))

        self.tip_label = ttk.Label(frame, text="", foreground="red")
        self.tip_label.grid(row=3, column=0, columnspan=2, pady=(6, 2))

        ttk.Button(frame, text="登录", width=14, command=self.do_login
                   ).grid(row=4, column=0, columnspan=2, pady=(6, 0))

        self.bind("<Return>", lambda e: self.do_login())
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.update_idletasks()
        mx, my = app.root.winfo_rootx(), app.root.winfo_rooty()
        mw, mh = app.root.winfo_width(), app.root.winfo_height()
        self.geometry(f"360x220+{mx + (mw - 360) // 2}+{my + (mh - 220) // 2}")

    def do_login(self):
        base_url = self.base_entry.get().strip()
        username = self.user_entry.get().strip()
        password = self.pwd_entry.get().strip()
        if not base_url or not username or not password:
            self.tip_label.config(text="请填写 API地址、账号和密码", foreground="red")
            return
        self.tip_label.config(text="正在登录，请稍候...", foreground="gray")

        dialog = self  # 闭包引用

        def worker():
            log.info(f"[登录] worker 线程启动，base_url={base_url}, username={username}")
            try:
                ok, msg = self.app.api_client.login(base_url, username, password)
                log.info(f"[登录] worker 返回 ok={ok}, msg={msg}")
            except Exception as e:
                log.exception("[登录] worker 抛出未捕获异常")
                ok, msg = False, f"登录异常：{e}"
            # 切回主线程：投递到 msg_queue，由 _poll_queue 在主线程消费
            self.app.msg_queue.put(
                ("login_done", (dialog, ok, msg, base_url, username, password))
            )

        threading.Thread(target=worker, daemon=True).start()

    def _done(self, ok, msg, base_url, username, password):
        if not self.winfo_exists():
            return
        try:
            if ok:
                self.app.config["api_base_url"] = base_url
                self.app.config["api_username"] = username
                self.app.config["api_password"] = password
                save_config(self.app.config)
                self.app.login_status_label.config(text=f"已登录：{username}", foreground="green")
                self.app._update_flow()
                self.app.set_status("云端登录成功")
                self.destroy()
            else:
                self.tip_label.config(text=msg, foreground="red")
        except Exception:
            log.exception("[登录] _done 处理异常")
            try:
                self.destroy()
            except Exception:
                pass


# ==================== 主应用 ====================
class PhotoIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("证件照制作工具（客户端）")
        self.root.geometry("1180x920")
        self.root.minsize(1080, 860)

        self.config = load_config()
        # 图像处理接口与云端提交共用同一个 Java 后端，server_url 未单独配置时跟随 api_base_url
        server_url = self.config.get("server_url") or self.config.get("api_base_url") or DEFAULT_SERVER_URL
        self.server = ServerClient(server_url)

        self.face_path = None
        self.id_path = None
        self.id_info = {}

        # --- 抠图结果与拖动状态 ---
        self.seg_fg = None          # PIL RGBA（原始分辨率）
        self.seg_meta = None        # src 尺寸 / 人脸框 / 人像上下界
        self.drag_dx = 0.0          # 用户水平偏移（照片像素）
        self.drag_dy = 0.0          # 用户垂直偏移（照片像素）
        self._geometry = None       # 当前规格的几何参数
        self._bg_cache = {}         # target 尺寸 -> PIL RGBA 背景
        self._preview_photo = None  # 防止 PhotoImage 被 GC
        self._preview_scale = 1.0   # 预览图显示尺寸 / 照片尺寸
        self._dragging = False
        self._drag_start = None

        self.face_processed = False
        self.processed_pil = None   # 当前合成结果（含拖动偏移，全分辨率）

        self.pending_id_recognition = False
        self._suppress_validate = False

        self.msg_queue = queue.Queue()

        self.api_client = ApiClient()
        self.login_dialog = None
        self.camera_dialog = None

        self.identity_sfzjh = None
        self.identity_ksbs = None
        self.identity_mismatched = []
        self.just_submitted = False

        self.similarity_result = None    # (sim, msg)
        self._similarity_running = False

        self._build_ui()
        self._restore_config()
        self._poll_queue()
        self._check_server_async()

    # ==================== UI ====================
    def _build_ui(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=12, pady=(8, 2))
        ttk.Label(header, text="证件照制作工具", font=("Helvetica", 16, "bold")).pack(side="left")

        header_right = ttk.Frame(header)
        header_right.pack(side="right")
        self.server_status_label = ttk.Label(header_right, text="服务检测中...", foreground="gray")
        self.server_status_label.pack(side="left", padx=(0, 8))
        ttk.Button(header_right, text="检测服务", width=10,
                   command=self._check_server_async).pack(side="left")
        self.login_status_label = ttk.Label(header_right, text="未登录", foreground="gray")
        self.login_status_label.pack(side="left", padx=(8, 8))
        ttk.Button(header_right, text="登录", width=10,
                   command=self.open_login_dialog).pack(side="left")

        # ---------- 文件选择 ----------
        file_frame = ttk.LabelFrame(self.root, text="文件选择", padding=8)
        file_frame.pack(fill="x", padx=12, pady=4)

        row1 = ttk.Frame(file_frame)
        row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="人脸照片：", width=12, anchor="w").pack(side="left")
        self.face_label = ttk.Label(row1, text="未选择", foreground="gray", anchor="w")
        self.face_label.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row1, text="选择文件", command=self.select_face).pack(side="right")
        ttk.Button(row1, text="拍照", command=self.open_camera_for_face, width=8
                   ).pack(side="right", padx=(0, 5))

        row2 = ttk.Frame(file_frame)
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="身份证照片：", width=12, anchor="w").pack(side="left")
        self.id_label = ttk.Label(row2, text="未选择（可选，选后自动识别）", foreground="gray", anchor="w")
        self.id_label.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row2, text="选择文件", command=self.select_id).pack(side="right")
        ttk.Button(row2, text="拍照", command=self.open_camera_for_id, width=8
                   ).pack(side="right", padx=(0, 5))

        # ---------- 参数 ----------
        opt_frame = ttk.LabelFrame(self.root, text="证件照参数", padding=8)
        opt_frame.pack(fill="x", padx=12, pady=4)

        size_row = ttk.Frame(opt_frame)
        size_row.pack(fill="x", pady=2)
        ttk.Label(size_row, text="规格：", width=8, anchor="w").pack(side="left")
        self.size_var = tk.StringVar(value=list(STANDARD_SIZES.keys())[0])
        size_combo = ttk.Combobox(size_row, textvariable=self.size_var,
                                  values=list(STANDARD_SIZES.keys()),
                                  state="readonly", width=22)
        size_combo.pack(side="left", padx=5)
        size_combo.bind("<<ComboboxSelected>>", lambda e: self._on_size_change())
        ttk.Label(size_row, text="背景：内置背景图（固定，不可更换）",
                  foreground="gray").pack(side="left", padx=15)

        # ---------- 采集流程 ----------
        flow_frame = ttk.LabelFrame(
            self.root,
            text="采集流程（单人单次：身份证确认身份 → 人脸采集 → 自动复位）",
            padding=8)
        flow_frame.pack(fill="x", padx=12, pady=4)

        steps_row = ttk.Frame(flow_frame)
        steps_row.pack(fill="x")
        self.flow_labels = {}
        self.flow_steps = [
            (1, "① 身份证识别"),
            (2, "② 身份确认(ksbs)"),
            (3, "③ 人脸照片处理"),
            (4, "④ 提交完成"),
        ]
        for i, (_, text) in enumerate(self.flow_steps):
            if i > 0:
                ttk.Label(steps_row, text="→", font=("Helvetica", 11)).pack(side="left", padx=6)
            lbl = ttk.Label(steps_row, text="○ " + text, font=("Helvetica", 11), foreground="gray")
            lbl.pack(side="left", padx=4)
            self.flow_labels[i + 1] = lbl

        self.locked_student_label = ttk.Label(
            flow_frame, text="当前锁定考生：无（请先识别并上传身份证信息）",
            font=("Helvetica", 11, "bold"), foreground="gray")
        self.locked_student_label.pack(anchor="w", pady=(6, 0))

        self.similarity_label = ttk.Label(
            flow_frame, text="人脸相似度比对：等待照片...",
            font=("Helvetica", 11, "bold"), foreground="gray")
        self.similarity_label.pack(anchor="w", pady=(4, 0))

        # ---------- 按钮区 ----------
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=12, pady=6)

        self.start_btn = ttk.Button(btn_frame, text="开始处理", command=self.start_process,
                                    state="disabled", width=14)
        self.start_btn.pack(side="left", padx=5)

        self.save_btn = ttk.Button(btn_frame, text="保存照片", command=self.save_photo,
                                   state="disabled", width=14)
        self.save_btn.pack(side="left", padx=5)

        self.identity_btn = ttk.Button(btn_frame, text="上传身份证信息",
                                       command=self.upload_identity, state="disabled", width=16)
        self.identity_btn.pack(side="left", padx=5)

        self.face_submit_btn = ttk.Button(btn_frame, text="提交人脸采集",
                                          command=self.submit_face, state="disabled", width=16)
        self.face_submit_btn.pack(side="left", padx=5)

        ttk.Button(btn_frame, text="退出", command=self.root.quit, width=10
                   ).pack(side="right", padx=5)

        # ---------- 预览 ----------
        preview_frame = ttk.LabelFrame(self.root, text="预览", padding=8)
        preview_frame.pack(fill="x", padx=12, pady=4)

        toggle_row = ttk.Frame(preview_frame)
        toggle_row.pack(fill="x", pady=(0, 6))
        self.mode_var = tk.StringVar(value="face")
        ttk.Radiobutton(toggle_row, text="人脸模式", value="face",
                        variable=self.mode_var, command=self._switch_preview_mode
                        ).pack(side="left", padx=5)
        ttk.Radiobutton(toggle_row, text="身份证模式", value="id",
                        variable=self.mode_var, command=self._switch_preview_mode
                        ).pack(side="left", padx=5)

        self.preview_container = ttk.Frame(preview_frame)
        self.preview_container.pack(fill="x")

        # ====== 模式 A：人脸 ======
        self.preview_face_frame = ttk.Frame(self.preview_container)

        left_face = ttk.Frame(self.preview_face_frame)
        left_face.pack(side="left", padx=15)
        ttk.Label(left_face, text="原始照片", font=("Helvetica", 11)).pack()

        face_box = tk.Frame(left_face, width=PREVIEW_W, height=PREVIEW_H,
                            background="#f5f5f5", relief="solid", borderwidth=1)
        face_box.pack(pady=4)
        face_box.pack_propagate(False)
        self.face_preview = tk.Label(face_box, background="#f5f5f5", text="无预览")
        self.face_preview.pack(fill="both", expand=True)

        right_face = ttk.Frame(self.preview_face_frame)
        right_face.pack(side="left", padx=15)
        ttk.Label(right_face, text="证件照预览（按住鼠标拖动调整位置）",
                  font=("Helvetica", 11)).pack()

        result_box = tk.Frame(right_face, width=PREVIEW_W, height=PREVIEW_H,
                              background="#f5f5f5", relief="solid", borderwidth=1)
        result_box.pack(pady=4)
        result_box.pack_propagate(False)
        self.result_preview = tk.Label(result_box, background="#f5f5f5",
                                       text="无预览", cursor="fleur")
        self.result_preview.pack(fill="both", expand=True)

        # 拖动微调控件
        adjust_row = ttk.Frame(right_face)
        adjust_row.pack(pady=(0, 4))
        ttk.Button(adjust_row, text="←", width=4, command=lambda: self._nudge(-NUDGE_STEP, 0)
                   ).pack(side="left", padx=2)
        ttk.Button(adjust_row, text="↑", width=4, command=lambda: self._nudge(0, -NUDGE_STEP)
                   ).pack(side="left", padx=2)
        ttk.Button(adjust_row, text="↓", width=4, command=lambda: self._nudge(0, NUDGE_STEP)
                   ).pack(side="left", padx=2)
        ttk.Button(adjust_row, text="→", width=4, command=lambda: self._nudge(NUDGE_STEP, 0)
                   ).pack(side="left", padx=2)
        ttk.Button(adjust_row, text="复位位置", width=9, command=self._reset_offset
                   ).pack(side="left", padx=(8, 2))

        self.offset_label = ttk.Label(right_face, text="偏移：0, 0", foreground="gray",
                                      font=("Helvetica", 9))
        self.offset_label.pack()

        # ====== 模式 B：身份证 ======
        self.preview_id_frame = ttk.Frame(self.preview_container)
        self.preview_id_frame.grid_columnconfigure(1, weight=1)

        left_id = ttk.Frame(self.preview_id_frame)
        left_id.grid(row=0, column=0, padx=(15, 10), sticky="n")
        ttk.Label(left_id, text="身份证照片", font=("Helvetica", 11)).pack()

        id_photo_box = tk.Frame(left_id, width=PREVIEW_W, height=PREVIEW_H,
                                background="#f5f5f5", relief="solid", borderwidth=1)
        id_photo_box.pack(pady=4)
        id_photo_box.pack_propagate(False)
        self.id_preview = tk.Label(id_photo_box, background="#f5f5f5", text="无预览")
        self.id_preview.pack(fill="both", expand=True)

        right_id = ttk.Frame(self.preview_id_frame)
        right_id.grid(row=0, column=1, padx=(10, 15), sticky="nsew")
        ttk.Label(right_id, text="身份证信息（可编辑）", font=("Helvetica", 11)).pack(anchor="w")

        info_box = tk.Frame(right_id, background="white", relief="solid", borderwidth=1)
        info_box.pack(fill="both", expand=True, pady=4)

        inner = tk.Frame(info_box, background="white", padx=10, pady=10)
        inner.pack(fill="both", expand=True)
        inner.grid_columnconfigure(1, weight=1)

        font_field = ("Helvetica", 11)
        self.id_entries = {}
        self.warn_labels = {}
        fields = [
            ("name", "姓名", "entry", 1),
            ("gender", "性别", "entry", 1),
            ("ethnicity", "民族", "entry", 1),
            ("birth", "出生", "entry", 1),
            ("address", "住址", "text", 3),
            ("id_number", "身份证号", "entry", 1),
        ]
        for i, (key, label_text, wtype, h) in enumerate(fields):
            tk.Label(inner, text=label_text + "：", background="white",
                     font=font_field, anchor="w", width=8
                     ).grid(row=i, column=0, sticky="nw", pady=3)
            if wtype == "entry":
                entry = tk.Entry(inner, font=font_field)
                entry.grid(row=i, column=1, sticky="ew", padx=(0, 8), pady=3)
            else:
                entry = tk.Text(inner, font=font_field, height=h, wrap="word",
                                relief="solid", borderwidth=1)
                entry.grid(row=i, column=1, sticky="ew", padx=(0, 8), pady=3)
            entry.bind("<KeyRelease>", lambda e: self._validate_id_consistency())
            entry.bind("<FocusOut>", self._on_id_field_focusout)
            self.id_entries[key] = entry

            warn = tk.Label(inner, text="", background="white",
                            font=("Helvetica", 10), foreground="red",
                            anchor="w", width=22)
            warn.grid(row=i, column=2, sticky="nw", pady=3)
            self.warn_labels[key] = warn

        bottom = tk.Frame(inner, background="white")
        bottom.grid(row=len(fields), column=0, columnspan=3, sticky="ew", pady=(10, 0))
        bottom.grid_columnconfigure(0, weight=1)
        self.id_status_label = tk.Label(bottom, text="", background="white",
                                        font=("Helvetica", 10), anchor="w", foreground="gray")
        self.id_status_label.grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="按身份证号纠正", command=self._apply_id_card_info
                   ).grid(row=0, column=1, sticky="e")

        self.preview_face_frame.pack()

        self.status = ttk.Label(self.root, text="正在准备中，请稍候...",
                                relief="sunken", anchor="w", padding=5)
        self.status.pack(fill="x", side="bottom")

        # ---------- 拖动事件绑定 ----------
        self.result_preview.bind("<ButtonPress-1>", self._on_drag_press)
        self.result_preview.bind("<B1-Motion>", self._on_drag_motion)
        self.result_preview.bind("<ButtonRelease-1>", self._on_drag_release)
        # 方向键微调（绑在根窗口，处理中时才生效）
        self.root.bind("<Left>", lambda e: self._nudge(-NUDGE_STEP, 0))
        self.root.bind("<Right>", lambda e: self._nudge(NUDGE_STEP, 0))
        self.root.bind("<Up>", lambda e: self._nudge(0, -NUDGE_STEP))
        self.root.bind("<Down>", lambda e: self._nudge(0, NUDGE_STEP))

    # ==================== 字段读写 ====================
    def _get_field(self, key):
        widget = self.id_entries[key]
        if isinstance(widget, tk.Text):
            val = widget.get("1.0", "end").strip()
            return val.replace("\n", "").replace(" ", "")
        return widget.get().strip()

    def _set_field(self, key, value):
        widget = self.id_entries[key]
        value = value or ""
        if isinstance(widget, tk.Text):
            widget.delete("1.0", "end")
            widget.insert("1.0", value)
        else:
            widget.delete(0, "end")
            widget.insert(0, value)

    # ==================== 预览模式切换 ====================
    def _switch_preview_mode(self):
        mode = self.mode_var.get()
        if mode == "face":
            self.preview_id_frame.pack_forget()
            self.preview_face_frame.pack()
        else:
            self.preview_face_frame.pack_forget()
            self.preview_id_frame.pack(fill="x")

    def _show_face_mode(self):
        self.mode_var.set("face")
        self._switch_preview_mode()

    def _show_id_mode(self):
        self.mode_var.set("id")
        self._switch_preview_mode()

    # ==================== 配置 ====================
    def _restore_config(self):
        size_name = self.config.get("size_name")
        if size_name and size_name in STANDARD_SIZES:
            self.size_var.set(size_name)

    def _save_size_config(self):
        self.config["size_name"] = self.size_var.get()
        save_config(self.config)

    # ==================== 线程通信 ====================
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "status":
                    self.status.config(text=payload)
                elif kind == "error":
                    messagebox.showerror("错误", payload)
                elif kind == "info":
                    messagebox.showinfo("提示", payload)
                elif kind == "id_info":
                    self._update_id_info_display(payload)
                elif kind == "similarity":
                    self._on_similarity_done(*payload)
                elif kind == "segment_done":
                    self._on_segment_done(payload)
                elif kind == "server_status":
                    ok, msg = payload
                    self.server_status_label.config(
                        text=f"服务：{msg}",
                        foreground="green" if ok else "#cc6600")
                elif kind == "ask_reset_with_text":
                    if messagebox.askyesno("采集完成", payload):
                        self._reset_for_next()
                    else:
                        self._update_flow()
                elif kind == "login_done":
                    dialog, ok, msg, base_url, username, password = payload
                    if ok:
                        self.server.token = self.api_client.token   # ← 关键一行
                    try:
                        if dialog.winfo_exists():
                            dialog._done(ok, msg, base_url, username, password)
                    except Exception:
                        log.exception("[登录] 处理 login_done 回调异常")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def set_status(self, text):
        self.msg_queue.put(("status", text))

    def show_error(self, msg):
        self.msg_queue.put(("error", msg))

    def show_info(self, msg):
        self.msg_queue.put(("info", msg))

    # ==================== 服务状态 ====================
    def _check_server_async(self):
        def worker():
            ok, msg = self.server.ping()
            self.msg_queue.put(("server_status", (ok, msg)))
        threading.Thread(target=worker, daemon=True).start()

    # ==================== 文件选择 & 摄像头 ====================
    def _pick_file(self, title):
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")])

    def select_face(self):
        path = self._pick_file("请选择人脸照片")
        if not path:
            return
        self._apply_face_photo(os.path.normpath(path))

    def select_id(self):
        path = self._pick_file("请选择身份证照片")
        if not path:
            return
        self._apply_id_photo(os.path.normpath(path))

    def open_camera_for_face(self):
        self._open_camera(title="摄像头采集 - 人脸照片",
                          hint="请正对镜头，露出五官，光线均匀，画面稳定后再拍照",
                          kind="face")

    def open_camera_for_id(self):
        self._open_camera(title="摄像头采集 - 身份证照片",
                          hint="请将身份证平放于桌面或手持对准镜头\n保证四角完整、无遮挡、无反光、文字清晰",
                          kind="id")

    def _open_camera(self, title, hint, kind):
        if self.camera_dialog is not None:
            try:
                if self.camera_dialog.winfo_exists():
                    self.camera_dialog.lift()
                    self.camera_dialog.focus_set()
                    return
            except Exception:
                pass
            self.camera_dialog = None

        self.camera_dialog = CameraDialog(
            self, title=title, hint=hint,
            on_capture=lambda f: self._on_camera_capture(f, kind),
            on_close=self._clear_camera_dialog)

    def _clear_camera_dialog(self):
        self.camera_dialog = None

    def _on_camera_capture(self, frame_bgr, kind):
        try:
            cap_dir = os.path.join(CONFIG_DIR, "captures")
            os.makedirs(cap_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(cap_dir, f"{kind}_{ts}.jpg")
            ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                self.show_error("摄像头照片保存失败")
                return
            buf.tofile(path)
            log.info(f"[摄像头] 已保存 {kind} 照片：{path}")
            if kind == "face":
                self._apply_face_photo(path, source="摄像头")
            else:
                self._apply_id_photo(path, source="摄像头")
        except Exception as e:
            log.exception("[摄像头] 保存照片失败")
            self.show_error(f"摄像头照片处理失败：{e}")

    # ==================== 应用照片 ====================
    def _apply_face_photo(self, path, source="文件"):
        self.face_path = path
        self.face_label.config(text=path, foreground="black")
        self._show_preview_file(path, self.face_preview)
        self._show_face_mode()
        self._invalidate_face_result()
        self._auto_compare_faces()
        self.set_status(f"已载入人脸照片（{source}）：{os.path.basename(path)}")

    def _apply_id_photo(self, path, source="文件"):
        self.id_path = path
        self.id_label.config(text=path, foreground="black")
        self._show_preview_file(path, self.id_preview)
        self._show_id_mode()

        self.identity_sfzjh = None
        self.identity_ksbs = None
        self.identity_mismatched = []
        self.just_submitted = False
        self._invalidate_face_result()
        self._update_flow()
        self._auto_recognize_id()
        self._auto_compare_faces()
        self.set_status(f"已载入身份证照片（{source}）：{os.path.basename(path)}")

    def _invalidate_face_result(self):
        self.seg_fg = None
        self.seg_meta = None
        self._geometry = None
        self.drag_dx = 0.0
        self.drag_dy = 0.0
        self.processed_pil = None
        self.face_processed = False
        self.just_submitted = False
        self.result_preview.config(image="", text="无预览")
        self.result_preview.image = None
        self.offset_label.config(text="偏移：0, 0")
        self.save_btn.config(state="disabled")
        self._update_flow()

    def _show_preview_file(self, image_path, label_widget):
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((PREVIEW_IMG_W, PREVIEW_IMG_H))
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            label_widget.config(text=f"预览失败: {e}", image="")

    # ==================== 身份证识别（服务端 OCR） ====================
    def _auto_recognize_id(self):
        if not self.id_path:
            return
        ok, _ = self.server.ping()
        if not ok:
            self.set_status("图像处理服务未连接，无法识别身份证")
            self.show_error("Java 后端服务未连接。\n请确认网络正常，并点击右上角“检测服务”。")
            return
        self.set_status("正在识别身份证...")

        def worker():
            try:
                b64 = encode_file_base64(self.id_path)
                data = self.server.ocr_id_card(b64)
                info = {
                    "name": data.get("name", ""),
                    "gender": data.get("gender", ""),
                    "ethnicity": data.get("ethnicity", ""),
                    "birth": data.get("birth", ""),
                    "address": data.get("address", ""),
                    "id_number": data.get("idNumber", ""),
                    "raw_lines": data.get("rawLines", []),
                }
                self.id_info = info
                self.msg_queue.put(("id_info", info))
                self.set_status("身份证识别完成")
            except Exception as e:
                self.set_status("身份证识别失败")
                self.show_error(f"身份证识别失败：{e}")

        threading.Thread(target=worker, daemon=True).start()

    def _update_id_info_display(self, info):
        self._suppress_validate = True
        try:
            for key in ["name", "gender", "ethnicity", "birth", "address", "id_number"]:
                self._set_field(key, info.get(key, "") or "")
        finally:
            self._suppress_validate = False
        self._validate_id_consistency()
        self._update_flow()

    # ==================== 身份证号解析（本地） ====================
    @staticmethod
    def _parse_id_card_info(id_number):
        result = {"birth": "", "gender": "", "valid": False, "msg": ""}
        s = (id_number or "").strip().upper()
        if len(s) != 18:
            result["msg"] = "身份证号必须为18位"
            return result
        if not re.match(r"^\d{17}[\dX]$", s):
            result["msg"] = "身份证号格式错误（含无效字符）"
            return result
        try:
            total = sum(int(s[i]) * ID_WEIGHTS[i] for i in range(17))
            expected = ID_CHECK_MAP[total % 11]
            if s[17] != expected:
                result["msg"] = f"校验位错误（应为 {expected}）"
                return result
        except ValueError:
            result["msg"] = "身份证号含无效字符"
            return result
        try:
            year, month, day = int(s[6:10]), int(s[10:12]), int(s[12:14])
            if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
                result["msg"] = "出生日期不合法"
                return result
            result["birth"] = f"{year}年{month}月{day}日"
        except ValueError:
            result["msg"] = "出生日期不合法"
            return result
        try:
            gender_digit = int(s[16])
            result["gender"] = "男" if gender_digit % 2 == 1 else "女"
        except ValueError:
            result["msg"] = "性别位不合法"
            return result
        result["valid"] = True
        return result

    def _on_id_field_focusout(self, _event=None):
        self._validate_id_consistency()
        self._update_flow()

    def _validate_id_consistency(self):
        if self._suppress_validate:
            return
        id_number = self._get_field("id_number")
        for warn in self.warn_labels.values():
            warn.config(text="")
        if not id_number:
            self.id_status_label.config(text="", foreground="gray")
            return
        if len(id_number) < 18:
            self.id_status_label.config(
                text=f"身份证号还差 {18 - len(id_number)} 位", foreground="gray")
            return

        parsed = self._parse_id_card_info(id_number)
        if not parsed["valid"]:
            self.id_status_label.config(text=f"⚠ {parsed['msg']}", foreground="red")
            return
        self.id_status_label.config(text="✓ 身份证号校验通过", foreground="green")

        ocr_gender = self._get_field("gender")
        if parsed["gender"] and ocr_gender and ocr_gender != parsed["gender"]:
            self.warn_labels["gender"].config(text=f"应为「{parsed['gender']}」")
        elif parsed["gender"] and not ocr_gender:
            self.warn_labels["gender"].config(text=f"（{parsed['gender']}）")

        ocr_birth = self._get_field("birth")

        def _norm(s):
            return re.sub(r"\D", "", s)

        if parsed["birth"] and ocr_birth:
            if _norm(ocr_birth) != _norm(parsed["birth"]):
                self.warn_labels["birth"].config(text=f"应为「{parsed['birth']}」")
        elif parsed["birth"] and not ocr_birth:
            self.warn_labels["birth"].config(text=f"（{parsed['birth']}）")

    def _apply_id_card_info(self):
        id_number = self._get_field("id_number")
        if not id_number:
            self.show_error("请先输入身份证号")
            return
        parsed = self._parse_id_card_info(id_number)
        if not parsed["valid"]:
            self.show_error(f"身份证号校验失败：{parsed['msg']}")
            return
        self._suppress_validate = True
        try:
            if parsed["birth"]:
                self._set_field("birth", parsed["birth"])
            if parsed["gender"]:
                self._set_field("gender", parsed["gender"])
        finally:
            self._suppress_validate = False
        self._validate_id_consistency()
        self.set_status("已按身份证号纠正出生日期和性别")

    # ==================== 人脸相似度比对（服务端） ====================
    def _auto_compare_faces(self):
        self.similarity_result = None
        if not self.face_path or not self.id_path:
            self.similarity_label.config(
                text="人脸相似度比对：等待照片...", foreground="gray")
            return
        if self._similarity_running:
            return
        self._similarity_running = True
        self.similarity_label.config(text="人脸相似度比对：正在计算...", foreground="blue")

        def worker():
            sim, msg = None, "比对异常"
            try:
                id_b64 = encode_file_base64(self.id_path)
                face_b64 = encode_file_base64(self.face_path)
                data = self.server.compare_faces(id_b64, face_b64)
                sim, msg = data.get("similarity"), data.get("msg", "ok")
            except Exception as e:
                log.exception("[比对] 异常")
                sim, msg = None, f"比对异常：{e}"
            self._similarity_running = False
            self.msg_queue.put(("similarity", (sim, msg)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_similarity_done(self, sim, msg):
        self.similarity_result = (sim, msg)
        if sim is None:
            self.similarity_label.config(
                text=f"人脸相似度比对：⚠ {msg}", foreground="#cc6600")
            return
        pct = f"{sim * 100:.1f}%"
        if sim >= SIMILARITY_PASS:
            self.similarity_label.config(
                text=f"人脸相似度比对：{pct}（✓ 通过）", foreground="green")
        elif sim >= SIMILARITY_WARN:
            self.similarity_label.config(
                text=f"人脸相似度比对：{pct}（⚠ 偏低，请人工核对）", foreground="#cc6600")
        else:
            self.similarity_label.config(
                text=f"人脸相似度比对：{pct}（✗ 疑似非同一人）", foreground="red")

    # ==================== 规格 ====================
    def _on_size_change(self):
        self._save_size_config()
        if self.seg_fg is not None:
            self._compute_geometry()
            self._render_result()

    # ==================== 核心处理：抠图 + 本地合成 ====================
    def start_process(self):
        if not self.face_path:
            self.show_error("请先选择人脸照片")
            return
        ok, _ = self.server.ping()
        if not ok:
            self.show_error("Java 后端服务未连接。\n请确认网络正常，并点击右上角“检测服务”。")
            return

        self.start_btn.config(state="disabled", text="处理中...")
        self.save_btn.config(state="disabled")
        self.face_processed = False
        self._update_flow()
        self.set_status("正在抠图处理...")

        def worker():
            try:
                b64 = encode_file_base64(self.face_path)
                data = self.server.segment(b64)
                self.msg_queue.put(("segment_done", data))
            except Exception as e:
                log.exception("[抠图] 异常")
                self.msg_queue.put(("error", f"处理失败：{e}"))
                self.msg_queue.put(("segment_done", None))

        threading.Thread(target=worker, daemon=True).start()

    def _on_segment_done(self, data):
        self.start_btn.config(state="normal", text="开始处理")
        if not data:
            self.set_status("处理失败")
            return
        try:
            from io import BytesIO
            fg = Image.open(BytesIO(base64.b64decode(data["fgPngBase64"]))).convert("RGBA")
            self.seg_fg = fg
            self.seg_meta = {
                "src_w": data["srcWidth"], "src_h": data["srcHeight"],
                "face": (data["faceX"], data["faceY"], data["faceW"], data["faceH"]),
                "person_top": data["personTop"],
                "person_bottom": data["personBottom"],
            }
            self.drag_dx = 0.0
            self.drag_dy = 0.0
            self._compute_geometry()
            self._render_result()
            self.face_processed = True
            self.save_btn.config(state="normal")
            self.set_status("1寸照片处理完成，可拖动调整位置后保存或提交")
        except Exception as e:
            log.exception("[合成] 异常")
            self.show_error(f"照片合成失败：{e}")
            self.set_status("处理失败")
        self._update_flow()

    def _compute_geometry(self):
        """与服务端 calc_geometry 相同的构图计算"""
        if not self.seg_meta:
            return
        tw, th = STANDARD_SIZES[self.size_var.get()]
        src_w, src_h = self.seg_meta["src_w"], self.seg_meta["src_h"]
        fx, fy, fw, fh = self.seg_meta["face"]
        person_top = self.seg_meta["person_top"]
        person_span = max(1, self.seg_meta["person_bottom"] - person_top)

        scale_fill = (th * (1.0 - HEAD_TOP_MARGIN)) / person_span
        scale_cap_h = (th * MAX_HEAD_HEIGHT_RATIO) / (fh * HAIR_HEIGHT_FACTOR)
        scale_cap_w = (tw * MAX_HEAD_WIDTH_RATIO) / (fw * HAIR_WIDTH_FACTOR)
        scale = min(scale_fill, scale_cap_h, scale_cap_w)

        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        paste_x = int(tw / 2 - (fx + fw / 2) * scale)
        paste_y = int(th * HEAD_TOP_MARGIN - person_top * scale)
        self._geometry = {"tw": tw, "th": th, "scale": scale,
                          "paste_x": paste_x, "paste_y": paste_y,
                          "new_w": new_w, "new_h": new_h}

    def _get_bg(self, tw, th):
        """内置背景图裁剪缩放到目标尺寸（带缓存）"""
        key = (tw, th)
        if key not in self._bg_cache:
            bg = Image.open(DEFAULT_BG_IMAGE).convert("RGB")
            bw, bh = bg.size
            scale = max(tw / bw, th / bh)
            new_w, new_h = int(bw * scale), int(bh * scale)
            bg = bg.resize((new_w, new_h), Image.LANCZOS)
            x1 = (new_w - tw) // 2
            y1 = (new_h - th) // 2
            self._bg_cache[key] = bg.crop((x1, y1, x1 + tw, y1 + th)).convert("RGBA")
        return self._bg_cache[key]

    def _compose_photo(self, dx=0.0, dy=0.0):
        """按当前拖动偏移合成证件照，返回 PIL RGBA（全分辨率）"""
        if self.seg_fg is None or self._geometry is None:
            return None
        g = self._geometry
        fg = self.seg_fg.resize((g["new_w"], g["new_h"]), Image.LANCZOS)
        canvas = self._get_bg(g["tw"], g["th"]).copy()
        px = int(g["paste_x"] + dx)
        py = int(g["paste_y"] + dy)
        canvas.alpha_composite(fg, (px, py))
        return canvas

    def _render_result(self):
        """合成并刷新预览（显示缩放 + 坐标映射参数）"""
        composed = self._compose_photo(self.drag_dx, self.drag_dy)
        if composed is None:
            return
        self.processed_pil = composed
        rgb = composed.convert("RGB")
        tw, th = rgb.size

        # 预览缩放（保持比例，适配预览框）
        disp_scale = min(PREVIEW_IMG_W / tw, PREVIEW_IMG_H / th)
        disp_w, disp_h = max(1, int(tw * disp_scale)), max(1, int(th * disp_scale))
        disp = rgb.resize((disp_w, disp_h), Image.LANCZOS)

        photo = ImageTk.PhotoImage(disp)
        self.result_preview.config(image=photo, text="")
        self.result_preview.image = photo
        self._preview_photo = photo
        # 鼠标位移 -> 照片像素 位移 的映射系数
        self._preview_scale = disp_w / tw

        self.offset_label.config(text=f"偏移：{int(self.drag_dx)}, {int(self.drag_dy)}")

    # ==================== 拖动微调 ====================
    def _on_drag_press(self, event):
        if not self.face_processed:
            return
        self._dragging = True
        self._drag_start = (event.x, event.y)

    def _on_drag_motion(self, event):
        if not self._dragging or self._drag_start is None:
            return
        scale = self._preview_scale if self._preview_scale > 0 else 1.0
        dx_px = (event.x - self._drag_start[0]) / scale
        dy_px = (event.y - self._drag_start[1]) / scale
        self._drag_start = (event.x, event.y)

        g = self._geometry or {}
        limit_x = g.get("tw", 400)
        limit_y = g.get("th", 500)
        self.drag_dx = max(-limit_x, min(limit_x, self.drag_dx + dx_px))
        self.drag_dy = max(-limit_y, min(limit_y, self.drag_dy + dy_px))
        self._render_result()

    def _on_drag_release(self, _event):
        self._dragging = False
        self._drag_start = None

    def _nudge(self, ddx, ddy):
        if not self.face_processed:
            return
        g = self._geometry or {}
        limit_x = g.get("tw", 400)
        limit_y = g.get("th", 500)
        self.drag_dx = max(-limit_x, min(limit_x, self.drag_dx + ddx))
        self.drag_dy = max(-limit_y, min(limit_y, self.drag_dy + ddy))
        self._render_result()

    def _reset_offset(self):
        if not self.face_processed:
            return
        self.drag_dx = 0.0
        self.drag_dy = 0.0
        self._render_result()
        self.set_status("已复位到默认构图位置")

    # ==================== 保存 ====================
    def save_photo(self):
        if self.processed_pil is None:
            self.show_error("还没有可保存的处理后照片")
            return
        size_name = self.size_var.get()
        safe_name = size_name.split("(")[0].strip()
        save_path = filedialog.asksaveasfilename(
            title="保存证件照",
            defaultextension=".jpg",
            initialfile=f"证件照_{safe_name}.jpg",
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png")])
        if not save_path:
            return
        try:
            img = self.processed_pil.convert("RGB")
            buf = self._to_bytes_rgb(img, 95)
            with open(save_path, "wb") as f:
                f.write(buf)
            self.show_info(f"照片已保存至：\n{save_path}")
        except Exception as e:
            self.show_error(f"保存失败：{e}")

    @staticmethod
    def _to_bytes_rgb(pil_rgb, quality=92):
        from io import BytesIO
        buf = BytesIO()
        pil_rgb.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    # ==================== 采集流程状态机 ====================
    def _id_fields_valid(self):
        if not self.id_path:
            return False
        sfzjh = self._get_field("id_number")
        if not self._parse_id_card_info(sfzjh)["valid"]:
            return False
        for key in ("name", "gender", "ethnicity"):
            if not self._get_field(key):
                return False
        return True

    def _update_flow(self):
        sfzjh = self._get_field("id_number")
        s1 = self._id_fields_valid()
        s2 = bool(self.identity_sfzjh) and self.identity_sfzjh == sfzjh
        s3 = self.face_processed and self.processed_pil is not None
        s4 = self.just_submitted and s2 and s3
        states = {1: s1, 2: s2, 3: s3, 4: s4}

        current = next((i for i in (1, 2, 3, 4) if not states[i]), None)
        for i, (_, text) in enumerate(self.flow_steps, start=1):
            if states[i]:
                mark, color = "✓", "green"
            elif i == current:
                mark, color = "●", "blue"
            else:
                mark, color = "○", "gray"
            self.flow_labels[i].config(text=f"{mark} {text}", foreground=color)

        if s2:
            xm = self._get_field("name")
            txt = f"当前锁定考生：ksbs={self.identity_ksbs}　姓名={xm}　身份证号={sfzjh}"
            if self.identity_mismatched:
                txt += "　⚠ 与考籍不一致：" + "、".join(self.identity_mismatched)
                self.locked_student_label.config(text=txt, foreground="#cc6600")
            else:
                self.locked_student_label.config(text=txt, foreground="green")
        else:
            self.locked_student_label.config(
                text="当前锁定考生：无（请先识别并上传身份证信息）", foreground="gray")

        if self.api_client.logged_in:
            self.identity_btn.config(state="normal" if s1 else "disabled")
            self.face_submit_btn.config(state="normal" if (s2 and s3) else "disabled")
        else:
            self.identity_btn.config(state="disabled")
            self.face_submit_btn.config(state="disabled")

    def _reset_for_next(self):
        self.face_path = None
        self.id_path = None
        self.face_processed = False
        self.processed_pil = None
        self.seg_fg = None
        self.seg_meta = None
        self._geometry = None
        self.drag_dx = 0.0
        self.drag_dy = 0.0
        self.identity_sfzjh = None
        self.identity_ksbs = None
        self.identity_mismatched = []
        self.just_submitted = False
        self.similarity_result = None
        self._similarity_running = False

        self.face_label.config(text="未选择", foreground="gray")
        self.id_label.config(text="未选择（可选，选后自动识别）", foreground="gray")
        self.face_preview.config(image="", text="无预览")
        self.face_preview.image = None
        self.result_preview.config(image="", text="无预览")
        self.result_preview.image = None
        self.id_preview.config(image="", text="无预览")
        self.id_preview.image = None
        self.offset_label.config(text="偏移：0, 0")

        self._suppress_validate = True
        try:
            for key in ("name", "gender", "ethnicity", "birth", "address", "id_number"):
                self._set_field(key, "")
            for warn in self.warn_labels.values():
                warn.config(text="")
            self.id_status_label.config(text="", foreground="gray")
        finally:
            self._suppress_validate = False

        self.save_btn.config(state="disabled")
        self.mode_var.set("face")
        self._switch_preview_mode()
        self.similarity_label.config(text="人脸相似度比对：等待照片...", foreground="gray")
        self._update_flow()
        self.status.config(text="已复位，请开始下一位：① 选身份证照片")

    # ==================== 登录 ====================
    def open_login_dialog(self):
        if self.login_dialog is not None and self.login_dialog.winfo_exists():
            self.login_dialog.lift()
            self.login_dialog.focus_set()
            return
        self.login_dialog = LoginDialog(self)

    # ==================== ① 上传身份证信息 ====================
    def upload_identity(self):
        if not self.api_client.logged_in:
            self.show_error("请先点击右上角“登录”")
            return
        if not self.id_path:
            self.show_error("请先选择（拍照）身份证照片并完成识别")
            return

        sfzjh = self._get_field("id_number")
        xm = self._get_field("name")
        xb = self._get_field("gender")
        mz = self._get_field("ethnicity")

        parsed = self._parse_id_card_info(sfzjh)
        if not parsed["valid"]:
            self.show_error(f"身份证号校验失败：{parsed['msg']}")
            return
        missing = [label for label, val in
                   [("姓名", xm), ("性别", xb), ("民族", mz)] if not val]
        if missing:
            self.show_error("身份证信息不完整，请补全：" + "、".join(missing))
            return

        self.identity_btn.config(state="disabled", text="上传中...")

        def worker():
            try:
                self.just_submitted = False
                self.set_status("正在编码身份证照片...")
                id_card_b64 = encode_file_base64(self.id_path)
                self.set_status("正在上传身份证信息...")
                identity = self.api_client.save_identity(sfzjh, xm, xb, mz, id_card_b64)

                ksbs = str(identity.get("ksbs", ""))
                mismatched = identity.get("mismatchedFields") or []
                self.identity_sfzjh = sfzjh
                self.identity_ksbs = ksbs
                self.identity_mismatched = mismatched

                if mismatched:
                    msg = ("身份证信息已上传！\n\n"
                           f"考生标识(ksbs)：{ksbs}\n身份证号：{sfzjh}\n\n"
                           "⚠ 以下字段与考籍信息不一致：" + "、".join(mismatched)
                           + "\n\n请核对无误后，再处理并提交人脸照片。")
                else:
                    msg = ("身份证信息已上传，与考籍信息一致！\n\n"
                           f"考生标识(ksbs)：{ksbs}\n身份证号：{sfzjh}\n\n"
                           "身份已锁定，下一步：选择人脸照片并处理。")
                self.msg_queue.put(("info", msg))
                self.set_status(f"身份已锁定（ksbs={ksbs}），请处理人脸照片")
            except Exception as e:
                self.set_status("身份证信息上传失败")
                self.show_error(f"身份证信息上传失败：{e}")
            finally:
                self.root.after(0, self._reset_submit_buttons)

        threading.Thread(target=worker, daemon=True).start()

    # ==================== ② 提交人脸采集 ====================
    def submit_face(self):
        if not self.api_client.logged_in:
            self.show_error("请先点击右上角“登录”")
            return
        if not self.face_processed or self.processed_pil is None:
            self.show_error("请先选择人脸照片并点击“开始处理”，生成证件照")
            return
        if not self.identity_sfzjh:
            self.show_error("请先点击“上传身份证信息”确认考生身份后再提交")
            return

        sfzjh = self._get_field("id_number")
        parsed = self._parse_id_card_info(sfzjh)
        if not parsed["valid"]:
            self.show_error(f"身份证号校验失败：{parsed['msg']}")
            return
        if sfzjh != self.identity_sfzjh:
            self.show_error("当前身份证号与已锁定身份的号码不一致！\n"
                            "请重新点击“上传身份证信息”确认身份，避免张冠李戴。")
            return

        if self.similarity_result is None:
            self.show_error("人脸相似度比对尚未完成。\n"
                            "请确认身份证照片与人脸照片都已选择，稍等片刻再提交。")
            return
        sim, sim_msg = self.similarity_result
        if sim is None:
            self.show_error(f"人脸相似度比对未完成：{sim_msg}")
            return

        if sim < SIMILARITY_WARN:
            if not messagebox.askyesno(
                    "人脸相似度告警",
                    f"人脸照片与身份证头像相似度仅 {sim*100:.1f}%，\n"
                    "疑似不是同一个人。\n\n确认要继续提交吗？（强烈建议重新拍照）",
                    icon="warning"):
                self.set_status("已取消提交")
                return
        elif sim < SIMILARITY_PASS:
            if not messagebox.askyesno(
                    "人脸相似度偏低",
                    f"人脸照片与身份证头像相似度 {sim*100:.1f}%，\n"
                    "低于推荐阈值，请人工确认是否为同一人。\n\n确认继续提交？",
                    icon="warning"):
                self.set_status("已取消提交")
                return

        xm = self._get_field("name")
        face_name = os.path.basename(self.face_path or "")
        warn = ""
        if self.identity_mismatched:
            warn = "\n⚠ 与考籍不一致字段：" + "、".join(self.identity_mismatched) + "\n"
        sim_line = f"\n人脸相似度：{sim*100:.1f}%\n"
        confirm = ("提交前请最后核对（人脸照片将绑定到以下考生）：\n\n"
                   f"考生 ksbs：{self.identity_ksbs}\n姓名：{xm}\n身份证号：{sfzjh}\n"
                   f"人脸照片文件：{face_name}" + sim_line + warn +
                   "\n确认是同一个人，再点“是”提交。")
        if not messagebox.askyesno("提交前最后核对", confirm, icon="warning"):
            self.set_status("已取消提交")
            return

        self.face_submit_btn.config(state="disabled", text="提交中...")
        self.identity_btn.config(state="disabled")

        def worker():
            extract_msg = ""
            try:
                self.set_status("① 正在编码证件照...")
                face_b64 = encode_pil_base64(self.processed_pil.convert("RGB"))
                self.set_status("② 正在上传证件照...")
                self.api_client.verify_face(sfzjh, face_b64)
                self.set_status("③ 正在提交人脸特征生成任务...")
                extract_msg = self.api_client.extract_feature(sfzjh) or "已提交"

                self.just_submitted = True
                self._append_collection_log({
                    "sfzjh": sfzjh,
                    "ksbs": self.identity_ksbs,
                    "xm": xm,
                    "face_file": self.face_path,
                    "id_file": self.id_path,
                    "similarity": round(sim, 4) if sim is not None else None,
                    "offset": (int(self.drag_dx), int(self.drag_dy)),
                    "mismatched_fields": self.identity_mismatched,
                    "extract_msg": extract_msg,
                })
                self.set_status("人脸采集提交完成")

                sim_line2 = f"人脸相似度：{sim*100:.1f}%\n" if sim is not None else ""
                summary = ("提交成功！\n\n"
                           f"考生 ksbs：{self.identity_ksbs}\n姓名：{xm}\n身份证号：{sfzjh}\n"
                           + sim_line2 +
                           "证件照：已上传\n"
                           f"人脸特征任务：{extract_msg}\n\n"
                           "是否清空当前资料，继续采集下一位？")
                self.msg_queue.put(("ask_reset_with_text", summary))
            except Exception as e:
                self.set_status("人脸采集提交失败")
                self.show_error(f"人脸采集提交失败：{e}")
            finally:
                self.root.after(0, self._reset_submit_buttons)

        threading.Thread(target=worker, daemon=True).start()

    def _reset_submit_buttons(self):
        self.face_submit_btn.config(state="normal", text="提交人脸采集")
        self.identity_btn.config(state="normal", text="上传身份证信息")
        self._update_flow()

    @staticmethod
    def _append_collection_log(record):
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
            with open(COLLECTION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"采集日志写入失败：{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoIDApp(root)
    root.mainloop()
