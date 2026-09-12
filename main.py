import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import re
import json
import base64
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import requests
import urllib3
import logging
import time
import sys

# ========== 修复 PyInstaller --windowed 模式下 sys.stdout/stderr 为 None ==========
# 打包成 exe 且无控制台时，sys.stdout/stderr 会是 None，PaddleOCR 写日志会崩
if getattr(sys, 'frozen', False):
    _log_dir = os.path.join(os.path.expanduser("~"), ".photo_id_tool")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, "runtime.log")

    if sys.stdout is None:
        sys.stdout = open(_log_path, 'a', encoding='utf-8', buffering=1)
    if sys.stderr is None:
        sys.stderr = open(_log_path, 'a', encoding='utf-8', buffering=1)

# ========== 日志配置：控制台 + 文件双输出 ==========
_log_handlers = [logging.StreamHandler()]

# 打包后额外把日志写到文件，方便排查
if getattr(sys, 'frozen', False):
    _fh = logging.FileHandler(_log_path, encoding='utf-8')
    _log_handlers.append(_fh)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=_log_handlers,
    force=True,
)
log = logging.getLogger("PhotoIDTool")

logging.getLogger("urllib3").setLevel(logging.WARNING)

PaddleOCR = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== 配置持久化目录 ==========
CONFIG_DIR = os.path.expanduser("~/.photo_id_tool")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
BG_CACHE_FILE = os.path.join(CONFIG_DIR, "bg_cache.jpg")
COLLECTION_LOG_FILE = os.path.join(CONFIG_DIR, "collection_log.jsonl")

# ========== 云端接口默认配置 ==========
DEFAULT_API_BASE = "https://111.12.149.164"
PATH_AUTH_LOGIN = "/admin/apiauth/auth/login"
PATH_FACE_IDENTITY = "/admin/apistudentaffair/public/student/face/identity"
PATH_FACE_VERIFY = "/admin/apistudentaffair/public/student/face/verify"
PATH_FACE_EXTRACT = "/admin/apistudentaffair/admin/face/extract"

# ========== 标准证件照尺寸（300 DPI，单位：像素） ==========
STANDARD_SIZES = {
    "一寸   (25×35mm)":  (295, 413),
    "小一寸 (22×32mm)":  (260, 378),
    "大一寸 (33×48mm)":  (390, 567),
    "二寸   (35×49mm)":  (413, 579),
    "小二寸 (35×45mm)":  (413, 531),
    "大二寸 (35×53mm)":  (413, 626),
}

# ========== 证件照构图参数 ==========
HEAD_HEIGHT_RATIO = 0.62
HEAD_WIDTH_RATIO  = 0.55
FACE_CENTER_Y_RATIO = 0.55
HAIR_WIDTH_FACTOR  = 1.15
HAIR_HEIGHT_FACTOR = 1.55

# ========== 预览框尺寸 ==========
PREVIEW_W, PREVIEW_H = 280, 300
PREVIEW_IMG_W, PREVIEW_IMG_H = 270, 290

# ========== 身份证号校验表 ==========
ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
ID_CHECK_MAP = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

# ========== 人脸相似度阈值 ==========
SIMILARITY_PASS = 0.50      # ≥ 通过
SIMILARITY_WARN = 0.35      # ≥ 但 < 通过：人工确认
                            # < 则判定疑似非同一人

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def ensure_paddle_models():
    """
    打包运行时：把 exe 内置的 PaddleOCR 模型复制到用户目录 ~/.paddleocr
    这样 PaddleOCR 按默认路径就能找到模型，不用改它的初始化代码
    """
    if not getattr(sys, "frozen", False) or not hasattr(sys, "_MEIPASS"):
        return  # 源码运行，不动
    import shutil
    src = os.path.join(sys._MEIPASS, "paddleocr_models")
    dst = os.path.expanduser("~/.paddleocr")
    if not os.path.isdir(src):
        log.warning(f"[ensure_paddle_models] 内置模型目录不存在：{src}")
        return
    # 如果目标已经有模型（且非空），就不重复拷贝
    if os.path.isdir(dst) and any(os.scandir(dst)):
        log.info(f"[ensure_paddle_models] 用户目录已有模型，跳过拷贝：{dst}")
        return
    try:
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        log.info(f"[ensure_paddle_models] 模型已复制到：{dst}")
    except Exception as e:
        log.exception(f"[ensure_paddle_models] 拷贝失败：{e}")


def save_config(cfg):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ========== 云端接口客户端 ==========
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

        log.info("=" * 60)
        log.info("开始登录")
        log.info(f"  URL      : {url}")
        log.info(f"  账号     : {username}")
        log.info(f"  SSL 验证 : 关闭 (verify=False)")
        log.info(f"  超时     : 30 秒")

        t0 = time.time()
        try:
            log.debug(f"  请求体   : {json.dumps(payload, ensure_ascii=False)}")
            log.info("  → 正在发送请求...")

            resp = requests.post(url, json=payload, verify=False, timeout=30)

            elapsed = time.time() - t0
            log.info(f"  ← 已收到响应，耗时 {elapsed:.2f} 秒")
            log.info(f"  HTTP 状态码: {resp.status_code}")
            log.debug(f"  响应头     : {dict(resp.headers)}")
            log.info(f"  响应体原文 : {resp.text[:1000]}")

        except requests.exceptions.ConnectTimeout:
            log.error(f"  ✗ 连接超时（{time.time()-t0:.2f}s）：无法连接到服务器")
            return False, "连接超时，请检查 API 地址和网络"
        except requests.exceptions.ReadTimeout:
            log.error(f"  ✗ 读取超时（{time.time()-t0:.2f}s）：服务器未在 30 秒内返回")
            return False, "服务器响应超时"
        except requests.exceptions.SSLError as e:
            log.error(f"  ✗ SSL 错误：{e}")
            return False, f"SSL 错误：{e}"
        except requests.exceptions.ConnectionError as e:
            log.error(f"  ✗ 连接错误：{e}")
            return False, f"连接失败：{e}"
        except requests.exceptions.RequestException as e:
            log.error(f"  ✗ 请求异常：{type(e).__name__}: {e}")
            return False, f"网络异常：{e}"
        except Exception as e:
            log.exception(f"  ✗ 未知异常：{e}")
            return False, f"登录异常：{e}"

        try:
            body = resp.json()
        except ValueError:
            log.error(f"  ✗ 响应不是合法 JSON")
            return False, f"登录响应无法解析(HTTP {resp.status_code})"

        log.debug(f"  解析后 JSON: {json.dumps(body, ensure_ascii=False)[:1000]}")

        if resp.status_code == 200 and body.get("code") == 200 and body.get("data"):
            token = body["data"].get("token")
            if not token:
                log.error("  ✗ 响应里没有 token 字段")
                return False, "登录响应中缺少 token"
            self.token = token
            self.username = username
            log.info(f"  ✓ 登录成功，token = {token[:20]}...")
            log.info("=" * 60)
            return True, "登录成功"

        log.error(f"  ✗ 登录失败：code={body.get('code')}, msg={body.get('msg')}")
        log.info("=" * 60)
        return False, body.get("msg") or f"登录失败(HTTP {resp.status_code})"

    def save_identity(self, sfzjh, xm, xb, mz, id_card_b64):
        url = f"{self.base_url}{PATH_FACE_IDENTITY}"
        log.info(f"[save_identity] POST {url}  sfzjh={sfzjh}")
        resp = requests.post(
            url,
            headers=self._headers(),
            json={
                "sfzjh": sfzjh,
                "xm": xm,
                "xb": xb,
                "mz": mz,
                "idCardPhotoBase64": id_card_b64,
                "idCardPhotoContentType": "image/jpeg",
            },
            verify=False,
            timeout=60,
        )
        log.debug(f"[save_identity] HTTP {resp.status_code} body={resp.text[:500]}")
        return self._parse_result(resp, "身份证信息上传失败")

    def verify_face(self, sfzjh, photo_b64):
        url = f"{self.base_url}{PATH_FACE_VERIFY}"
        log.info(f"[verify_face] POST {url}  sfzjh={sfzjh}")
        resp = requests.post(
            url,
            headers=self._headers(),
            json={"sfzjh": sfzjh, "photoBase64": photo_b64},
            verify=False,
            timeout=60,
        )
        log.debug(f"[verify_face] HTTP {resp.status_code} body={resp.text[:500]}")
        return self._parse_result(resp, "人脸照片上传失败")

    def extract_feature(self, sfzjh):
        url = f"{self.base_url}{PATH_FACE_EXTRACT}"
        log.info(f"[extract_feature] POST {url}  sfzjh={sfzjh}")
        resp = requests.post(
            url,
            headers=self._headers(with_auth=True),
            params={"sfzjh": sfzjh},
            verify=False,
            timeout=30,
        )
        log.debug(f"[extract_feature] HTTP {resp.status_code} body={resp.text[:500]}")
        try:
            body = resp.json()
        except ValueError:
            resp.raise_for_status()
            return
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


# ========== 人脸相似度比对（本地 InsightFace） ==========
# ========== 人脸相似度比对（本地 InsightFace） ==========
class FaceComparator:
    """基于 InsightFace 的本地人脸相似度比对
    优先从 exe 内置目录 / 项目目录 / 用户目录查找模型，不再联网下载。
    """

    def __init__(self):
        self.app = None
        self._ready = False
        self._error = None
        self._model_name = "buffalo_l"
        self._required_files = ["det_10g.onnx", "w600k_r50.onnx"]

    def _find_model_root(self):
        """返回 (root, model_dir)，找不到返回 (None, None)"""
        candidates = []

        # 1) PyInstaller 打包后解压的临时目录
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            candidates.append(sys._MEIPASS)

        # 2) 与脚本/exe 同级的目录
        try:
            here = os.path.dirname(os.path.abspath(sys.argv[0]))
            candidates.append(here)
        except Exception:
            pass
        candidates.append(os.getcwd())

        # 3) 用户目录
        candidates.append(os.path.expanduser("~/.insightface"))

        for root in candidates:
            if not root or not os.path.isdir(root):
                continue
            model_dir = os.path.join(root, "models", self._model_name)
            if not os.path.isdir(model_dir):
                continue
            if all(
                os.path.isfile(os.path.join(model_dir, f))
                for f in self._required_files
            ):
                return root, model_dir
        return None, None

    def init(self):
        try:
            log.info("[FaceComparator] 正在查找本地 InsightFace 模型...")
            root, model_dir = self._find_model_root()
            if root is None:
                self._error = "未找到 buffalo_l 模型"
                log.error("[FaceComparator] 未找到模型目录")
                log.error("[FaceComparator] 已查找以下位置：")
                log.error("  - 打包临时目录 / 项目目录 / 用户主目录")
                log.error("[FaceComparator] 手动放置模型：")
                log.error("  下载 buffalo_l.zip 解压到 ~/.insightface/models/buffalo_l/")
                return

            log.info(f"[FaceComparator] 找到模型目录：{model_dir}")

            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name=self._model_name,
                root=root,
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self._ready = True
            log.info("[FaceComparator] 模型加载完成 ✓")
        except Exception as e:
            self._error = str(e)
            log.exception(f"[FaceComparator] 模型加载失败：{e}")

    @property
    def ready(self):
        return self._ready

    @property
    def error(self):
        return self._error

    def _get_embedding(self, img_bgr):
        if not self._ready or self.app is None:
            return None
        try:
            faces = self.app.get(img_bgr)
            if not faces:
                return None
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            return face.normed_embedding
        except Exception as e:
            log.warning(f"[FaceComparator] 特征提取失败：{e}")
            return None

    def compare(self, img1_bgr, img2_bgr):
        if not self._ready:
            return None, f"人脸比对模块未就绪（{self._error or '加载中'}）"

        e1 = self._get_embedding(img1_bgr)
        if e1 is None:
            return None, "身份证照片中未检测到人脸"

        e2 = self._get_embedding(img2_bgr)
        if e2 is None:
            return None, "人脸照片中未检测到人脸"

        sim = float(np.dot(e1, e2))
        return sim, "ok"

# ========== 登录弹窗 ==========
class LoginDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("登录")
        self.resizable(False, False)
        self.transient(app.root)
        self.grab_set()

        self._login_result = None

        frame = ttk.Frame(self, padding=18)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="云端账号登录", font=("Helvetica", 13, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 10), sticky="w"
        )

        ttk.Label(frame, text="API地址：").grid(row=1, column=0, sticky="e", pady=4)
        self.base_entry = ttk.Entry(frame, width=30)
        self.base_entry.grid(row=1, column=1, pady=4)
        self.base_entry.insert(0, app.config.get("api_base_url", DEFAULT_API_BASE))

        ttk.Label(frame, text="账号：").grid(row=2, column=0, sticky="e", pady=4)
        self.user_entry = ttk.Entry(frame, width=30)
        self.user_entry.grid(row=2, column=1, pady=4)
        self.user_entry.insert(0, app.config.get("api_username", "admin"))

        ttk.Label(frame, text="密码：").grid(row=3, column=0, sticky="e", pady=4)
        self.pwd_entry = ttk.Entry(frame, width=30, show="*")
        self.pwd_entry.grid(row=3, column=1, pady=4)
        self.pwd_entry.insert(0, app.config.get("api_password", ""))

        self.tip_label = ttk.Label(frame, text="", foreground="red")
        self.tip_label.grid(row=4, column=0, columnspan=2, pady=(6, 2))

        self.login_btn = ttk.Button(frame, text="登录", width=14, command=self.do_login)
        self.login_btn.grid(row=5, column=0, columnspan=2, pady=(6, 0))

        self.bind("<Return>", lambda e: self.do_login())
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        self.update_idletasks()
        mx, my = app.root.winfo_rootx(), app.root.winfo_rooty()
        mw, mh = app.root.winfo_width(), app.root.winfo_height()
        w, h = 360, 250
        self.geometry(f"{w}x{h}+{mx + (mw - w) // 2}+{my + (mh - h) // 2}")

        if self.user_entry.get():
            self.pwd_entry.focus_set()
        else:
            self.user_entry.focus_set()

        self.after(100, self._poll_login_result)

    def do_login(self):
        base_url = self.base_entry.get().strip()
        username = self.user_entry.get().strip()
        password = self.pwd_entry.get().strip()
        if not base_url or not username or not password:
            self.tip_label.config(text="请填写 API地址、账号和密码", foreground="red")
            return

        self._login_result = None
        self.login_btn.config(state="disabled", text="登录中...")
        self.tip_label.config(text="正在登录，请稍候...", foreground="gray")
        threading.Thread(
            target=self._login_worker,
            args=(base_url, username, password),
            daemon=True,
        ).start()

    def _login_worker(self, base_url, username, password):
        try:
            ok, msg = self.app.api_client.login(base_url, username, password)
        except Exception as e:
            log.exception("[LoginDialog] 登录线程异常")
            ok, msg = False, f"登录异常：{e}"
        self._login_result = (ok, msg, base_url, username, password)

    def _poll_login_result(self):
        if not self.winfo_exists():
            return

        result = self._login_result
        if result is not None:
            self._login_result = None
            ok, msg, base_url, username, password = result
            if ok:
                self._success(base_url, username, password)
                return
            else:
                self._fail(msg)

        self.after(100, self._poll_login_result)

    def _success(self, base_url, username, password):
        log.info("[LoginDialog] 执行 _success，更新主界面状态")
        self.app.config["api_base_url"] = base_url
        self.app.config["api_username"] = username
        self.app.config["api_password"] = password
        save_config(self.app.config)

        self.app.login_status_label.config(
            text=f"已登录：{username}", foreground="green"
        )
        self.app._update_flow()
        self.app.set_status("云端登录成功")
        self.destroy()

    def _fail(self, msg):
        log.warning(f"[LoginDialog] 登录失败：{msg}")
        self.tip_label.config(text=msg, foreground="red")
        self.login_btn.config(state="normal", text="登录")


class PhotoIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("证件照制作工具")
        self.root.geometry("1180x900")
        self.root.minsize(1080, 840)

        self.face_path = None
        self.id_path = None
        self.bg_image_path = None
        self.processed_image = None
        self.face_processed = False
        self.id_info = {}
        self.ocr_engine = None

        self.pending_id_recognition = False
        self._suppress_validate = False

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_face_detection = mp.solutions.face_detection
        self.segmenter = None
        self.face_detector = None

        self.msg_queue = queue.Queue()
        self.config = load_config()

        self.api_client = ApiClient()
        self.login_dialog = None

        self.identity_sfzjh = None
        self.identity_ksbs = None
        self.identity_mismatched = []
        self.just_submitted = False

        # 人脸相似度比对
        self.face_comparator = FaceComparator()
        self.similarity_result = None    # (sim, msg)
        self._similarity_running = False

        self._build_ui()
        self._restore_config()
        self._poll_queue()
        ensure_paddle_models() 
        self._init_ocr_async()
        self._init_face_comparator_async()

    # ==================== UI ====================
    def _build_ui(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=12, pady=(8, 2))

        ttk.Label(
            header, text="证件照制作工具",
            font=("Helvetica", 16, "bold")
        ).pack(side="left")

        header_right = ttk.Frame(header)
        header_right.pack(side="right")
        self.login_status_label = ttk.Label(
            header_right, text="未登录", foreground="gray"
        )
        self.login_status_label.pack(side="left", padx=(0, 8))
        ttk.Button(
            header_right, text="登录", width=10, command=self.open_login_dialog
        ).pack(side="left")

        # ---------- 文件选择 ----------
        file_frame = ttk.LabelFrame(self.root, text="文件选择", padding=8)
        file_frame.pack(fill="x", padx=12, pady=4)

        row1 = ttk.Frame(file_frame)
        row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="人脸照片：", width=12, anchor="w").pack(side="left")
        self.face_label = ttk.Label(row1, text="未选择", foreground="gray", anchor="w")
        self.face_label.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row1, text="选择文件", command=self.select_face).pack(side="right")

        row2 = ttk.Frame(file_frame)
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="身份证照片：", width=12, anchor="w").pack(side="left")
        self.id_label = ttk.Label(row2, text="未选择（可选，选后自动识别）", foreground="gray", anchor="w")
        self.id_label.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(row2, text="选择文件", command=self.select_id).pack(side="right")

        # ---------- 参数 ----------
        opt_frame = ttk.LabelFrame(self.root, text="证件照参数", padding=8)
        opt_frame.pack(fill="x", padx=12, pady=4)

        size_row = ttk.Frame(opt_frame)
        size_row.pack(fill="x", pady=2)
        ttk.Label(size_row, text="规格：", width=8, anchor="w").pack(side="left")
        self.size_var = tk.StringVar(value=list(STANDARD_SIZES.keys())[0])
        size_combo = ttk.Combobox(
            size_row, textvariable=self.size_var,
            values=list(STANDARD_SIZES.keys()), state="readonly", width=22
        )
        size_combo.pack(side="left", padx=5)
        size_combo.bind("<<ComboboxSelected>>", lambda e: self._on_size_change())

        color_row = ttk.Frame(opt_frame)
        color_row.pack(fill="x", pady=2)
        ttk.Label(color_row, text="背景：", width=8, anchor="w").pack(side="left")
        self.bg_var = tk.StringVar(value="white")
        for text, val in [("白底", "white"), ("蓝底", "blue"), ("红底", "red"), ("自定义图片", "image")]:
            ttk.Radiobutton(
                color_row, text=text, value=val, variable=self.bg_var,
                command=self._on_bg_change
            ).pack(side="left", padx=6)

        bg_row = ttk.Frame(opt_frame)
        bg_row.pack(fill="x", pady=2)
        ttk.Label(bg_row, text="", width=8).pack(side="left")
        self.bg_image_label = ttk.Label(
            bg_row, text="未选择背景图", foreground="gray", anchor="w"
        )
        self.bg_image_label.pack(side="left", fill="x", expand=True, padx=5)
        self.bg_image_btn = ttk.Button(
            bg_row, text="选择背景图", command=self.select_bg_image, state="disabled"
        )
        self.bg_image_btn.pack(side="right")

        # ---------- 采集流程 ----------
        flow_frame = ttk.LabelFrame(
            self.root,
            text="采集流程（单人单次：身份证确认身份 → 人脸采集 → 自动复位）",
            padding=8
        )
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
            font=("Helvetica", 11, "bold"), foreground="gray"
        )
        self.locked_student_label.pack(anchor="w", pady=(6, 0))

        self.similarity_label = ttk.Label(
            flow_frame, text="人脸相似度比对：等待照片...",
            font=("Helvetica", 11, "bold"), foreground="gray"
        )
        self.similarity_label.pack(anchor="w", pady=(4, 0))

        # ---------- 按钮区 ----------
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=12, pady=6)

        self.start_btn = ttk.Button(
            btn_frame, text="开始处理", command=self.start_process, state="disabled", width=14
        )
        self.start_btn.pack(side="left", padx=5)

        self.save_btn = ttk.Button(
            btn_frame, text="保存照片", command=self.save_photo, state="disabled", width=14
        )
        self.save_btn.pack(side="left", padx=5)

        self.identity_btn = ttk.Button(
            btn_frame, text="上传身份证信息", command=self.upload_identity,
            state="disabled", width=16
        )
        self.identity_btn.pack(side="left", padx=5)

        self.face_submit_btn = ttk.Button(
            btn_frame, text="提交人脸采集", command=self.submit_face,
            state="disabled", width=16
        )
        self.face_submit_btn.pack(side="left", padx=5)

        ttk.Button(btn_frame, text="退出", command=self.root.quit, width=10).pack(side="right", padx=5)

        # ---------- 预览 ----------
        preview_frame = ttk.LabelFrame(self.root, text="预览", padding=8)
        preview_frame.pack(fill="x", padx=12, pady=4)

        toggle_row = ttk.Frame(preview_frame)
        toggle_row.pack(fill="x", pady=(0, 6))
        self.mode_var = tk.StringVar(value="face")
        ttk.Radiobutton(
            toggle_row, text="人脸模式", value="face",
            variable=self.mode_var, command=self._switch_preview_mode
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            toggle_row, text="身份证模式", value="id",
            variable=self.mode_var, command=self._switch_preview_mode
        ).pack(side="left", padx=5)

        self.preview_container = ttk.Frame(preview_frame)
        self.preview_container.pack(fill="x")

        # ====== 模式 A：人脸 ======
        self.preview_face_frame = ttk.Frame(self.preview_container)

        left_face = ttk.Frame(self.preview_face_frame)
        left_face.pack(side="left", padx=15)
        ttk.Label(left_face, text="原始照片", font=("Helvetica", 11)).pack()

        face_box = tk.Frame(
            left_face, width=PREVIEW_W, height=PREVIEW_H,
            background="#f5f5f5", relief="solid", borderwidth=1
        )
        face_box.pack(pady=4)
        face_box.pack_propagate(False)
        self.face_preview = tk.Label(face_box, background="#f5f5f5", text="无预览")
        self.face_preview.pack(fill="both", expand=True)

        right_face = ttk.Frame(self.preview_face_frame)
        right_face.pack(side="left", padx=15)
        ttk.Label(right_face, text="证件照预览", font=("Helvetica", 11)).pack()

        result_box = tk.Frame(
            right_face, width=PREVIEW_W, height=PREVIEW_H,
            background="#f5f5f5", relief="solid", borderwidth=1
        )
        result_box.pack(pady=4)
        result_box.pack_propagate(False)
        self.result_preview = tk.Label(result_box, background="#f5f5f5", text="无预览")
        self.result_preview.pack(fill="both", expand=True)

        # ====== 模式 B：身份证 ======
        self.preview_id_frame = ttk.Frame(self.preview_container)
        self.preview_id_frame.grid_columnconfigure(1, weight=1)

        left_id = ttk.Frame(self.preview_id_frame)
        left_id.grid(row=0, column=0, padx=(15, 10), sticky="n")
        ttk.Label(left_id, text="身份证照片", font=("Helvetica", 11)).pack()

        id_photo_box = tk.Frame(
            left_id, width=PREVIEW_W, height=PREVIEW_H,
            background="#f5f5f5", relief="solid", borderwidth=1
        )
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
            ("name",      "姓名",     "entry", 1),
            ("gender",    "性别",     "entry", 1),
            ("ethnicity", "民族",     "entry", 1),
            ("birth",     "出生",     "entry", 1),
            ("address",   "住址",     "text",  3),
            ("id_number", "身份证号", "entry", 1),
        ]

        for i, (key, label_text, wtype, h) in enumerate(fields):
            tk.Label(
                inner, text=label_text + "：", background="white",
                font=font_field, anchor="w", width=8
            ).grid(row=i, column=0, sticky="nw", pady=3)

            if wtype == "entry":
                entry = tk.Entry(inner, font=font_field)
                entry.grid(row=i, column=1, sticky="ew", padx=(0, 8), pady=3)
                entry.bind("<KeyRelease>", lambda e: self._validate_id_consistency())
                entry.bind("<FocusOut>",  self._on_id_field_focusout)
            else:
                entry = tk.Text(inner, font=font_field, height=h, wrap="word",
                                relief="solid", borderwidth=1)
                entry.grid(row=i, column=1, sticky="ew", padx=(0, 8), pady=3)
                entry.bind("<KeyRelease>", lambda e: self._validate_id_consistency())
                entry.bind("<FocusOut>",  self._on_id_field_focusout)

            self.id_entries[key] = entry

            warn = tk.Label(
                inner, text="", background="white",
                font=("Helvetica", 10), foreground="red",
                anchor="w", width=22
            )
            warn.grid(row=i, column=2, sticky="nw", pady=3)
            self.warn_labels[key] = warn

        bottom = tk.Frame(inner, background="white")
        bottom.grid(row=len(fields), column=0, columnspan=3, sticky="ew", pady=(10, 0))
        bottom.grid_columnconfigure(0, weight=1)

        self.id_status_label = tk.Label(
            bottom, text="", background="white",
            font=("Helvetica", 10), anchor="w", foreground="gray"
        )
        self.id_status_label.grid(row=0, column=0, sticky="w")

        self.fix_btn = ttk.Button(
            bottom, text="按身份证号纠正",
            command=self._apply_id_card_info
        )
        self.fix_btn.grid(row=0, column=1, sticky="e")

        self.preview_face_frame.pack()

        self.status = ttk.Label(
            self.root, text="正在准备中，请稍候...",
            relief="sunken", anchor="w", padding=5
        )
        self.status.pack(fill="x", side="bottom")

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

    # ==================== 配置恢复 ====================
    def _restore_config(self):
        size_name = self.config.get("size_name")
        if size_name and size_name in STANDARD_SIZES:
            self.size_var.set(size_name)

        bg_type = self.config.get("bg_type", "white")
        self.bg_var.set(bg_type)
        if bg_type == "image":
            self.bg_image_btn.config(state="normal")

        if bg_type == "image" and os.path.exists(BG_CACHE_FILE):
            self.bg_image_path = BG_CACHE_FILE
            self.bg_image_label.config(
                text=f"已缓存：{BG_CACHE_FILE}", foreground="black"
            )

    def _save_bg_config(self):
        self.config["size_name"] = self.size_var.get()
        self.config["bg_type"] = self.bg_var.get()
        save_config(self.config)

    # ==================== 线程通信 ====================
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "status":
                    self.status.config(text=payload)
                elif kind == "btn_state":
                    widget, state = payload
                    widget.config(state=state)
                elif kind == "btn_text":
                    widget, text = payload
                    widget.config(text=text)
                elif kind == "preview_file":
                    widget, path = payload
                    self._show_preview_file(path, widget)
                elif kind == "preview_array":
                    widget, img_bgr = payload
                    self._show_preview_array(img_bgr, widget)
                elif kind == "id_info":
                    self._update_id_info_display(payload)
                elif kind == "similarity":
                    sim, msg = payload
                    self._on_similarity_done(sim, msg)
                elif kind == "flow":
                    self._update_flow()
                    # 如果人脸比对模型刚好加载完，而之前有过等待，重新触发
                    if (self.face_path and self.id_path
                            and self.face_comparator.ready
                            and self.similarity_result is None
                            and not self._similarity_running):
                        self._auto_compare_faces()
                elif kind == "ask_reset_with_text":
                    if messagebox.askyesno("采集完成", payload):
                        self._reset_for_next()
                    else:
                        self._update_flow()
                elif kind == "error":
                    messagebox.showerror("错误", payload)
                elif kind == "info":
                    messagebox.showinfo("提示", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def set_status(self, text):
        self.msg_queue.put(("status", text))

    def set_btn(self, widget, state=None, text=None):
        if state is not None:
            self.msg_queue.put(("btn_state", (widget, state)))
        if text is not None:
            self.msg_queue.put(("btn_text", (widget, text)))

    def show_error(self, msg):
        self.msg_queue.put(("error", msg))

    def show_info(self, msg):
        self.msg_queue.put(("info", msg))

    # ==================== 预览渲染 ====================
    def _show_preview_file(self, image_path, label_widget):
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((PREVIEW_IMG_W, PREVIEW_IMG_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            label_widget.config(text=f"预览失败: {e}", image="")

    def _show_preview_array(self, img_bgr, label_widget):
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img.thumbnail((PREVIEW_IMG_W, PREVIEW_IMG_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            label_widget.config(text=f"预览失败: {e}", image="")

    # ==================== 身份证信息显示 ====================
    def _update_id_info_display(self, info):
        self._suppress_validate = True
        try:
            for key in ["name", "gender", "ethnicity", "birth", "address", "id_number"]:
                self._set_field(key, info.get(key, "") or "")
        finally:
            self._suppress_validate = False
        self._validate_id_consistency()
        self._update_flow()

    # ==================== 身份证号解析 ====================
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
            year = int(s[6:10])
            month = int(s[10:12])
            day = int(s[12:14])
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

    # ==================== 一致性校验 ====================
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
                text=f"身份证号还差 {18 - len(id_number)} 位", foreground="gray"
            )
            return

        parsed = self._parse_id_card_info(id_number)

        if not parsed["valid"]:
            self.id_status_label.config(text=f"⚠ {parsed['msg']}", foreground="red")
            return
        else:
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

    # ==================== 按身份证号纠正 ====================
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

    # ==================== OCR 异步初始化 ====================
    def _init_ocr_async(self):
        def worker():
            global PaddleOCR
            try:
                from paddleocr import PaddleOCR as _PaddleOCR
                PaddleOCR = _PaddleOCR

                self.set_status("正在初始化识别引擎，首次运行需下载模型，请耐心等待...")
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

                self.set_status("就绪")
                self.set_btn(self.start_btn, state="normal")

                if self.pending_id_recognition:
                    self.pending_id_recognition = False
                    self._auto_recognize_id()
            except Exception as e:
                self.set_status("初始化失败")
                self.show_error(f"识别引擎初始化失败：{e}")

        threading.Thread(target=worker, daemon=True).start()

    # ==================== 人脸比对模型异步初始化 ====================
    def _init_face_comparator_async(self):
        threading.Thread(target=self.face_comparator.init, daemon=True).start()

    # ==================== 文件选择 ====================
    def _pick_file(self, title):
        return filedialog.askopenfilename(
            title=title,
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )

    def _validate_path(self, path):
        if not path:
            return False, "未选择文件"
        if not os.path.exists(path):
            return False, f"路径不存在：{path}"
        if not os.path.isfile(path):
            return False, f"路径不是文件：{path}"
        img = cv2.imread(path)
        if img is None:
            return False, f"无法读取图片：{path}"
        return True, ""

    def select_face(self):
        path = self._pick_file("请选择人脸照片")
        if not path:
            return
        path = os.path.normpath(path)
        ok, err = self._validate_path(path)
        if not ok:
            self.show_error(err)
            return
        self.face_path = path
        self.face_label.config(text=path, foreground="black")
        self._show_preview_file(path, self.face_preview)
        self._show_face_mode()
        self._invalidate_face_result()
        self._auto_compare_faces()

    def select_id(self):
        path = self._pick_file("请选择身份证照片")
        if not path:
            return
        path = os.path.normpath(path)
        ok, err = self._validate_path(path)
        if not ok:
            self.show_error(err)
            return
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

    def _invalidate_face_result(self):
        self.processed_image = None
        self.face_processed = False
        self.just_submitted = False
        self.result_preview.config(image="", text="无预览")
        self.result_preview.image = None
        self.save_btn.config(state="disabled")
        self._update_flow()

    def _auto_recognize_id(self):
        if not self.id_path:
            return
        if self.ocr_engine is None:
            self.pending_id_recognition = True
            self.set_status("识别引擎初始化中，稍后自动识别身份证...")
            return

        self.set_status("正在识别身份证...")
        threading.Thread(target=self._id_recognition_worker, daemon=True).start()

    def _id_recognition_worker(self):
        try:
            self.id_info = self._recognize_id_card(self.id_path)
            self.msg_queue.put(("id_info", self.id_info))
            self.set_status("身份证识别完成")
        except Exception as e:
            self.set_status("身份证识别失败")
            self.show_error(f"身份证识别失败：{e}")

    def select_bg_image(self):
        path = self._pick_file("请选择背景图片")
        if not path:
            return
        path = os.path.normpath(path)
        ok, err = self._validate_path(path)
        if not ok:
            self.show_error(err)
            return

        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            img = cv2.imread(path)
            cv2.imwrite(BG_CACHE_FILE, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.bg_image_path = BG_CACHE_FILE
            self.bg_image_label.config(
                text=f"已缓存：{BG_CACHE_FILE}", foreground="black"
            )
            self.config["bg_type"] = "image"
            save_config(self.config)
        except Exception as e:
            self.show_error(f"背景图保存失败：{e}")
            return

    # ==================== 人脸相似度比对 ====================
    def _auto_compare_faces(self):
        """两张照片都在时自动后台比对"""
        # 换了任意一张照片，旧结果作废
        self.similarity_result = None

        if not self.face_path or not self.id_path:
            self.similarity_label.config(
                text="人脸相似度比对：等待照片...", foreground="gray"
            )
            return

        if not self.face_comparator.ready:
            err = self.face_comparator.error
            if err:
                self.similarity_label.config(
                    text=f"人脸相似度比对：模型加载失败（{err}）",
                    foreground="#cc6600"
                )
            else:
                self.similarity_label.config(
                    text="人脸相似度比对：模型加载中，请稍候...",
                    foreground="gray"
                )
            return

        if self._similarity_running:
            return

        self._similarity_running = True
        self.similarity_label.config(
            text="人脸相似度比对：正在计算...", foreground="blue"
        )
        threading.Thread(target=self._compare_worker, daemon=True).start()

    def _compare_worker(self):
        sim, msg = None, "比对异常"
        try:
            img_id = cv2.imread(self.id_path)
            img_face = cv2.imread(self.face_path)
            if img_id is None or img_face is None:
                sim, msg = None, "图片读取失败"
            else:
                # 先裁剪放大身份证上的头像
                img_id_enhanced, bbox = self._crop_id_card_portrait(img_id)
                if bbox is None:
                    log.warning("[比对] 身份证上未检测到人像小头像，使用原图")
                else:
                    log.info(f"[比对] 检测到身份证人像位置 {bbox}")
                sim, msg = self.face_comparator.compare(img_id_enhanced, img_face)
        except Exception as e:
            log.exception("[compare_worker] 异常")
            sim, msg = None, f"比对异常：{e}"
        self._similarity_running = False
        self.msg_queue.put(("similarity", (sim, msg)))

    def _crop_id_card_portrait(self, id_img_bgr):
        """在身份证照片里找到人像小头像，裁剪并放大，返回增强后的图
        如果检测不到，返回原图
        """
        if self.face_detector is None:
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,   # 远距离模型，适合身份证上的小头像
                min_detection_confidence=0.4
            )

        h, w = id_img_bgr.shape[:2]
        rgb = cv2.cvtColor(id_img_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)

        if not results.detections:
            return id_img_bgr, None

        # 身份证上一般只有一个人像，取置信度最高的
        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(int(bbox.width * w), w - x)
        bh = min(int(bbox.height * h), h - y)

        if bw < 20 or bh < 20:
            return id_img_bgr, None

        # 向外扩展 40%，保留头发和下巴
        pad_x = int(bw * 0.4)
        pad_y = int(bh * 0.4)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        crop = id_img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return id_img_bgr, None

        # 放大到至少 200px 宽，让 InsightFace 的检测器能稳定工作
        ch, cw = crop.shape[:2]
        target = 224
        scale = max(target / cw, target / ch, 1.0)
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        crop_big = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 轻微锐化，弥补原图压缩损失
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        crop_big = cv2.filter2D(crop_big, -1, kernel)

        log.info(f"[比对] 身份证头像裁剪：{cw}x{ch} → {new_w}x{new_h}")
        return crop_big, (x, y, bw, bh)


    def _on_similarity_done(self, sim, msg):
        self.similarity_result = (sim, msg)

        if sim is None:
            self.similarity_label.config(
                text=f"人脸相似度比对：⚠ {msg}", foreground="#cc6600"
            )
            return

        pct = f"{sim * 100:.1f}%"
        if sim >= SIMILARITY_PASS:
            self.similarity_label.config(
                text=f"人脸相似度比对：{pct}（✓ 通过）", foreground="green"
            )
        elif sim >= SIMILARITY_WARN:
            self.similarity_label.config(
                text=f"人脸相似度比对：{pct}（⚠ 偏低，请人工核对）",
                foreground="#cc6600"
            )
        else:
            self.similarity_label.config(
                text=f"人脸相似度比对：{pct}（✗ 疑似非同一人）",
                foreground="red"
            )

    # ==================== 选项 ====================
    def _get_bg_color(self):
        mapping = {
            "white": (255, 255, 255),
            "blue":  (219, 142, 67),
            "red":   (0, 0, 255),
        }
        return mapping.get(self.bg_var.get(), (255, 255, 255))

    def _on_bg_change(self):
        if self.bg_var.get() == "image":
            self.bg_image_btn.config(state="normal")
        else:
            self.bg_image_btn.config(state="disabled")

        self._save_bg_config()

        if self.processed_image is not None:
            self.start_process()

    def _on_size_change(self):
        self._save_bg_config()
        if self.processed_image is not None:
            self.start_process()

    # ==================== 保存 ====================
    def save_photo(self):
        if self.processed_image is None:
            self.show_error("还没有可保存的处理后照片")
            return
        size_name = self.size_var.get()
        safe_name = size_name.split("(")[0].strip()
        default_name = f"证件照_{safe_name}.jpg"
        save_path = filedialog.asksaveasfilename(
            title="保存证件照",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png")]
        )
        if not save_path:
            return
        try:
            cv2.imwrite(save_path, self.processed_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.show_info(f"照片已保存至：\n{save_path}")
        except Exception as e:
            self.show_error(f"保存失败：{e}")

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
        s3 = self.face_processed and self.processed_image is not None
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
                text="当前锁定考生：无（请先识别并上传身份证信息）", foreground="gray"
            )

        if self.api_client.logged_in:
            self.identity_btn.config(state="normal" if s1 else "disabled")
            self.face_submit_btn.config(state="normal" if (s2 and s3) else "disabled")
        else:
            self.identity_btn.config(state="disabled")
            self.face_submit_btn.config(state="disabled")

    def refresh_flow(self):
        self.msg_queue.put(("flow", None))

    def _reset_for_next(self):
        self.face_path = None
        self.id_path = None
        self.processed_image = None
        self.face_processed = False
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
        self.similarity_label.config(
            text="人脸相似度比对：等待照片...", foreground="gray"
        )
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

        self.set_btn(self.identity_btn, state="disabled", text="上传中...")
        threading.Thread(
            target=self._identity_worker,
            args=(sfzjh, xm, xb, mz),
            daemon=True,
        ).start()

    def _identity_worker(self, sfzjh, xm, xb, mz):
        try:
            self.just_submitted = False
            self.set_status("正在编码身份证照片...")
            id_card_b64 = self._encode_image_file_base64(self.id_path)

            self.set_status("正在上传身份证信息...")
            identity = self.api_client.save_identity(sfzjh, xm, xb, mz, id_card_b64)

            ksbs = str(identity.get("ksbs", ""))
            mismatched = identity.get("mismatchedFields") or []

            self.identity_sfzjh = sfzjh
            self.identity_ksbs = ksbs
            self.identity_mismatched = mismatched
            self.refresh_flow()

            if mismatched:
                msg = (
                    "身份证信息已上传！\n\n"
                    f"考生标识(ksbs)：{ksbs}\n"
                    f"身份证号：{sfzjh}\n\n"
                    "⚠ 以下字段与考籍信息不一致：" + "、".join(mismatched)
                    + "\n\n请核对无误后，再处理并提交人脸照片。"
                )
            else:
                msg = (
                    "身份证信息已上传，与考籍信息一致！\n\n"
                    f"考生标识(ksbs)：{ksbs}\n"
                    f"身份证号：{sfzjh}\n\n"
                    "身份已锁定，下一步：选择人脸照片并处理。"
                )
            self.show_info(msg)
            self.set_status("身份已锁定（ksbs=%s），请处理人脸照片" % ksbs)
        except requests.exceptions.RequestException as e:
            self.set_status("身份证信息上传失败")
            self.show_error(f"网络请求失败：{e}")
        except Exception as e:
            self.set_status("身份证信息上传失败")
            self.show_error(f"身份证信息上传失败：{e}")
        finally:
            self.set_btn(self.identity_btn, state="normal", text="上传身份证信息")
            self.refresh_flow()

    # ==================== ② 提交人脸采集 ====================
    def submit_face(self):
        if not self.api_client.logged_in:
            self.show_error("请先点击右上角“登录”")
            return
        if not self.face_processed or self.processed_image is None:
            self.show_error("请先选择人脸照片并点击“开始处理”，生成1寸证件照")
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
            self.show_error(
                "当前身份证号与已锁定身份的号码不一致！\n"
                "请重新点击“上传身份证信息”确认身份，避免张冠李戴。"
            )
            return

        # ---------- 人脸相似度前置校验 ----------
        if not self.face_comparator.ready:
            err = self.face_comparator.error
            if err:
                self.show_error(f"人脸相似度模型加载失败：{err}")
                return
            self.show_error("人脸相似度模型正在加载，请稍候再提交")
            return

        if self.similarity_result is None:
            self.show_error(
                "人脸相似度比对尚未完成。\n"
                "请确认身份证照片与人脸照片都已选择，稍等片刻再提交。"
            )
            return

        sim, sim_msg = self.similarity_result
        if sim is None:
            self.show_error(f"人脸相似度比对未完成：{sim_msg}")
            return

        if sim < SIMILARITY_WARN:
            if not messagebox.askyesno(
                "人脸相似度告警",
                f"人脸照片与身份证头像相似度仅 {sim*100:.1f}%，\n"
                "疑似不是同一个人。\n\n"
                "确认要继续提交吗？（强烈建议重新拍照）",
                icon="warning",
            ):
                self.set_status("已取消提交")
                return
        elif sim < SIMILARITY_PASS:
            if not messagebox.askyesno(
                "人脸相似度偏低",
                f"人脸照片与身份证头像相似度 {sim*100:.1f}%，\n"
                "低于推荐阈值，请人工确认是否为同一人。\n\n"
                "确认继续提交？",
                icon="warning",
            ):
                self.set_status("已取消提交")
                return

        # ---------- 提交前最后一道人工核对 ----------
        xm = self._get_field("name")
        face_name = os.path.basename(self.face_path or "")
        warn = ""
        if self.identity_mismatched:
            warn = "\n⚠ 与考籍不一致字段：" + "、".join(self.identity_mismatched) + "\n"
        sim_line = f"\n人脸相似度：{sim*100:.1f}%\n"
        confirm = (
            "提交前请最后核对（人脸照片将绑定到以下考生）：\n\n"
            f"考生 ksbs：{self.identity_ksbs}\n"
            f"姓名：{xm}\n"
            f"身份证号：{sfzjh}\n"
            f"人脸照片文件：{face_name}"
            + sim_line + warn +
            "\n确认是同一个人，再点“是”提交。"
        )
        if not messagebox.askyesno("提交前最后核对", confirm, icon="warning"):
            self.set_status("已取消提交")
            return

        self.set_btn(self.face_submit_btn, state="disabled", text="提交中...")
        self.set_btn(self.identity_btn, state="disabled")
        threading.Thread(
            target=self._face_worker, args=(sfzjh, xm), daemon=True
        ).start()

    def _face_worker(self, sfzjh, xm):
        extract_msg = ""
        try:
            self.set_status("① 正在编码1寸人脸照片...")
            face_b64 = self._encode_bgr_base64(self.processed_image)

            self.set_status("② 正在上传1寸人脸照片...")
            self.api_client.verify_face(sfzjh, face_b64)

            self.set_status("③ 正在提交人脸特征生成任务...")
            extract_msg = self.api_client.extract_feature(sfzjh) or "已提交"

            self.just_submitted = True

            sim, _ = self.similarity_result or (None, "")
            self._append_collection_log({
                "sfzjh": sfzjh,
                "ksbs": self.identity_ksbs,
                "xm": xm,
                "face_file": self.face_path,
                "id_file": self.id_path,
                "similarity": round(sim, 4) if sim is not None else None,
                "mismatched_fields": self.identity_mismatched,
                "extract_msg": extract_msg,
            })
            self.refresh_flow()
            self.set_status("人脸采集提交完成")

            sim_line = f"人脸相似度：{sim*100:.1f}%\n" if sim is not None else ""
            summary = (
                "提交成功！\n\n"
                f"考生 ksbs：{self.identity_ksbs}\n"
                f"姓名：{xm}\n"
                f"身份证号：{sfzjh}\n"
                + sim_line +
                f"1寸人脸照片：已上传\n"
                f"人脸特征任务：{extract_msg}\n\n"
                "是否清空当前资料，继续采集下一位？"
            )
            self.msg_queue.put(("ask_reset_with_text", summary))
        except requests.exceptions.RequestException as e:
            self.set_status("人脸采集提交失败")
            self.show_error(f"网络请求失败：{e}")
        except Exception as e:
            self.set_status("人脸采集提交失败")
            self.show_error(f"人脸采集提交失败：{e}")
        finally:
            self.set_btn(self.face_submit_btn, state="normal", text="提交人脸采集")
            self.set_btn(self.identity_btn, state="normal")
            self.refresh_flow()

    @staticmethod
    def _append_collection_log(record):
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
            with open(COLLECTION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"采集日志写入失败：{e}")

    @staticmethod
    def _encode_bgr_base64(img_bgr, quality=92):
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("证件照编码失败")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    @staticmethod
    def _encode_image_file_base64(image_path, quality=92):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"无法读取图片：{image_path}")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("身份证照片编码失败")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ==================== 主处理 ====================
    def start_process(self):
        if not self.face_path:
            self.show_error("请先选择人脸照片")
            return
        if self.bg_var.get() == "image" and not self.bg_image_path:
            self.show_error("请先选择自定义背景图")
            return

        self.set_btn(self.start_btn, state="disabled", text="处理中...")
        self.set_btn(self.save_btn, state="disabled")
        self.face_processed = False
        self._update_flow()
        threading.Thread(target=self._process_worker, daemon=True).start()

    def _process_worker(self):
        try:
            self.set_status("正在处理证件照...")

            result_img = self._make_id_photo(self.face_path)
            if result_img is not None:
                self.processed_image = result_img
                self.face_processed = True
                self.msg_queue.put(("preview_array", (self.result_preview, result_img)))
                self.set_btn(self.save_btn, state="normal")
                self.set_status("1寸照片处理完成，可以提交人脸采集")
            else:
                self.processed_image = None
                self.face_processed = False
                self.show_error("未检测到人脸，请换一张照片")
                self.set_status("处理失败")
        except Exception as e:
            self.processed_image = None
            self.face_processed = False
            self.set_status("处理失败")
            self.show_error(f"处理失败：{e}")
        finally:
            self.set_btn(self.start_btn, state="normal", text="开始处理")
            self.refresh_flow()

    # ==================== 核心算法 ====================
    def _detect_face(self, image):
        if self.face_detector is None:
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        if not results.detections:
            return None

        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box

        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(int(bbox.width * w), w - x)
        bh = min(int(bbox.height * h), h - y)
        return (x, y, bw, bh)

    def _alpha_composite(self, canvas, fg_rgba, x, y):
        ch, cw = canvas.shape[:2]
        fh, fw = fg_rgba.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(cw, x + fw)
        y2 = min(ch, y + fh)
        if x1 >= x2 or y1 >= y2:
            return

        fx1 = x1 - x
        fy1 = y1 - y
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)

        fg_crop = fg_rgba[fy1:fy2, fx1:fx2].astype(np.float32)
        fg_bgr = fg_crop[:, :, :3]
        fg_alpha = fg_crop[:, :, 3:4] / 255.0

        bg_crop = canvas[y1:y2, x1:x2].astype(np.float32)
        blended = fg_bgr * fg_alpha + bg_crop * (1.0 - fg_alpha)
        canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    def _prepare_bg_image(self, bg_path, target_w, target_h):
        bg = cv2.imread(bg_path)
        if bg is None:
            return None
        bh, bw = bg.shape[:2]
        scale = max(target_w / bw, target_h / bh)
        new_w = int(bw * scale)
        new_h = int(bh * scale)
        bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        x1 = (new_w - target_w) // 2
        y1 = (new_h - target_h) // 2
        return bg[y1:y1 + target_h, x1:x1 + target_w].copy()

    def _make_id_photo(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None

        target_w, target_h = STANDARD_SIZES[self.size_var.get()]

        if self.bg_var.get() == "image":
            canvas = self._prepare_bg_image(self.bg_image_path, target_w, target_h)
            if canvas is None:
                self.show_error("无法读取背景图")
                return None
        else:
            bg_color = self._get_bg_color()
            canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

        face_bbox = self._detect_face(image)
        if face_bbox is None:
            return None

        fx, fy, fw, fh = face_bbox

        if self.segmenter is None:
            self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_result = self.segmenter.process(rgb)
        mask = seg_result.segmentation_mask

        alpha = (mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.dilate(alpha, kernel, iterations=1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        b, g, r = cv2.split(image)
        fg_rgba = cv2.merge([b, g, r, alpha])

        scale_by_height = (target_h * HEAD_HEIGHT_RATIO) / (fh * HAIR_HEIGHT_FACTOR)
        scale_by_width = (target_w * HEAD_WIDTH_RATIO) / (fw * HAIR_WIDTH_FACTOR)
        scale = min(scale_by_height, scale_by_width)

        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        resized_fg = cv2.resize(fg_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        scaled_fx = fx * scale
        scaled_fy = fy * scale
        scaled_fw = fw * scale
        scaled_fh = fh * scale
        scaled_face_cx = scaled_fx + scaled_fw / 2
        scaled_face_cy = scaled_fy + scaled_fh / 2

        target_cx = target_w / 2
        target_cy = target_h * FACE_CENTER_Y_RATIO

        paste_x = int(target_cx - scaled_face_cx)
        paste_y = int(target_cy - scaled_face_cy)

        self._alpha_composite(canvas, resized_fg, paste_x, paste_y)
        return canvas

    # ==================== 身份证识别 ====================
    def _recognize_id_card(self, image_path):
        result = self.ocr_engine.ocr(image_path, cls=True)

        items = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                items.append({
                    "x_left": min(xs),
                    "x_right": max(xs),
                    "y_center": sum(ys) / 4,
                    "text": text,
                })

        items.sort(key=lambda i: i["y_center"])
        lines = []
        for item in items:
            placed = False
            for line in lines:
                if abs(line["y_center"] - item["y_center"]) < 15:
                    line["items"].append(item)
                    line["y_center"] = sum(i["y_center"] for i in line["items"]) / len(line["items"])
                    placed = True
                    break
            if not placed:
                lines.append({"y_center": item["y_center"], "items": [item]})

        for line in lines:
            line["items"].sort(key=lambda i: i["x_left"])
            line["text"] = " ".join(i["text"] for i in line["items"])

        raw_lines = [line["text"] for line in lines]

        id_info = {
            "name": "", "gender": "", "ethnicity": "",
            "birth": "", "address": "", "id_number": "",
            "raw_lines": raw_lines
        }

        def find_value_after_label(label, line):
            if label not in line["text"]:
                return None
            for idx, item in enumerate(line["items"]):
                if label in item["text"]:
                    after = item["text"].split(label, 1)[-1].strip(" :：")
                    if after:
                        return after
                    if idx + 1 < len(line["items"]):
                        return line["items"][idx + 1]["text"].strip()
            return None

        for line in lines:
            if not id_info["name"]:
                v = find_value_after_label("姓名", line)
                if v:
                    id_info["name"] = v
            if not id_info["gender"]:
                v = find_value_after_label("性别", line)
                if v:
                    id_info["gender"] = v
            if not id_info["ethnicity"]:
                v = find_value_after_label("民族", line)
                if v:
                    id_info["ethnicity"] = v

        birth_pattern = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
        for line in lines:
            if id_info["birth"]:
                break
            m = birth_pattern.search(line["text"])
            if m:
                id_info["birth"] = f"{m.group(1)}年{m.group(2)}月{m.group(3)}日"

        address_lines = []
        capturing = False
        for line in lines:
            if "住址" in line["text"]:
                capturing = True
                for idx, item in enumerate(line["items"]):
                    if "住址" in item["text"]:
                        after = item["text"].split("住址", 1)[-1].strip(" :：")
                        if after:
                            address_lines.append(after)
                        for j in range(idx + 1, len(line["items"])):
                            address_lines.append(line["items"][j]["text"].strip())
                        break
                continue
            if capturing:
                if "公民身份号码" in line["text"] or "身份证号" in line["text"]:
                    break
                address_lines.append(line["text"])
        if address_lines:
            id_info["address"] = "".join(address_lines).replace(" ", "")

        id_pattern = re.compile(r"\d{17}[\dXx]")
        full_text = " ".join(line["text"] for line in lines).replace(" ", "")
        m = id_pattern.search(full_text)
        if m:
            id_info["id_number"] = m.group(0).upper()
        else:
            for line in lines:
                v = find_value_after_label("公民身份号码", line)
                if v:
                    id_info["id_number"] = v
                    break

        return id_info


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoIDApp(root)
    root.mainloop()