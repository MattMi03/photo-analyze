"""
证件照图像处理微服务 (photo-processing-service / PhotoIDServer.exe)
==================================================================
封装原桌面工具 main.py 中的核心 ML 图像处理能力：
  - 证件照人像抠图（MediaPipe），返回透明 PNG + 定位元数据，客户端本地合成并支持拖动微调
  - 证件照一键生成（合成内置背景 + 标准尺寸裁剪）
  - 身份证 OCR 识别（PaddleOCR）
  - 人脸相似度比对（InsightFace / buffalo_l）

客户端（PhotoIDClient.exe）与 Java 后端 student-affair-service 均通过 HTTP 调用本服务。

源码运行：
    pip install -r requirements.txt
    python server_main.py            # 或 uvicorn app:app --host 0.0.0.0 --port 10006
"""

import os
import re
import sys
import shutil
import base64
import logging
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("photo-processing-service")


# ==================== 打包资源路径 ====================

def resource_path(*parts):
    """获取内置资源路径：PyInstaller 打包后从 sys._MEIPASS 查找，源码运行时从脚本目录查找"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)


def ensure_paddle_models():
    """
    打包运行时：把 exe 内置的 PaddleOCR 模型复制到用户目录 ~/.paddleocr，
    PaddleOCR 按默认路径即可找到模型。
    """
    if not getattr(sys, "frozen", False) or not hasattr(sys, "_MEIPASS"):
        return
    src = os.path.join(sys._MEIPASS, "paddleocr_models")
    dst = os.path.expanduser("~/.paddleocr")
    if not os.path.isdir(src):
        log.warning(f"[ensure_paddle_models] 内置模型目录不存在：{src}")
        return
    if os.path.isdir(dst) and any(os.scandir(dst)):
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


# 固定背景图（随服务端一起打包，不可更换）
DEFAULT_BG_IMAGE = resource_path("assets", "default_bg.jpg")

# ========== 标准证件照尺寸（300 DPI，单位：像素） ==========
STANDARD_SIZES = {
    "一寸   (25×35mm)": (295, 413),
    "小一寸 (22×32mm)": (260, 378),
    "大一寸 (33×48mm)": (390, 567),
    "二寸   (35×49mm)": (413, 579),
    "小二寸 (35×45mm)": (413, 531),
    "大二寸 (35×53mm)": (413, 626),
}

# ========== 证件照构图参数（客户端按相同参数做几何计算） ==========
HEAD_TOP_MARGIN = 0.10
MAX_HEAD_HEIGHT_RATIO = 0.75
MAX_HEAD_WIDTH_RATIO = 0.90
HAIR_WIDTH_FACTOR = 1.15
HAIR_HEIGHT_FACTOR = 1.55

# ========== 人脸相似度阈值 ==========
SIMILARITY_PASS = 0.50
SIMILARITY_WARN = 0.35


# ==================== Base64 编解码 ====================

def b64_to_bgr(b64: str) -> np.ndarray:
    """Base64 -> BGR ndarray"""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片，请检查 Base64 是否正确")
    return img


def bgr_to_b64(img_bgr: np.ndarray, quality: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("图片编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def bgra_to_png_b64(img_bgra: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgra)
    if not ok:
        raise ValueError("PNG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ==================== 请求/响应模型 ====================

class IdPhotoGenerateRequest(BaseModel):
    facePhotoBase64: str
    sizeName: Optional[str] = None


class IdPhotoSegmentRequest(BaseModel):
    facePhotoBase64: str


class IdCardOcrRequest(BaseModel):
    idCardPhotoBase64: str


class FaceCompareRequest(BaseModel):
    idCardPhotoBase64: str
    facePhotoBase64: str


class ApiResult(BaseModel):
    code: int = 200
    msg: str = "success"
    data: Optional[dict] = None


# ==================== 人像检测 / 抠图 / 合成 ====================

class PortraitEngine:
    """MediaPipe 人脸检测 + 人像分割（单例懒加载）"""

    def __init__(self):
        import mediapipe as mp
        self._mp = mp
        self._face_detection = None
        self._id_face_detection = None
        self._segmenter = None

    def _get_face_detector(self):
        if self._face_detection is None:
            self._face_detection = self._mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        return self._face_detection

    def _get_id_face_detector(self):
        if self._id_face_detection is None:
            self._id_face_detection = self._mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
        return self._id_face_detection

    def _get_segmenter(self):
        if self._segmenter is None:
            self._segmenter = self._mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        return self._segmenter

    def detect_face(self, image: np.ndarray, id_card_mode: bool = False):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector = self._get_id_face_detector() if id_card_mode else self._get_face_detector()
        results = detector.process(rgb)
        if not results.detections:
            return None
        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(int(bbox.width * w), w - x)
        bh = min(int(bbox.height * h), h - y)
        return (x, y, bw, bh)

    def segment(self, image: np.ndarray):
        """
        返回 (fg_bgra, mask, face_bbox, person_top, person_bottom)
        face_bbox / person 检测失败时抛 ValueError
        """
        face_bbox = self.detect_face(image)
        if face_bbox is None:
            raise ValueError("未检测到人脸，请换一张照片")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_result = self._get_segmenter().process(rgb)
        mask = seg_result.segmentation_mask

        alpha = (mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.dilate(alpha, kernel, iterations=1)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        b, g, r = cv2.split(image)
        fg_bgra = cv2.merge([b, g, r, alpha])

        src_h, src_w = image.shape[:2]
        row_counts = (mask > 0.5).sum(axis=1)
        rows = np.where(row_counts > max(2, src_w * 0.02))[0]
        if len(rows) == 0:
            raise ValueError("未检测到人像区域")
        person_top = int(rows[0])
        person_bottom = int(rows[-1])

        return fg_bgra, mask, face_bbox, person_top, person_bottom

    def crop_id_card_portrait(self, id_img_bgr: np.ndarray):
        """在身份证照片里找到人像小头像，裁剪、放大、锐化；检测不到时返回原图"""
        h, w = id_img_bgr.shape[:2]
        results = self._get_id_face_detector().process(
            cv2.cvtColor(id_img_bgr, cv2.COLOR_BGR2RGB)
        )
        if not results.detections:
            return id_img_bgr, None
        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = min(int(bbox.width * w), w - x)
        bh = min(int(bbox.height * h), h - y)
        if bw < 20 or bh < 20:
            return id_img_bgr, None
        pad_x, pad_y = int(bw * 0.4), int(bh * 0.4)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        crop = id_img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return id_img_bgr, None
        ch, cw = crop.shape[:2]
        target = 224
        scale = max(target / cw, target / ch, 1.0)
        crop_big = cv2.resize(
            crop, (int(cw * scale), int(ch * scale)),
            interpolation=cv2.INTER_LANCZOS4,
        )
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        crop_big = cv2.filter2D(crop_big, -1, kernel)
        return crop_big, (x, y, bw, bh)


def alpha_composite(canvas, fg_bgra, x, y):
    """把带 alpha 的前景贴到 canvas 上（越界自动裁剪）"""
    ch, cw = canvas.shape[:2]
    fh, fw = fg_bgra.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2 = min(cw, x + fw)
    y2 = min(ch, y + fh)
    if x1 >= x2 or y1 >= y2:
        return
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    fg_crop = fg_bgra[fy1:fy2, fx1:fx2].astype(np.float32)
    fg_bgr = fg_crop[:, :, :3]
    fg_alpha = fg_crop[:, :, 3:4] / 255.0
    bg_crop = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = fg_bgr * fg_alpha + bg_crop * (1.0 - fg_alpha)
    canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)


def prepare_bg_image(bg_path, target_w, target_h):
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


def calc_geometry(src_w, src_h, face_bbox, person_top, person_span, target_w, target_h):
    """
    统一几何计算（客户端拖动微调时也用同一套参数）：
    返回 scale, paste_x, paste_y, new_w, new_h
    """
    fx, fy, fw, fh = face_bbox
    scale_fill = (target_h * (1.0 - HEAD_TOP_MARGIN)) / person_span
    scale_cap_h = (target_h * MAX_HEAD_HEIGHT_RATIO) / (fh * HAIR_HEIGHT_FACTOR)
    scale_cap_w = (target_w * MAX_HEAD_WIDTH_RATIO) / (fw * HAIR_WIDTH_FACTOR)
    scale = min(scale_fill, scale_cap_h, scale_cap_w)

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    paste_x = int(target_w / 2 - (fx + fw / 2) * scale)
    paste_y = int(target_h * HEAD_TOP_MARGIN - person_top * scale)
    return scale, paste_x, paste_y, new_w, new_h


# ==================== 身份证 OCR ====================

class IdCardOcr:
    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        return self._ocr

    def recognize(self, img_bgr: np.ndarray) -> dict:
        result = self._get_ocr().ocr(img_bgr, cls=True)
        items = []
        if result and result[0]:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                items.append({
                    "x_left": min(xs), "x_right": max(xs),
                    "y_center": sum(ys) / 4, "text": text,
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
        info = {"name": "", "gender": "", "ethnicity": "",
                "birth": "", "address": "", "idNumber": "",
                "rawLines": raw_lines}

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
            if not info["name"]:
                v = find_value_after_label("姓名", line)
                if v: info["name"] = v
            if not info["gender"]:
                v = find_value_after_label("性别", line)
                if v: info["gender"] = v
            if not info["ethnicity"]:
                v = find_value_after_label("民族", line)
                if v: info["ethnicity"] = v

        birth_pattern = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
        for line in lines:
            if info["birth"]:
                break
            m = birth_pattern.search(line["text"])
            if m:
                info["birth"] = f"{m.group(1)}年{m.group(2)}月{m.group(3)}日"

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
            info["address"] = "".join(address_lines).replace(" ", "")

        id_pattern = re.compile(r"\d{17}[\dXx]")
        full_text = " ".join(line["text"] for line in lines).replace(" ", "")
        m = id_pattern.search(full_text)
        if m:
            info["idNumber"] = m.group(0).upper()
        else:
            for line in lines:
                v = find_value_after_label("公民身份号码", line)
                if v:
                    info["idNumber"] = v
                    break

        return info


# ==================== 人脸相似度比对 ====================

class FaceComparator:
    def __init__(self):
        self._app = None
        self._ready = False
        self._error = None
        self._model_name = "buffalo_l"
        self._required_files = ["det_10g.onnx", "w600k_r50.onnx"]

    def _find_model_root(self):
        """返回 (root, model_dir)，依次查找：打包内置目录 / 工作目录 / 用户目录"""
        candidates = []
        # 1) PyInstaller 打包内置
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            candidates.append(sys._MEIPASS)
        # 2) 服务脚本同级目录
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
        # 3) 工作目录
        candidates.append(os.getcwd())
        # 4) 用户目录
        candidates.append(os.path.expanduser("~/.insightface"))

        for root in candidates:
            if not root or not os.path.isdir(root):
                continue
            model_dir = os.path.join(root, "models", self._model_name)
            if not os.path.isdir(model_dir):
                continue
            if all(os.path.isfile(os.path.join(model_dir, f)) for f in self._required_files):
                return root, model_dir
        return None, None

    def init(self):
        try:
            root, model_dir = self._find_model_root()
            if root is None:
                self._error = "未找到 buffalo_l 模型"
                log.error("FaceComparator 未找到模型，请将 buffalo_l 放到 models/buffalo_l/ 或 ~/.insightface/models/buffalo_l/")
                return
            log.info(f"FaceComparator 使用模型目录：{model_dir}")
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name=self._model_name, root=root,
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._ready = True
            log.info("FaceComparator 模型加载完成")
        except Exception as e:
            self._error = str(e)
            log.exception(f"FaceComparator 模型加载失败：{e}")

    @property
    def ready(self):
        return self._ready

    @property
    def error(self):
        return self._error

    def _get_embedding(self, img_bgr):
        if not self._ready:
            return None
        try:
            faces = self._app.get(img_bgr)
            if not faces:
                return None
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return face.normed_embedding
        except Exception as e:
            log.warning(f"特征提取失败：{e}")
            return None

    def compare(self, id_img_bgr, face_img_bgr, engine: "PortraitEngine"):
        if not self._ready:
            return None, f"人脸比对模块未就绪（{self._error or '加载中'}）"
        id_enhanced, bbox = engine.crop_id_card_portrait(id_img_bgr)
        if bbox is None:
            log.info("[比对] 身份证上未检测到人像小头像，使用原图")
        e1 = self._get_embedding(id_enhanced)
        if e1 is None:
            return None, "身份证照片中未检测到人脸"
        e2 = self._get_embedding(face_img_bgr)
        if e2 is None:
            return None, "人脸照片中未检测到人脸"
        return float(np.dot(e1, e2)), "ok"


# ==================== FastAPI 应用 ====================

app = FastAPI(title="证件照图像处理服务", version="1.1.0")

_engine = PortraitEngine()
_id_card_ocr = IdCardOcr()
_face_comparator = FaceComparator()


@app.on_event("startup")
def on_startup():
    ensure_paddle_models()
    bg_exists = os.path.exists(DEFAULT_BG_IMAGE)
    log.info(f"默认背景图: {DEFAULT_BG_IMAGE} (存在={bg_exists})")
    log.info("MediaPipe / PaddleOCR 模型懒加载；InsightFace 后台预加载")
    import threading
    threading.Thread(target=_face_comparator.init, daemon=True).start()


def _ok(data: dict, msg: str = "success") -> dict:
    return {"code": 200, "msg": msg, "data": data}


def _err(msg: str, code: int = 400) -> dict:
    return {"code": code, "msg": msg, "data": None}


@app.post("/api/photo/id-photo/segment")
def segment_id_photo(req: IdPhotoSegmentRequest):
    """
    人像抠图接口（供客户端拖动微调）：
    返回原始分辨率透明 PNG + 人脸框/人像上下边界，客户端本地按统一几何参数合成。
    """
    try:
        face_bgr = b64_to_bgr(req.facePhotoBase64)
    except ValueError as e:
        return _err(str(e))
    try:
        fg_bgra, _mask, face_bbox, person_top, person_bottom = _engine.segment(face_bgr)
    except ValueError as e:
        return _err(str(e))
    except Exception as e:
        log.exception("人像抠图异常")
        return _err(f"人像抠图异常：{e}", code=500)

    h, w = face_bgr.shape[:2]
    return _ok({
        "fgPngBase64": bgra_to_png_b64(fg_bgra),
        "srcWidth": w,
        "srcHeight": h,
        "faceX": face_bbox[0], "faceY": face_bbox[1],
        "faceW": face_bbox[2], "faceH": face_bbox[3],
        "personTop": person_top,
        "personBottom": person_bottom,
    }, "人像抠图成功")


@app.post("/api/photo/id-photo/generate")
def generate_id_photo(req: IdPhotoGenerateRequest):
    """一键生成（Java 后端调用）：抠图 + 合成内置背景 + 标准尺寸"""
    try:
        face_bgr = b64_to_bgr(req.facePhotoBase64)
    except ValueError as e:
        return _err(str(e))

    size_name = req.sizeName or "一寸   (25×35mm)"
    if size_name not in STANDARD_SIZES:
        size_name = "一寸   (25×35mm)"
    target_w, target_h = STANDARD_SIZES[size_name]

    canvas = prepare_bg_image(DEFAULT_BG_IMAGE, target_w, target_h)
    if canvas is None:
        return _err(f"无法读取内置背景图：{DEFAULT_BG_IMAGE}", code=500)

    try:
        fg_bgra, _mask, face_bbox, person_top, person_bottom = _engine.segment(face_bgr)
        src_h, src_w = face_bgr.shape[:2]
        person_span = max(1, person_bottom - person_top)
        _, paste_x, paste_y, new_w, new_h = calc_geometry(
            src_w, src_h, face_bbox, person_top, person_span, target_w, target_h
        )
        resized_fg = cv2.resize(fg_bgra, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        alpha_composite(canvas, resized_fg, paste_x, paste_y)
    except ValueError as e:
        return _err(str(e))
    except Exception as e:
        log.exception("证件照生成异常")
        return _err(f"证件照生成异常：{e}", code=500)

    return _ok({
        "photoBase64": bgr_to_b64(canvas),
        "contentType": "image/jpeg",
        "targetWidth": target_w,
        "targetHeight": target_h,
        "sizeName": size_name,
    }, "证件照生成成功")


@app.post("/api/photo/id-card/ocr")
def ocr_id_card(req: IdCardOcrRequest):
    try:
        img_bgr = b64_to_bgr(req.idCardPhotoBase64)
    except ValueError as e:
        return _err(str(e))
    try:
        info = _id_card_ocr.recognize(img_bgr)
    except Exception as e:
        log.exception("身份证识别异常")
        return _err(f"身份证识别异常：{e}", code=500)
    return _ok(info, "身份证识别成功")


@app.post("/api/photo/face/compare")
def compare_faces(req: FaceCompareRequest):
    try:
        id_bgr = b64_to_bgr(req.idCardPhotoBase64)
        face_bgr = b64_to_bgr(req.facePhotoBase64)
    except ValueError as e:
        return _err(str(e))

    sim, msg = _face_comparator.compare(id_bgr, face_bgr, _engine)
    level, passed = "fail", False
    if sim is not None:
        passed = sim >= SIMILARITY_PASS
        level = "pass" if sim >= SIMILARITY_PASS else ("warn" if sim >= SIMILARITY_WARN else "fail")
    return _ok({
        "similarity": sim, "passed": passed, "level": level, "msg": msg,
    }, msg)


@app.get("/health")
def health():
    return {"code": 200, "msg": "ok",
            "data": {"faceComparatorReady": _face_comparator.ready,
                     "faceComparatorError": _face_comparator.error}}
