# 证件照图像处理服务 (photo-processing-service)
# Linux ARM64 部署镜像，含全部 ML 模型（MediaPipe / PaddleOCR / InsightFace buffalo_l）
# 构建：docker build -t photo-processing-service .
# 运行：docker run -d -p 10006:10006 --name photo-processing photo-processing-service

FROM python:3.11-slim

# 系统依赖：
#   libgl1 / libglib2.0-0  -> opencv 运行时
#   libgomp1               -> paddle / onnxruntime 运行时
#   build-essential        -> insightface 源码编译需要 g++
#   wget / unzip           -> 下载并解压 buffalo_l 模型
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷贝依赖清单，利用 Docker 层缓存
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ==================== 模型下载 ====================

# InsightFace buffalo_l 人脸模型
RUN mkdir -p /app/models/buffalo_l \
    && (wget -q --tries=3 --timeout=60 -O /tmp/buffalo_l.zip \
            "https://ghfast.top/https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip" \
        || wget -q --tries=3 --timeout=60 -O /tmp/buffalo_l.zip \
            "https://gh-proxy.com/https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip" \
        || wget -q --tries=3 --timeout=60 -O /tmp/buffalo_l.zip \
            "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip") \
    && unzip -o /tmp/buffalo_l.zip -d /app/models/buffalo_l \
    && rm -f /tmp/buffalo_l.zip \
    && ls -la /app/models/buffalo_l

# PaddleOCR 模型：构建时预下载到 ~/.paddleocr（运行时同一路径自动加载）
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)" \
    && ls -la /root/.paddleocr

# ==================== 应用代码 ====================

COPY app.py server_main.py /app/
COPY assets /app/assets

# 服务端口
EXPOSE 10006

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10006/health', timeout=5)" || exit 1

# 启动
CMD ["python", "server_main.py"]