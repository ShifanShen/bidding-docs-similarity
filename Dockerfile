FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装uv
RUN pip install uv

# 通过uv安装所有的依赖
RUN uv pip install -e .

# 创建必要的目录
RUN mkdir -p tmp_files

# 暴露端口
EXPOSE 8020

# 镜像运行时执行的命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]