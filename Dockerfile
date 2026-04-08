FROM python:3.11-slim

WORKDIR /app

# 使用国内镜像源加速
RUN sed -i 's|deb.debian.org|mirrors.tencent.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|deb.debian.org|mirrors.tencent.com|g' /etc/apt/sources.list 2>/dev/null || true

# 系统依赖（lxml 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.tencent.com/pypi/simple/ -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p data vectordb wiki/topics wiki/entities wiki/insights

# 暴露端口
EXPOSE 8900

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8900"]
