# NexusMind 后端镜像
FROM python:3.12-slim

WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir jieba "pymilvus[milvus_lite]" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝业务代码
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY cli.py ./
# 拷贝数据文件（ChatBI 电商数据库）
COPY data/ ./data/
# 拷贝知识库文档（schema 文档等）
COPY docs/ ./docs/

# 容器内访问同一 compose 网络的兄弟容器
ENV PYTHONPATH=/app/backend

EXPOSE 8000

WORKDIR /app/backend

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
