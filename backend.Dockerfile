# NexusMind 后端镜像
FROM python:3.12-slim

WORKDIR /app
ARG PIP_INDEX_URL=https://pypi.org/simple

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url "$PIP_INDEX_URL" \
    && pip check

# 拷贝业务代码
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY cli.py ./
# Runtime/demo data is provisioned into a volume, never copied from a developer.
RUN mkdir -p /app/data
# 拷贝知识库文档（schema 文档等）
COPY docs/ ./docs/

# 容器内访问同一 compose 网络的兄弟容器
ENV PYTHONPATH=/app/backend

EXPOSE 8000

WORKDIR /app/backend

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
