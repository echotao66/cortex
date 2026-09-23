# cortex 的生产镜像。
#
# 基础镜像刻意用 3.12 而不是本地的 3.14：chromadb / onnxruntime 这类带原生
# 扩展的包，3.12 的 manylinux wheel 是齐的；3.14 太新，很可能没有对应 wheel，
# pip 会退化成源码编译然后失败。这正是选 Docker 最实际的收益——
# 把「本机能跑、服务器装不上」这类问题一次性挡掉。
# Debian bookworm 自带的 sqlite 是 3.40，满足 chromadb 要求的 >=3.35。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MCP_FS_ROOT=/app/cortex

WORKDIR /app/cortex

# 先只拷依赖清单再装——改了业务代码不会让这一层缓存失效，重建快很多
COPY requirements.txt requirements-chroma.txt ./
RUN pip install --no-cache-dir -r requirements-chroma.txt

COPY . .

# 非 root 运行。.cortex 在运行时会被 compose 的 bind mount 覆盖掉，
# 这里先建出来只是为了让单独 `docker run` 也能起得来。
RUN useradd -m -u 10001 cortex \
 && mkdir -p /app/cortex/.cortex \
 && chown -R cortex:cortex /app
USER cortex

EXPOSE 8000

# --workers 1 是硬约束不是保守取值：chroma 的 PersistentClient 同一目录
# 同一时刻只允许一个进程打开，多 worker 会互相抢锁。
# 这里直接传 --host 0.0.0.0，不依赖 main.py 里那个「默认回环」的默认值——
# 那个默认是留给本地开发的安全默认。
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
