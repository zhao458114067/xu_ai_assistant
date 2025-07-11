FROM python:3.11

RUN apt-get update && apt install -y vim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 拷贝代码
COPY ./requirements.txt ./requirements.txt

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir textract python-magic

CMD python /app/src/generate_vector_store.py