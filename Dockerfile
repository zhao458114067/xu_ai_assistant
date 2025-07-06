FROM python:3.11

# 更换 apt 源为阿里云
RUN echo "deb https://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse\n\
deb https://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" > /etc/apt/sources.list

RUN apt-get update && apt install -y vim

# 设置 pip 永久源（阿里云）
RUN mkdir -p /root/.pip && \
    echo "[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple\ntrusted-host = mirrors.aliyun.com" > /root/.pip/pip.conf

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

RUN mkdir -p /root/nltk_data/tokenizers
ADD ./resources/punkt.zip /root/nltk_data/tokenizers