#!/bin/bash

sudo ubuntu-drivers autoinstall
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)  # e.g. ubuntu20.04
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt update && apt install -y nvidia-container-toolkit docker.io
nvidia-smi

mkdir -p /var/py_workspace && cd /var/py_workspace

git clone https://github.com/zhao458114067/xu_ai_assistant.git && cd xu_ai_assistant

docker build -t xu_ai_assistant .

docker run -it -d -v /vector_repo:/vector_repo -v /vector_store:/vector_store -v /var/py_workspace/xu_ai_assistant:/app --network host --gpus all xu_ai_assistant