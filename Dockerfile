# CUDA 11.7 系 (11.7.1) での開発イメージ
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# 必要なツールをインストール
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3-dev wget curl \
    iputils-ping net-tools sudo iperf3 iproute2 && \
    rm -rf /var/lib/apt/lists/*

# Python環境を準備
RUN python3 -m pip install --upgrade pip

# PyTorchと関連ライブラリをインストール (cu117)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip install pyyaml mlflow

# 作業ディレクトリを設定
WORKDIR /workspace
ENV PYTHONPATH=/workspace





# requirements_benchmark.txtを使ってパッケージをインストール
COPY requirements_benchmark.txt /workspace/requirements_benchmark.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements_benchmark.txt
