FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/models && \
    python3 -c "import hashlib, urllib.request; url='https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx'; path='/app/models/minifasnet_v2.onnx'; urllib.request.urlretrieve(url, path); expected='d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b'; actual=hashlib.sha256(open(path,'rb').read()).hexdigest(); assert actual == expected, f'SHA256 invalido: {actual}'"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 uninstall -y onnxruntime && \
    pip3 install --no-cache-dir --force-reinstall "numpy<2" "onnxruntime-gpu==1.16.3"

COPY . .

EXPOSE 8000

CMD ["python3", "app.py"]
