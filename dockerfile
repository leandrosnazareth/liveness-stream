FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 uninstall -y onnxruntime && \
    pip3 install --no-cache-dir --force-reinstall "numpy<2" "onnxruntime-gpu==1.16.3"

COPY . .

EXPOSE 8000

CMD ["python3", "app.py"]
