from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import base64

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


app = FastAPI()


def limitar(valor, minimo=0.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def calcular_liveness_score(variacao_profundidade, media_saturacao):
    score_profundidade = limitar((variacao_profundidade - 8.0) / 14.0)
    score_saturacao = limitar((190.0 - media_saturacao) / 80.0)
    return limitar((score_profundidade * 0.65) + (score_saturacao * 0.35))


ort_providers = ort.get_available_providers()
print(f"[*] ONNXRuntime providers disponiveis: {ort_providers}")
model_loaded = False

try:
    if "CUDAExecutionProvider" not in ort_providers:
        raise RuntimeError("CUDAExecutionProvider indisponivel no ONNXRuntime")

    app_face = FaceAnalysis(
        allowed_modules=["detection", "landmark_3d_68"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app_face.prepare(ctx_id=0, det_size=(640, 640))
    provider_ativo = "CUDA"
    model_loaded = True
    print("[*] Modelo Multi-Face Neural carregado com sucesso!")
except Exception as e:
    print(f"[-] Erro ao carregar na GPU, usando modo de compatibilidade: {e}")
    app_face = FaceAnalysis(allowed_modules=["detection", "landmark_3d_68"])
    app_face.prepare(ctx_id=-1, det_size=(640, 640))
    provider_ativo = "CPU"
    model_loaded = True


@app.get("/health")
async def health():
    return {
        "status": "ok" if model_loaded else "erro",
        "onnx_providers": ort_providers,
        "gpu_enabled": provider_ativo == "CUDA",
        "provider_ativo": provider_ativo,
        "model_loaded": model_loaded,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        np_array = np.frombuffer(request_object_content, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "erro", "mensagem": "Frame invalido"}

        faces = app_face.get(frame)
        faces_resultado = []

        for indice, face in enumerate(faces, start=1):
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            landmarks = face.landmark_3d_68
            profundidade_nariz = landmarks[30][2]
            profundidade_orelha = landmarks[0][2]
            variacao_profundidade = abs(profundidade_nariz - profundidade_orelha)

            y1_c, y2_c = max(0, y1), min(frame.shape[0], y2)
            x1_c, x2_c = max(0, x1), min(frame.shape[1], x2)
            rosto_recortado = frame[y1_c:y2_c, x1_c:x2_c]

            media_saturacao = 0
            if rosto_recortado.size > 0:
                hsv = cv2.cvtColor(rosto_recortado, cv2.COLOR_BGR2HSV)
                _, s, _ = cv2.split(hsv)
                media_saturacao = np.mean(s)

            real_score = calcular_liveness_score(variacao_profundidade, media_saturacao)
            spoof_score = 1.0 - real_score

            if real_score >= 0.70:
                label = "REAL"
                confianca = real_score
                cor_borda = (0, 255, 0)
            elif real_score <= 0.45:
                label = "SPOOF / FOTO"
                confianca = spoof_score
                cor_borda = (0, 0, 255)
            else:
                label = "INCERTO"
                confianca = max(real_score, spoof_score)
                cor_borda = (0, 255, 255)

            texto_label = f"{label} {confianca * 100:.0f}%"
            faces_resultado.append(
                {
                    "id": indice,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "confianca": round(float(confianca), 3),
                    "real_score": round(float(real_score), 3),
                    "spoof_score": round(float(spoof_score), 3),
                }
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), cor_borda, 3)

            texto_tamanho, _ = cv2.getTextSize(
                texto_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            largura_label = texto_tamanho[0] + 14
            topo_label = max(0, y1 - 32)
            cv2.rectangle(frame, (x1, topo_label), (x1 + largura_label, y1), cor_borda, -1)
            cv2.putText(
                frame,
                texto_label,
                (x1 + 7, max(22, y1 - 9)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "status": "sucesso",
            "quantidade_rostos": len(faces),
            "provider_ativo": provider_ativo,
            "resolucao": {
                "largura": int(frame.shape[1]),
                "altura": int(frame.shape[0]),
            },
            "faces": faces_resultado,
            "imagem_processada": f"data:image/jpeg;base64,{frame_base64}",
        }

    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IA Multi-Liveness Real-Time</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background: #1a1a1a; color: #fff; margin: 0; padding: 20px; }
            #container { display: flex; flex-direction: column; align-items: center; margin-top: 10px; }
            video, canvas { display: none; }
            #output-img { border: 4px solid #444; border-radius: 8px; width: 640px; height: 480px; background: #000; }
            #contador { margin-top: 15px; font-size: 22px; color: #aaa; font-weight: bold; }
            #status-panel { width: 640px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 14px; }
            .status-item { background: #252525; border: 1px solid #3a3a3a; border-radius: 6px; padding: 10px 12px; text-align: left; }
            .status-label { display: block; color: #9ca3af; font-size: 12px; margin-bottom: 4px; }
            .status-value { display: block; color: #f5f5f5; font-size: 18px; font-weight: 700; line-height: 1.1; }
            .status-ok { color: #22c55e; }
            .status-warn { color: #facc15; }
            .status-error { color: #ef4444; }
            @media (max-width: 720px) {
                #output-img, #status-panel { width: 100%; max-width: 640px; }
                #output-img { height: auto; }
                #status-panel { grid-template-columns: repeat(2, 1fr); }
            }
        </style>
    </head>
    <body>
        <h1>Deteccao de Vivacidade Multi-Rosto (GPU Ativa)</h1>
        <div id="container">
            <video id="video" width="640" height="480" autoplay></video>
            <canvas id="canvas" width="640" height="480"></canvas>
            <img id="output-img" />
            <div id="contador">Detectando ambiente...</div>
            <div id="status-panel">
                <div class="status-item"><span class="status-label">FPS captura</span><span class="status-value" id="fps-captura">0</span></div>
                <div class="status-item"><span class="status-label">FPS inferencia</span><span class="status-value" id="fps-inferencia">0</span></div>
                <div class="status-item"><span class="status-label">Latencia media</span><span class="status-value" id="latencia-media">0 ms</span></div>
                <div class="status-item"><span class="status-label">Rostos</span><span class="status-value" id="status-rostos">0</span></div>
                <div class="status-item"><span class="status-label">Provider</span><span class="status-value" id="provider-ativo">--</span></div>
                <div class="status-item"><span class="status-label">Resolucao</span><span class="status-value" id="resolucao-frame">640x480</span></div>
                <div class="status-item"><span class="status-label">Camera</span><span class="status-value" id="status-camera">iniciando</span></div>
                <div class="status-item"><span class="status-label">Fila</span><span class="status-value" id="status-fila">livre</span></div>
            </div>
        </div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            const outputImg = document.getElementById('output-img');
            const contadorDiv = document.getElementById('contador');
            const fpsCapturaEl = document.getElementById('fps-captura');
            const fpsInferenciaEl = document.getElementById('fps-inferencia');
            const latenciaMediaEl = document.getElementById('latencia-media');
            const statusRostosEl = document.getElementById('status-rostos');
            const providerAtivoEl = document.getElementById('provider-ativo');
            const resolucaoFrameEl = document.getElementById('resolucao-frame');
            const statusCameraEl = document.getElementById('status-camera');
            const statusFilaEl = document.getElementById('status-fila');

            let framesCapturados = 0;
            let framesInferidos = 0;
            let ultimaLeituraMetricas = performance.now();
            let latencias = [];
            let requisicaoEmAndamento = false;

            function atualizarClasseStatus(elemento, classe) {
                elemento.classList.remove('status-ok', 'status-warn', 'status-error');
                if (classe) {
                    elemento.classList.add(classe);
                }
            }

            function atualizarMetricas() {
                const agora = performance.now();
                const segundos = (agora - ultimaLeituraMetricas) / 1000;
                if (segundos < 1) {
                    return;
                }

                const latenciaMedia = latencias.length
                    ? latencias.reduce((total, valor) => total + valor, 0) / latencias.length
                    : 0;

                fpsCapturaEl.textContent = (framesCapturados / segundos).toFixed(1);
                fpsInferenciaEl.textContent = (framesInferidos / segundos).toFixed(1);
                latenciaMediaEl.textContent = `${latenciaMedia.toFixed(0)} ms`;

                framesCapturados = 0;
                framesInferidos = 0;
                latencias = [];
                ultimaLeituraMetricas = agora;
            }

            navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
                .then(stream => {
                    video.srcObject = stream;
                    statusCameraEl.textContent = "ativa";
                    atualizarClasseStatus(statusCameraEl, "status-ok");
                })
                .catch(err => {
                    contadorDiv.innerHTML = "Erro ao acessar webcam.";
                    statusCameraEl.textContent = "erro";
                    atualizarClasseStatus(statusCameraEl, "status-error");
                });

            setInterval(() => {
                if (video.srcObject && !requisicaoEmAndamento) {
                    requisicaoEmAndamento = true;
                    statusFilaEl.textContent = "processando";
                    atualizarClasseStatus(statusFilaEl, "status-warn");
                    const inicioRequisicao = performance.now();

                    context.drawImage(video, 0, 0, 640, 480);
                    framesCapturados += 1;

                    canvas.toBlob(blob => {
                        const formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');

                        fetch('/predict', { method: 'POST', body: formData })
                            .then(res => res.json())
                            .then(data => {
                                if(data.status === "sucesso") {
                                    const latencia = performance.now() - inicioRequisicao;
                                    latencias.push(latencia);
                                    framesInferidos += 1;

                                    outputImg.src = data.imagem_processada;
                                    contadorDiv.innerHTML = `Rostos na cena: ${data.quantidade_rostos}`;
                                    statusRostosEl.textContent = data.quantidade_rostos;
                                    providerAtivoEl.textContent = data.provider_ativo || "--";
                                    atualizarClasseStatus(providerAtivoEl, data.provider_ativo === "CUDA" ? "status-ok" : "status-warn");

                                    if (data.resolucao) {
                                        resolucaoFrameEl.textContent = `${data.resolucao.largura}x${data.resolucao.altura}`;
                                    }
                                }
                            })
                            .catch(err => {
                                contadorDiv.innerHTML = "Conexao perdida";
                                statusCameraEl.textContent = "sem conexao";
                                atualizarClasseStatus(statusCameraEl, "status-error");
                            })
                            .finally(() => {
                                requisicaoEmAndamento = false;
                                statusFilaEl.textContent = "livre";
                                atualizarClasseStatus(statusFilaEl, "status-ok");
                            });
                    }, 'image/jpeg', 0.5);
                }
                atualizarMetricas();
            }, 150);
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
