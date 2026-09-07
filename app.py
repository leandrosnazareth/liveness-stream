from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse
import base64
import os
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


app = FastAPI()

TEMPORAL_WINDOW_SIZE = int(os.getenv("TEMPORAL_WINDOW_SIZE", "12"))
TEMPORAL_MIN_FRAMES = int(os.getenv("TEMPORAL_MIN_FRAMES", "5"))
TEMPORAL_REAL_THRESHOLD = float(os.getenv("TEMPORAL_REAL_THRESHOLD", "0.70"))
TEMPORAL_SPOOF_THRESHOLD = float(os.getenv("TEMPORAL_SPOOF_THRESHOLD", "0.45"))
TEMPORAL_STABILITY_DELTA = float(os.getenv("TEMPORAL_STABILITY_DELTA", "0.18"))
TEMPORAL_MATCH_IOU = float(os.getenv("TEMPORAL_MATCH_IOU", "0.30"))
TEMPORAL_STALE_FRAMES = int(os.getenv("TEMPORAL_STALE_FRAMES", "20"))

face_tracks = {}
next_track_id = 1
frame_sequence = 0


def limitar(valor, minimo=0.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def calcular_liveness_score(variacao_profundidade, media_saturacao):
    score_profundidade = limitar((variacao_profundidade - 8.0) / 14.0)
    score_saturacao = limitar((190.0 - media_saturacao) / 80.0)
    return limitar((score_profundidade * 0.65) + (score_saturacao * 0.35))


def calcular_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def obter_track_id(bbox, frame_atual, tracks_usados):
    global next_track_id

    melhor_track_id = None
    melhor_iou = 0.0
    for track_id, track in face_tracks.items():
        if track_id in tracks_usados:
            continue
        iou = calcular_iou(bbox, track["bbox"])
        if iou > melhor_iou:
            melhor_iou = iou
            melhor_track_id = track_id

    if melhor_track_id is None or melhor_iou < TEMPORAL_MATCH_IOU:
        melhor_track_id = next_track_id
        next_track_id += 1
        face_tracks[melhor_track_id] = {
            "bbox": bbox,
            "scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "last_seen": frame_atual,
        }

    tracks_usados.add(melhor_track_id)
    return melhor_track_id


def atualizar_decisao_temporal(track_id, bbox, real_score, frame_atual):
    track = face_tracks[track_id]
    track["bbox"] = bbox
    track["last_seen"] = frame_atual
    track["scores"].append(float(real_score))

    scores = list(track["scores"])
    score_medio = sum(scores) / len(scores)
    estabilidade = max(scores) - min(scores) if scores else 1.0
    frames_analisados = len(scores)
    estavel = (
        frames_analisados >= TEMPORAL_MIN_FRAMES
        and estabilidade <= TEMPORAL_STABILITY_DELTA
    )

    if not estavel:
        label = "INCERTO"
        confianca = max(score_medio, 1.0 - score_medio)
        cor_borda = (0, 255, 255)
        cor_interface = "#facc15"
    elif score_medio >= TEMPORAL_REAL_THRESHOLD:
        label = "REAL"
        confianca = score_medio
        cor_borda = (0, 255, 0)
        cor_interface = "#22c55e"
    elif score_medio <= TEMPORAL_SPOOF_THRESHOLD:
        label = "SPOOF"
        confianca = 1.0 - score_medio
        cor_borda = (0, 0, 255)
        cor_interface = "#ef4444"
    else:
        label = "INCERTO"
        confianca = max(score_medio, 1.0 - score_medio)
        cor_borda = (0, 255, 255)
        cor_interface = "#facc15"

    return {
        "label": label,
        "confianca": confianca,
        "real_score_medio": score_medio,
        "frames_analisados": frames_analisados,
        "estabilidade": estabilidade,
        "estavel": estavel,
        "cor_borda": cor_borda,
        "cor_interface": cor_interface,
    }


def limpar_tracks_antigos(frame_atual):
    expirados = [
        track_id
        for track_id, track in face_tracks.items()
        if frame_atual - track["last_seen"] > TEMPORAL_STALE_FRAMES
    ]
    for track_id in expirados:
        del face_tracks[track_id]


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
async def predict(
    file: UploadFile = File(...),
    modo: str = Query("metadata", pattern="^(metadata|imagem)$"),
):
    global frame_sequence

    try:
        frame_sequence += 1
        frame_atual = frame_sequence
        inicio_total = time.perf_counter()
        request_object_content = await file.read()

        inicio_decode = time.perf_counter()
        np_array = np.frombuffer(request_object_content, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        tempo_decode_ms = (time.perf_counter() - inicio_decode) * 1000

        if frame is None:
            return {"status": "erro", "mensagem": "Frame invalido"}

        inicio_deteccao = time.perf_counter()
        faces = app_face.get(frame)
        tempo_deteccao_ms = (time.perf_counter() - inicio_deteccao) * 1000
        faces_resultado = []
        tracks_usados = set()

        inicio_liveness = time.perf_counter()
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            bbox = [int(x1), int(y1), int(x2), int(y2)]

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

            track_id = obter_track_id(bbox, frame_atual, tracks_usados)
            decisao_temporal = atualizar_decisao_temporal(
                track_id, bbox, real_score, frame_atual
            )
            label = decisao_temporal["label"]
            confianca = decisao_temporal["confianca"]
            cor_borda = decisao_temporal["cor_borda"]
            cor_interface = decisao_temporal["cor_interface"]

            texto_label = f"{label} {confianca * 100:.0f}%"
            faces_resultado.append(
                {
                    "id": track_id,
                    "bbox": bbox,
                    "label": label,
                    "confianca": round(float(confianca), 3),
                    "score": round(float(confianca), 3),
                    "real_score": round(float(real_score), 3),
                    "spoof_score": round(float(spoof_score), 3),
                    "real_score_medio": round(
                        float(decisao_temporal["real_score_medio"]), 3
                    ),
                    "frames_analisados": decisao_temporal["frames_analisados"],
                    "estabilidade": round(float(decisao_temporal["estabilidade"]), 3),
                    "estavel": decisao_temporal["estavel"],
                    "cor": cor_interface,
                }
            )

            if modo == "imagem":
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor_borda, 3)

                texto_tamanho, _ = cv2.getTextSize(
                    texto_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                largura_label = texto_tamanho[0] + 14
                topo_label = max(0, y1 - 32)
                cv2.rectangle(
                    frame, (x1, topo_label), (x1 + largura_label, y1), cor_borda, -1
                )
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
        tempo_liveness_ms = (time.perf_counter() - inicio_liveness) * 1000
        limpar_tracks_antigos(frame_atual)

        frame_base64 = None
        tempo_encode_ms = 0.0
        inicio_encode = time.perf_counter()
        if modo == "imagem":
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            tempo_encode_ms = (time.perf_counter() - inicio_encode) * 1000
        tempo_total_ms = (time.perf_counter() - inicio_total) * 1000

        resposta = {
            "status": "sucesso",
            "modo": modo,
            "quantidade_rostos": len(faces),
            "provider_ativo": provider_ativo,
            "resolucao": {
                "largura": int(frame.shape[1]),
                "altura": int(frame.shape[0]),
            },
            "faces": faces_resultado,
            "analise_temporal": {
                "janela": TEMPORAL_WINDOW_SIZE,
                "min_frames": TEMPORAL_MIN_FRAMES,
                "real_threshold": TEMPORAL_REAL_THRESHOLD,
                "spoof_threshold": TEMPORAL_SPOOF_THRESHOLD,
                "stability_delta": TEMPORAL_STABILITY_DELTA,
                "tracks_ativos": len(face_tracks),
            },
            "metricas": {
                "decode_ms": round(tempo_decode_ms, 2),
                "deteccao_ms": round(tempo_deteccao_ms, 2),
                "liveness_ms": round(tempo_liveness_ms, 2),
                "encode_ms": round(tempo_encode_ms, 2),
                "total_ms": round(tempo_total_ms, 2),
            },
        }
        if frame_base64 is not None:
            resposta["imagem_processada"] = f"data:image/jpeg;base64,{frame_base64}"
        return resposta

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
            video { display: none; }
            canvas, #output-img { border: 4px solid #444; border-radius: 8px; width: 640px; height: 480px; background: #000; }
            #output-img { display: none; }
            #contador { margin-top: 15px; font-size: 22px; color: #aaa; font-weight: bold; }
            #modo-controle { display: flex; gap: 8px; margin: 0 0 12px; }
            .modo-btn { border: 1px solid #3a3a3a; border-radius: 6px; background: #252525; color: #d4d4d4; cursor: pointer; font-weight: 700; padding: 9px 14px; }
            .modo-btn.ativo { background: #0f766e; border-color: #14b8a6; color: #fff; }
            #status-panel { width: 640px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 14px; }
            .status-item { background: #252525; border: 1px solid #3a3a3a; border-radius: 6px; padding: 10px 12px; text-align: left; }
            .status-label { display: block; color: #9ca3af; font-size: 12px; margin-bottom: 4px; }
            .status-value { display: block; color: #f5f5f5; font-size: 18px; font-weight: 700; line-height: 1.1; }
            .status-ok { color: #22c55e; }
            .status-warn { color: #facc15; }
            .status-error { color: #ef4444; }
            @media (max-width: 720px) {
                canvas, #output-img, #status-panel { width: 100%; max-width: 640px; }
                canvas, #output-img { height: auto; }
                #status-panel { grid-template-columns: repeat(2, 1fr); }
            }
        </style>
    </head>
    <body>
        <h1>Deteccao de Vivacidade Multi-Rosto (GPU Ativa)</h1>
        <div id="container">
            <div id="modo-controle">
                <button class="modo-btn ativo" id="modo-metadata" type="button">Metadados</button>
                <button class="modo-btn" id="modo-imagem" type="button">Imagem completa</button>
            </div>
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
                <div class="status-item"><span class="status-label">Decode</span><span class="status-value" id="tempo-decode">0 ms</span></div>
                <div class="status-item"><span class="status-label">Deteccao</span><span class="status-value" id="tempo-deteccao">0 ms</span></div>
                <div class="status-item"><span class="status-label">Liveness</span><span class="status-value" id="tempo-liveness">0 ms</span></div>
                <div class="status-item"><span class="status-label">Encode</span><span class="status-value" id="tempo-encode">0 ms</span></div>
                <div class="status-item"><span class="status-label">Total backend</span><span class="status-value" id="tempo-total-backend">0 ms</span></div>
                <div class="status-item"><span class="status-label">Frames temporais</span><span class="status-value" id="frames-temporais">0/0</span></div>
                <div class="status-item"><span class="status-label">Estabilidade</span><span class="status-value" id="estabilidade-temporal">--</span></div>
            </div>
        </div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            const outputImg = document.getElementById('output-img');
            const contadorDiv = document.getElementById('contador');
            const modoMetadataBtn = document.getElementById('modo-metadata');
            const modoImagemBtn = document.getElementById('modo-imagem');
            const fpsCapturaEl = document.getElementById('fps-captura');
            const fpsInferenciaEl = document.getElementById('fps-inferencia');
            const latenciaMediaEl = document.getElementById('latencia-media');
            const statusRostosEl = document.getElementById('status-rostos');
            const providerAtivoEl = document.getElementById('provider-ativo');
            const resolucaoFrameEl = document.getElementById('resolucao-frame');
            const statusCameraEl = document.getElementById('status-camera');
            const statusFilaEl = document.getElementById('status-fila');
            const tempoDecodeEl = document.getElementById('tempo-decode');
            const tempoDeteccaoEl = document.getElementById('tempo-deteccao');
            const tempoLivenessEl = document.getElementById('tempo-liveness');
            const tempoEncodeEl = document.getElementById('tempo-encode');
            const tempoTotalBackendEl = document.getElementById('tempo-total-backend');
            const framesTemporaisEl = document.getElementById('frames-temporais');
            const estabilidadeTemporalEl = document.getElementById('estabilidade-temporal');

            let framesCapturados = 0;
            let framesInferidos = 0;
            let ultimaLeituraMetricas = performance.now();
            let latencias = [];
            let requisicaoEmAndamento = false;
            let modoRetorno = "metadata";

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

            function selecionarModo(novoModo) {
                modoRetorno = novoModo;
                modoMetadataBtn.classList.toggle('ativo', novoModo === "metadata");
                modoImagemBtn.classList.toggle('ativo', novoModo === "imagem");
                canvas.style.display = novoModo === "metadata" ? "block" : "none";
                outputImg.style.display = novoModo === "imagem" ? "block" : "none";
            }

            function desenharFacesLocalmente(faces) {
                faces.forEach(face => {
                    const [x1, y1, x2, y2] = face.bbox;
                    const cor = face.cor || "#22c55e";
                    const texto = `${face.label} ${(face.confianca * 100).toFixed(0)}%`;

                    context.lineWidth = 3;
                    context.strokeStyle = cor;
                    context.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    context.font = "bold 16px Arial";
                    const larguraTexto = context.measureText(texto).width + 14;
                    const topoLabel = Math.max(0, y1 - 32);
                    context.fillStyle = cor;
                    context.fillRect(x1, topoLabel, larguraTexto, y1 - topoLabel);
                    context.fillStyle = "#ffffff";
                    context.fillText(texto, x1 + 7, Math.max(20, y1 - 10));
                });
            }

            modoMetadataBtn.addEventListener('click', () => selecionarModo("metadata"));
            modoImagemBtn.addEventListener('click', () => selecionarModo("imagem"));
            selecionarModo("metadata");

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

                        fetch(`/predict?modo=${modoRetorno}`, { method: 'POST', body: formData })
                            .then(res => res.json())
                            .then(data => {
                                if(data.status === "sucesso") {
                                    const latencia = performance.now() - inicioRequisicao;
                                    latencias.push(latencia);
                                    framesInferidos += 1;

                                    if (modoRetorno === "imagem" && data.imagem_processada) {
                                        outputImg.src = data.imagem_processada;
                                    } else {
                                        desenharFacesLocalmente(data.faces || []);
                                    }
                                    contadorDiv.innerHTML = `Rostos na cena: ${data.quantidade_rostos}`;
                                    statusRostosEl.textContent = data.quantidade_rostos;
                                    providerAtivoEl.textContent = data.provider_ativo || "--";
                                    atualizarClasseStatus(providerAtivoEl, data.provider_ativo === "CUDA" ? "status-ok" : "status-warn");

                                    if (data.resolucao) {
                                        resolucaoFrameEl.textContent = `${data.resolucao.largura}x${data.resolucao.altura}`;
                                    }

                                    if (data.metricas) {
                                        tempoDecodeEl.textContent = `${data.metricas.decode_ms} ms`;
                                        tempoDeteccaoEl.textContent = `${data.metricas.deteccao_ms} ms`;
                                        tempoLivenessEl.textContent = `${data.metricas.liveness_ms} ms`;
                                        tempoEncodeEl.textContent = `${data.metricas.encode_ms} ms`;
                                        tempoTotalBackendEl.textContent = `${data.metricas.total_ms} ms`;
                                    }

                                    const primeiraFace = (data.faces || [])[0];
                                    if (data.analise_temporal && primeiraFace) {
                                        framesTemporaisEl.textContent = `${primeiraFace.frames_analisados}/${data.analise_temporal.min_frames}`;
                                        estabilidadeTemporalEl.textContent = primeiraFace.estavel ? "estavel" : "analisando";
                                        atualizarClasseStatus(estabilidadeTemporalEl, primeiraFace.estavel ? "status-ok" : "status-warn");
                                    } else if (data.analise_temporal) {
                                        framesTemporaisEl.textContent = `0/${data.analise_temporal.min_frames}`;
                                        estabilidadeTemporalEl.textContent = "--";
                                        atualizarClasseStatus(estabilidadeTemporalEl, "");
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
