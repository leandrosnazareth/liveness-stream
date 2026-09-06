from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import onnxruntime as ort
import base64
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

try:
    if "CUDAExecutionProvider" not in ort_providers:
        raise RuntimeError("CUDAExecutionProvider indisponivel no ONNXRuntime")

    # Inicializa o ecossistema com suporte duplo priorizando os núcleos CUDA da sua RTX
    app_face = FaceAnalysis(
        allowed_modules=['detection', 'landmark_3d_68'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    app_face.prepare(ctx_id=0, det_size=(640, 640))
    print("[*] Modelo Multi-Face Neural carregado com sucesso!")
except Exception as e:
    print(f"[-] Erro ao carregar na GPU, usando modo de compatibilidade: {e}")
    app_face = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68'])
    app_face.prepare(ctx_id=-1, det_size=(640, 640))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        request_object_content = await file.read()
        np_array = np.frombuffer(request_object_content, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"status": "erro", "mensagem": "Frame inválido"}

        # Detecta múltiplos rostos simultaneamente na cena
        faces = app_face.get(frame)

        faces_resultado = []

        for indice, face in enumerate(faces, start=1):
            # Extrai os cantos da caixa delimitadora do rosto atual
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Extrai a malha geométrica 3D do rosto
            landmarks = face.landmark_3d_68
            
            # Métrica de Profundidade Baseada em Tensores (Distância entre os eixos Z)
            profundidade_nariz = landmarks[30][2]
            profundidade_orelha = landmarks[0][2]
            variacao_profundidade = abs(profundidade_nariz - profundidade_orelha)

            # Análise local de saturação e reflexo focada apenas no quadrado do rosto
            y1_c, y2_c = max(0, y1), min(frame.shape[0], y2)
            x1_c, x2_c = max(0, x1), min(frame.shape[1], x2)
            rosto_recortado = frame[y1_c:y2_c, x1_c:x2_c]
            
            media_saturacao = 0
            if rosto_recortado.size > 0:
                hsv = cv2.cvtColor(rosto_recortado, cv2.COLOR_BGR2HSV)
                _, s, _ = cv2.split(hsv)
                media_saturacao = np.mean(s)

            # --- Regra de Decisão do Liveness (Nomenclatura 100% Limpa) ---
            real_score = calcular_liveness_score(variacao_profundidade, media_saturacao)
            spoof_score = 1.0 - real_score

            if real_score >= 0.70:
                label = "REAL"
                confianca = real_score
                cor_borda = (0, 255, 0)  # Verde em BGR
            elif real_score <= 0.45:
                label = "SPOOF / FOTO"
                confianca = spoof_score
                cor_borda = (0, 0, 255)  # Vermelho em BGR
            else:
                label = "INCERTO"
                confianca = max(real_score, spoof_score)
                cor_borda = (0, 255, 255)  # Amarelo em BGR

            texto_label = f"{label} {confianca * 100:.0f}%"
            faces_resultado.append({
                "id": indice,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "label": label,
                "confianca": round(float(confianca), 3),
                "real_score": round(float(real_score), 3),
                "spoof_score": round(float(spoof_score), 3),
            })

            # Desenha o retângulo ao redor do rosto identificado
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor_borda, 3)
            
            # Adiciona o fundo e o texto da etiqueta acima da cabeça
            texto_tamanho, _ = cv2.getTextSize(texto_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            largura_label = texto_tamanho[0] + 14
            topo_label = max(0, y1 - 32)
            cv2.rectangle(frame, (x1, topo_label), (x1 + largura_label, y1), cor_borda, -1)
            cv2.putText(frame, texto_label, (x1 + 7, max(22, y1 - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Transforma o frame processado com os desenhos de volta para codificação Web
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "sucesso",
            "quantidade_rostos": len(faces),
            "faces": faces_resultado,
            "imagem_processada": f"data:image/jpeg;base64,{frame_base64}"
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
        </style>
    </head>
    <body>
        <h1>Detecção de Vivacidade Multi-Rosto (GPU Ativa)</h1>
        <div id="container">
            <video id="video" width="640" height="480" autoplay></video>
            <canvas id="canvas" width="640" height="480"></canvas>
            <img id="output-img" />
            <div id="contador">Detectando ambiente...</div>
        </div>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            const outputImg = document.getElementById('output-img');
            const contadorDiv = document.getElementById('contador');

            navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { contadorDiv.innerHTML = "Erro ao acessar webcam."; });

            setInterval(() => {
                if (video.srcObject) {
                    context.drawImage(video, 0, 0, 640, 480);

                    canvas.toBlob(blob => {
                        const formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');

                        fetch('/predict', { method: 'POST', body: formData })
                            .then(res => res.json())
                            .then(data => {
                                if(data.status === "sucesso") {
                                    outputImg.src = data.imagem_processada;
                                    contadorDiv.innerHTML = `Rostos na cena: ${data.quantidade_rostos}`;
                                }
                            })
                            .catch(err => { contadorDiv.innerHTML = "Conexão perdida"; });
                    }, 'image/jpeg', 0.5);
                }
            }, 150);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
