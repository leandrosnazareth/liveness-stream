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
PAD_REAL_THRESHOLD = float(os.getenv("PAD_REAL_THRESHOLD", "0.78"))
PAD_SPOOF_THRESHOLD = float(os.getenv("PAD_SPOOF_THRESHOLD", "0.42"))
PAD_MIN_MOTION_SCORE = float(os.getenv("PAD_MIN_MOTION_SCORE", "0.25"))
PAD_STRONG_ATTACK_SCORE = float(os.getenv("PAD_STRONG_ATTACK_SCORE", "0.62"))
PAD_FLAT_SUPPORT_SCORE = float(os.getenv("PAD_FLAT_SUPPORT_SCORE", "0.48"))

face_tracks = {}
next_track_id = 1
frame_sequence = 0


def limitar(valor, minimo=0.0, maximo=1.0):
    return max(minimo, min(maximo, valor))


def calcular_liveness_score(variacao_profundidade, media_saturacao):
    score_profundidade = limitar((variacao_profundidade - 8.0) / 14.0)
    score_saturacao = limitar((190.0 - media_saturacao) / 80.0)
    return limitar((score_profundidade * 0.65) + (score_saturacao * 0.35))


def normalizar_intervalo(valor, minimo, maximo):
    return limitar((valor - minimo) / (maximo - minimo))


def calcular_eye_aspect_ratio(landmarks):
    try:
        olho_esq = landmarks[[36, 37, 38, 39, 40, 41]][:, :2]
        olho_dir = landmarks[[42, 43, 44, 45, 46, 47]][:, :2]

        def ear(olho):
            altura_1 = np.linalg.norm(olho[1] - olho[5])
            altura_2 = np.linalg.norm(olho[2] - olho[4])
            largura = np.linalg.norm(olho[0] - olho[3])
            return float((altura_1 + altura_2) / (2.0 * largura)) if largura else 0.0

        return (ear(olho_esq) + ear(olho_dir)) / 2.0
    except Exception:
        return 0.0


def calcular_suporte_plano(frame, bbox):
    altura, largura = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    centro_x = (x1 + x2) / 2.0
    centro_y = (y1 + y2) / 2.0
    area_frame = max(1, largura * altura)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mascara_clara = cv2.inRange(
        hsv,
        np.array([0, 0, 145], dtype=np.uint8),
        np.array([179, 80, 255], dtype=np.uint8),
    )
    kernel = np.ones((7, 7), np.uint8)
    mascara_clara = cv2.morphologyEx(mascara_clara, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(
        mascara_clara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    melhor_score = 0.0
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_frame * 0.035:
            continue
        if area > area_frame * 0.68:
            continue

        rx, ry, rw, rh = cv2.boundingRect(contorno)
        if not (rx <= centro_x <= rx + rw and ry <= centro_y <= ry + rh):
            continue
        if rx <= 3 or ry <= 3 or rx + rw >= largura - 3 or ry + rh >= altura - 3:
            continue

        area_retangulo = max(1, rw * rh)
        preenchimento = limitar(area / area_retangulo)
        tamanho_relativo = limitar(area / (area_frame * 0.45))
        borda_retangular = limitar(preenchimento, 0.0, 1.0)

        margem_x = max(0, min(centro_x - rx, rx + rw - centro_x)) / max(rw, 1)
        margem_y = max(0, min(centro_y - ry, ry + rh - centro_y)) / max(rh, 1)
        face_dentro = limitar((min(margem_x, margem_y) - 0.02) / 0.20)

        regiao = frame[ry : ry + rh, rx : rx + rw]
        if regiao.size == 0:
            continue

        regiao_hsv = cv2.cvtColor(regiao, cv2.COLOR_BGR2HSV)
        saturacao_media = float(np.mean(regiao_hsv[:, :, 1]))
        brilho_media = float(np.mean(regiao_hsv[:, :, 2]))
        papel_claro = limitar((brilho_media - 135.0) / 90.0) * (
            1.0 - normalizar_intervalo(saturacao_media, 55.0, 130.0)
        )

        score = limitar(
            (tamanho_relativo * 0.25)
            + (borda_retangular * 0.25)
            + (face_dentro * 0.25)
            + (papel_claro * 0.25)
        )
        if tamanho_relativo >= 0.18 and face_dentro >= 0.15:
            melhor_score = max(melhor_score, score)

    x1_c, x2_c = max(0, x1), min(largura, x2)
    y1_c, y2_c = max(0, y1), min(altura, y2)
    margem = int(max(x2 - x1, y2 - y1) * 0.85)
    ex1, ex2 = max(0, x1 - margem), min(largura, x2 + margem)
    ey1, ey2 = max(0, y1 - margem), min(altura, y2 + margem)
    entorno = frame[ey1:ey2, ex1:ex2]
    rosto = frame[y1_c:y2_c, x1_c:x2_c]

    if entorno.size > 0 and rosto.size > 0:
        entorno_hsv = cv2.cvtColor(entorno, cv2.COLOR_BGR2HSV)
        rosto_area = max(1, rosto.shape[0] * rosto.shape[1])
        entorno_area = max(1, entorno.shape[0] * entorno.shape[1] - rosto_area)
        pixels_papel = np.count_nonzero(
            (entorno_hsv[:, :, 1] < 70) & (entorno_hsv[:, :, 2] > 155)
        )
        proporcao_papel = limitar((pixels_papel - rosto_area) / entorno_area)
        face_area_relativa = rosto_area / max(1, (ex2 - ex1) * (ey2 - ey1))
        score_entorno = normalizar_intervalo(proporcao_papel, 0.38, 0.82) * (
            1.0 - normalizar_intervalo(face_area_relativa, 0.42, 0.72)
        )
        melhor_score = max(melhor_score, score_entorno)

    return melhor_score


def calcular_evidencias(face, frame, bbox, landmarks, variacao_profundidade):
    x1, y1, x2, y2 = bbox
    y1_c, y2_c = max(0, y1), min(frame.shape[0], y2)
    x1_c, x2_c = max(0, x1), min(frame.shape[1], x2)
    rosto_recortado = frame[y1_c:y2_c, x1_c:x2_c]

    face_score = float(getattr(face, "det_score", 0.0) or 0.0)
    qualidade_face = limitar((face_score - 0.55) / 0.40)
    profundidade_score = normalizar_intervalo(variacao_profundidade, 14.5, 32.0)
    media_saturacao = 0.0
    media_brilho = 0.0
    desvio_cor = 0.0
    nitidez = 0.0
    textura_natural = 0.0
    cor_natural = 0.0
    tela_score = 0.0
    foto_score = 0.0
    suporte_plano = calcular_suporte_plano(frame, bbox)

    if rosto_recortado.size > 0:
        hsv = cv2.cvtColor(rosto_recortado, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        cinza = cv2.cvtColor(rosto_recortado, cv2.COLOR_BGR2GRAY)
        media_saturacao = float(np.mean(s))
        media_brilho = float(np.mean(v))
        desvio_cor = float(np.mean(np.std(rosto_recortado.reshape(-1, 3), axis=0)))
        nitidez = float(cv2.Laplacian(cinza, cv2.CV_64F).var())

        textura_base = normalizar_intervalo(nitidez, 35.0, 260.0)
        textura_excessiva = normalizar_intervalo(nitidez, 700.0, 1600.0)
        textura_natural = limitar(textura_base * (1.0 - 0.55 * textura_excessiva))

        saturacao_ok = 1.0 - abs(media_saturacao - 85.0) / 95.0
        brilho_ok = 1.0 - abs(media_brilho - 135.0) / 130.0
        variacao_ok = normalizar_intervalo(desvio_cor, 18.0, 55.0)
        cor_natural = limitar((saturacao_ok * 0.35) + (brilho_ok * 0.25) + (variacao_ok * 0.40))

        brilho_uniforme = normalizar_intervalo(media_brilho, 155.0, 230.0) * (
            1.0 - normalizar_intervalo(desvio_cor, 20.0, 70.0)
        )
        saturacao_artificial = normalizar_intervalo(media_saturacao, 145.0, 210.0)
        nitidez_pixel = normalizar_intervalo(nitidez, 500.0, 1400.0)
        tela_score = limitar(
            (brilho_uniforme * 0.40)
            + (saturacao_artificial * 0.35)
            + (nitidez_pixel * 0.25)
        )

        pouca_textura = 1.0 - textura_base
        pouca_cor = 1.0 - variacao_ok
        pouca_profundidade = 1.0 - profundidade_score
        foto_score = limitar(
            (pouca_profundidade * 0.45)
            + (pouca_textura * 0.25)
            + (pouca_cor * 0.30)
        )
        foto_score = max(foto_score, suporte_plano)

    anti_spoof_instantaneo = limitar(
        (profundidade_score * 0.30)
        + (textura_natural * 0.25)
        + (cor_natural * 0.20)
        + (qualidade_face * 0.10)
        + ((1.0 - max(tela_score, foto_score, suporte_plano)) * 0.15)
    )

    return {
        "face_detectada": qualidade_face,
        "profundidade": profundidade_score,
        "textura_natural": textura_natural,
        "cor_natural": cor_natural,
        "movimento_natural": 0.0,
        "anti_spoofing": anti_spoof_instantaneo,
        "foto_score": foto_score,
        "tela_score": tela_score,
        "suporte_plano": suporte_plano,
        "variacao_profundidade": float(variacao_profundidade),
        "media_saturacao": media_saturacao,
        "media_brilho": media_brilho,
        "desvio_cor": desvio_cor,
        "nitidez": nitidez,
        "eye_aspect_ratio": calcular_eye_aspect_ratio(landmarks),
    }


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
            "depth_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "texture_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "color_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "face_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "foto_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "tela_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "flat_support_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "centers": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "areas": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "depth_values": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "eye_aspects": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "last_seen": frame_atual,
        }

    tracks_usados.add(melhor_track_id)
    return melhor_track_id


def calcular_movimento_temporal(track):
    centers = list(track["centers"])
    areas = list(track["areas"])
    depths = list(track["depth_values"])
    eyes = list(track["eye_aspects"])
    if len(centers) < TEMPORAL_MIN_FRAMES:
        return 0.0

    deslocamentos = [
        np.linalg.norm(np.array(centers[i]) - np.array(centers[i - 1]))
        for i in range(1, len(centers))
    ]
    area_media = max(sum(areas) / len(areas), 1.0)
    escala_face = area_media ** 0.5
    movimento_centro = limitar((sum(deslocamentos) / len(deslocamentos)) / (escala_face * 0.035))
    movimento_area = limitar((max(areas) - min(areas)) / (area_media * 0.12))
    movimento_profundidade = limitar((max(depths) - min(depths)) / 8.0) if depths else 0.0
    movimento_olhos = limitar((max(eyes) - min(eyes)) / 0.08) if eyes else 0.0
    return limitar(
        (movimento_centro * 0.35)
        + (movimento_area * 0.20)
        + (movimento_profundidade * 0.25)
        + (movimento_olhos * 0.20)
    )


def media_deque(valores):
    valores = list(valores)
    return sum(valores) / len(valores) if valores else 0.0


def calcular_confianca_incerto(score_medio, frames_analisados):
    progresso_temporal = limitar(frames_analisados / max(TEMPORAL_MIN_FRAMES, 1))
    proximidade_zona_cinza = 1.0 - limitar(abs(score_medio - 0.5) * 2.0)
    return limitar(
        0.50 + (progresso_temporal * 0.12) + (proximidade_zona_cinza * 0.16),
        0.50,
        0.78,
    )


def atualizar_decisao_temporal(track_id, bbox, evidencias, frame_atual):
    track = face_tracks[track_id]
    track["bbox"] = bbox
    track["last_seen"] = frame_atual
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    track["depth_scores"].append(float(evidencias["profundidade"]))
    track["texture_scores"].append(float(evidencias["textura_natural"]))
    track["color_scores"].append(float(evidencias["cor_natural"]))
    track["face_scores"].append(float(evidencias["face_detectada"]))
    track["foto_scores"].append(float(evidencias["foto_score"]))
    track["tela_scores"].append(float(evidencias["tela_score"]))
    track.setdefault("flat_support_scores", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track["flat_support_scores"].append(float(evidencias["suporte_plano"]))
    track["centers"].append(center)
    track["areas"].append(float(area))
    track["depth_values"].append(float(evidencias["variacao_profundidade"]))
    track["eye_aspects"].append(float(evidencias["eye_aspect_ratio"]))

    movimento_natural = calcular_movimento_temporal(track)
    ataque_visual = max(
        evidencias["foto_score"],
        evidencias["tela_score"],
        evidencias["suporte_plano"],
    )
    anti_spoof_instantaneo = limitar(
        (evidencias["profundidade"] * 0.25)
        + (evidencias["textura_natural"] * 0.20)
        + (evidencias["cor_natural"] * 0.15)
        + (movimento_natural * 0.25)
        + (evidencias["face_detectada"] * 0.05)
        + ((1.0 - ataque_visual) * 0.10)
    )
    track["scores"].append(float(anti_spoof_instantaneo))

    scores = list(track["scores"])
    score_medio = media_deque(track["scores"])
    profundidade_media = media_deque(track["depth_scores"])
    textura_media = media_deque(track["texture_scores"])
    cor_media = media_deque(track["color_scores"])
    face_media = media_deque(track["face_scores"])
    foto_media = media_deque(track["foto_scores"])
    tela_media = media_deque(track["tela_scores"])
    suporte_plano_media = media_deque(track["flat_support_scores"])
    estabilidade = max(scores) - min(scores) if scores else 1.0
    frames_analisados = len(scores)
    estavel = (
        frames_analisados >= TEMPORAL_MIN_FRAMES
        and estabilidade <= TEMPORAL_STABILITY_DELTA
    )
    evidencias_suficientes_real = (
        profundidade_media >= 0.45
        and textura_media >= 0.42
        and cor_media >= 0.35
        and movimento_natural >= PAD_MIN_MOTION_SCORE
        and max(foto_media, tela_media, suporte_plano_media) < PAD_FLAT_SUPPORT_SCORE
    )

    if not estavel:
        label = "INCERTO"
        confianca = calcular_confianca_incerto(score_medio, frames_analisados)
        cor_borda = (0, 255, 255)
        cor_interface = "#facc15"
    elif suporte_plano_media >= PAD_FLAT_SUPPORT_SCORE or foto_media >= PAD_STRONG_ATTACK_SCORE:
        label = "FOTO"
        confianca = max(foto_media, suporte_plano_media)
        cor_borda = (0, 0, 255)
        cor_interface = "#ef4444"
    elif tela_media >= PAD_STRONG_ATTACK_SCORE:
        label = "TELA"
        confianca = tela_media
        cor_borda = (0, 0, 255)
        cor_interface = "#ef4444"
    elif score_medio >= PAD_REAL_THRESHOLD and evidencias_suficientes_real:
        label = "REAL"
        confianca = score_medio
        cor_borda = (0, 255, 0)
        cor_interface = "#22c55e"
    elif score_medio <= PAD_SPOOF_THRESHOLD:
        label = "SPOOF"
        confianca = 1.0 - score_medio
        cor_borda = (0, 0, 255)
        cor_interface = "#ef4444"
    else:
        label = "INCERTO"
        confianca = calcular_confianca_incerto(score_medio, frames_analisados)
        cor_borda = (0, 255, 255)
        cor_interface = "#facc15"

    return {
        "label": label,
        "confianca": confianca,
        "real_score_medio": score_medio,
        "evidencias": {
            "face_detectada": face_media,
            "profundidade": profundidade_media,
            "textura_natural": textura_media,
            "cor_natural": cor_media,
            "movimento_natural": movimento_natural,
            "anti_spoofing": score_medio,
            "foto_score": foto_media,
            "tela_score": tela_media,
            "suporte_plano": suporte_plano_media,
        },
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

            evidencias = calcular_evidencias(
                face, frame, bbox, landmarks, variacao_profundidade
            )
            track_id = obter_track_id(bbox, frame_atual, tracks_usados)
            decisao_temporal = atualizar_decisao_temporal(
                track_id, bbox, evidencias, frame_atual
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
                    "face_score": round(float(evidencias["face_detectada"]), 3),
                    "anti_spoofing_score": round(
                        float(decisao_temporal["evidencias"]["anti_spoofing"]), 3
                    ),
                    "real_score_medio": round(
                        float(decisao_temporal["real_score_medio"]), 3
                    ),
                    "evidencias": {
                        chave: round(float(valor), 3)
                        for chave, valor in decisao_temporal["evidencias"].items()
                    },
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
                "pad_real_threshold": PAD_REAL_THRESHOLD,
                "pad_spoof_threshold": PAD_SPOOF_THRESHOLD,
                "pad_min_motion_score": PAD_MIN_MOTION_SCORE,
                "pad_strong_attack_score": PAD_STRONG_ATTACK_SCORE,
                "pad_flat_support_score": PAD_FLAT_SUPPORT_SCORE,
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
                <div class="status-item"><span class="status-label">Face detectada</span><span class="status-value" id="score-face">0%</span></div>
                <div class="status-item"><span class="status-label">Profundidade</span><span class="status-value" id="score-profundidade">0%</span></div>
                <div class="status-item"><span class="status-label">Textura natural</span><span class="status-value" id="score-textura">0%</span></div>
                <div class="status-item"><span class="status-label">Movimento natural</span><span class="status-value" id="score-movimento">0%</span></div>
                <div class="status-item"><span class="status-label">Anti-spoofing</span><span class="status-value" id="score-antispoofing">0%</span></div>
                <div class="status-item"><span class="status-label">Suporte plano</span><span class="status-value" id="score-suporte-plano">0%</span></div>
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
            const scoreFaceEl = document.getElementById('score-face');
            const scoreProfundidadeEl = document.getElementById('score-profundidade');
            const scoreTexturaEl = document.getElementById('score-textura');
            const scoreMovimentoEl = document.getElementById('score-movimento');
            const scoreAntispoofingEl = document.getElementById('score-antispoofing');
            const scoreSuportePlanoEl = document.getElementById('score-suporte-plano');

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
                                        if (primeiraFace.evidencias) {
                                            scoreFaceEl.textContent = `${(primeiraFace.evidencias.face_detectada * 100).toFixed(0)}%`;
                                            scoreProfundidadeEl.textContent = `${(primeiraFace.evidencias.profundidade * 100).toFixed(0)}%`;
                                            scoreTexturaEl.textContent = `${(primeiraFace.evidencias.textura_natural * 100).toFixed(0)}%`;
                                            scoreMovimentoEl.textContent = `${(primeiraFace.evidencias.movimento_natural * 100).toFixed(0)}%`;
                                            scoreAntispoofingEl.textContent = `${(primeiraFace.evidencias.anti_spoofing * 100).toFixed(0)}%`;
                                            scoreSuportePlanoEl.textContent = `${(primeiraFace.evidencias.suporte_plano * 100).toFixed(0)}%`;
                                        }
                                    } else if (data.analise_temporal) {
                                        framesTemporaisEl.textContent = `0/${data.analise_temporal.min_frames}`;
                                        estabilidadeTemporalEl.textContent = "--";
                                        atualizarClasseStatus(estabilidadeTemporalEl, "");
                                        scoreFaceEl.textContent = "0%";
                                        scoreProfundidadeEl.textContent = "0%";
                                        scoreTexturaEl.textContent = "0%";
                                        scoreMovimentoEl.textContent = "0%";
                                        scoreAntispoofingEl.textContent = "0%";
                                        scoreSuportePlanoEl.textContent = "0%";
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
