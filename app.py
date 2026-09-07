from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse
import base64
import csv
import os
import time
from collections import deque
from datetime import datetime, timezone

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
PAD_MIN_MOTION_SCORE = float(os.getenv("PAD_MIN_MOTION_SCORE", "0.18"))
PAD_STRONG_ATTACK_SCORE = float(os.getenv("PAD_STRONG_ATTACK_SCORE", "0.62"))
PAD_FLAT_SUPPORT_SCORE = float(os.getenv("PAD_FLAT_SUPPORT_SCORE", "0.48"))
PAD_MIN_REAL_FACE_SCALE = float(os.getenv("PAD_MIN_REAL_FACE_SCALE", "0.23"))
PAD_LARGE_REAL_FACE_SCALE = float(os.getenv("PAD_LARGE_REAL_FACE_SCALE", "0.30"))
PAD_MODEL_PATH = os.getenv("PAD_MODEL_PATH", "/app/models/minifasnet_v2.onnx")
PAD_MODEL_PATHS = [
    caminho.strip()
    for caminho in os.getenv("PAD_MODEL_PATHS", PAD_MODEL_PATH).split(",")
    if caminho.strip()
]
PAD_MODEL_ENABLED = os.getenv("PAD_MODEL_ENABLED", "true").lower() == "true"
PAD_MODEL_REAL_THRESHOLD = float(os.getenv("PAD_MODEL_REAL_THRESHOLD", "0.72"))
PAD_MODEL_ATTACK_THRESHOLD = float(os.getenv("PAD_MODEL_ATTACK_THRESHOLD", "0.55"))
PAD_MODEL_CROP_SCALES = [
    float(valor.strip())
    for valor in os.getenv("PAD_MODEL_CROP_SCALES", "2.0,2.7,3.4").split(",")
    if valor.strip()
]
CALIBRATION_LOG_ENABLED = os.getenv("CALIBRATION_LOG_ENABLED", "true").lower() == "true"
CALIBRATION_LOG_PATH = os.getenv(
    "CALIBRATION_LOG_PATH", "/app/logs/liveness_calibration.csv"
)
ACTIVE_CHALLENGE_ENABLED = (
    os.getenv("ACTIVE_CHALLENGE_ENABLED", "true").lower() == "true"
)
ACTIVE_CHALLENGE_SECONDS = int(os.getenv("ACTIVE_CHALLENGE_SECONDS", "12"))
ACTIVE_CHALLENGE_MIN_SCORE = float(os.getenv("ACTIVE_CHALLENGE_MIN_SCORE", "0.55"))

CHALLENGES = [
    ("PISQUE", "Pisque"),
    ("VIRE_ESQUERDA", "Vire a cabeca para a esquerda"),
    ("VIRE_DIREITA", "Vire a cabeca para a direita"),
    ("APROXIME", "Aproxime o rosto"),
]

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
    largura_face = max(1, x2 - x1)
    altura_face = max(1, y2 - y1)
    area_face = largura_face * altura_face
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

        overlap_x1 = max(x1, rx)
        overlap_y1 = max(y1, ry)
        overlap_x2 = min(x2, rx + rw)
        overlap_y2 = min(y2, ry + rh)
        overlap = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
        proporcao_face_no_suporte = overlap / max(1, area_face)
        suporte_cobre_face = (
            rx <= x1 + largura_face * 0.15
            and ry <= y1 + altura_face * 0.20
            and rx + rw >= x2 - largura_face * 0.15
            and ry + rh >= y2 - altura_face * 0.15
        )
        if proporcao_face_no_suporte < 0.62 or not suporte_cobre_face:
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
        mascara_entorno = (
            (entorno_hsv[:, :, 1] < 70) & (entorno_hsv[:, :, 2] > 155)
        ).astype(np.uint8)
        rx1, ry1 = x1_c - ex1, y1_c - ey1
        rx2, ry2 = x2_c - ex1, y2_c - ey1

        topo = mascara_entorno[: max(0, ry1), :]
        base = mascara_entorno[min(mascara_entorno.shape[0], ry2) :, :]
        esquerda = mascara_entorno[:, : max(0, rx1)]
        direita = mascara_entorno[:, min(mascara_entorno.shape[1], rx2) :]

        proporcoes_lados = [
            float(np.mean(regiao)) if regiao.size else 0.0
            for regiao in (topo, base, esquerda, direita)
        ]
        lados_com_papel = sum(1 for valor in proporcoes_lados if valor >= 0.38)
        papel_cercando_face = min(proporcoes_lados) if lados_com_papel >= 3 else 0.0
        score_entorno = normalizar_intervalo(papel_cercando_face, 0.38, 0.78)
        melhor_score = max(melhor_score, score_entorno)

    return melhor_score


def softmax(valores):
    valores = np.asarray(valores, dtype=np.float32)
    valores = valores - np.max(valores)
    exp = np.exp(valores)
    soma = np.sum(exp)
    return exp / soma if soma else exp


def recortar_face_expandida(frame, bbox, escala=2.7):
    altura, largura = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    largura_face = max(1, x2 - x1)
    altura_face = max(1, y2 - y1)
    centro_x = (x1 + x2) / 2.0
    centro_y = (y1 + y2) / 2.0
    lado = max(largura_face, altura_face) * escala

    nx1 = int(max(0, centro_x - lado / 2.0))
    ny1 = int(max(0, centro_y - lado / 2.0))
    nx2 = int(min(largura, centro_x + lado / 2.0))
    ny2 = int(min(altura, centro_y + lado / 2.0))
    crop = frame[ny1:ny2, nx1:nx2]
    return crop if crop.size else None


def preparar_entrada_pad(crop):
    entrada = cv2.resize(crop, (80, 80)).astype(np.float32) / 255.0
    entrada = np.transpose(entrada, (2, 0, 1))[None, :, :, :]
    return entrada


def inferir_sessao_pad(sessao, nome_entrada, entrada):
    saida = sessao.run(None, {nome_entrada: entrada})[0]
    probabilidades = np.asarray(saida).reshape(-1)
    if probabilidades.size < 3:
        return None

    if not np.isclose(float(np.sum(probabilidades[:3])), 1.0, atol=0.08):
        probabilidades = softmax(probabilidades[:3])
    else:
        probabilidades = probabilidades[:3]
    return probabilidades


def inferir_modelo_pad(frame, bbox):
    if not pad_model_loaded or not pad_sessions:
        return {
            "pad_model_live": 0.0,
            "pad_model_print": 0.0,
            "pad_model_replay": 0.0,
            "pad_model_attack": 0.0,
            "pad_model_ensemble": 0,
            "pad_model_usado": False,
        }

    resultados = []
    for escala in PAD_MODEL_CROP_SCALES:
        crop = recortar_face_expandida(frame, bbox, escala=escala)
        if crop is None:
            continue
        entrada = preparar_entrada_pad(crop)
        for modelo in pad_sessions:
            probabilidades = inferir_sessao_pad(
                modelo["session"], modelo["input_name"], entrada
            )
            if probabilidades is not None:
                resultados.append(probabilidades)

    if not resultados:
        return {
            "pad_model_live": 0.0,
            "pad_model_print": 0.0,
            "pad_model_replay": 0.0,
            "pad_model_attack": 0.0,
            "pad_model_ensemble": 0,
            "pad_model_usado": False,
        }

    probabilidades = np.mean(np.stack(resultados), axis=0)
    live = float(probabilidades[0])
    print_attack = float(probabilidades[1])
    replay_attack = float(probabilidades[2])
    return {
        "pad_model_live": limitar(live),
        "pad_model_print": limitar(print_attack),
        "pad_model_replay": limitar(replay_attack),
        "pad_model_attack": limitar(print_attack + replay_attack),
        "pad_model_ensemble": len(resultados),
        "pad_model_usado": True,
    }


def calcular_evidencias(face, frame, bbox, landmarks, variacao_profundidade):
    x1, y1, x2, y2 = bbox
    y1_c, y2_c = max(0, y1), min(frame.shape[0], y2)
    x1_c, x2_c = max(0, x1), min(frame.shape[1], x2)
    rosto_recortado = frame[y1_c:y2_c, x1_c:x2_c]
    largura_face = max(1, x2 - x1)
    altura_face = max(1, y2 - y1)
    escala_face = (
        (largura_face * altura_face) / max(1, frame.shape[0] * frame.shape[1])
    ) ** 0.5
    nariz_x = float((landmarks[30][0] - x1) / largura_face)
    boca_abertura = float(
        np.linalg.norm(landmarks[62][:2] - landmarks[66][:2]) / altura_face
    )

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
    pad_model = inferir_modelo_pad(frame, bbox)

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
        "escala_face": float(escala_face),
        **pad_model,
        "variacao_profundidade": float(variacao_profundidade),
        "media_saturacao": media_saturacao,
        "media_brilho": media_brilho,
        "desvio_cor": desvio_cor,
        "nitidez": nitidez,
        "eye_aspect_ratio": calcular_eye_aspect_ratio(landmarks),
        "nose_x_ratio": nariz_x,
        "mouth_open_ratio": boca_abertura,
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
            "face_scales": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "pad_model_live_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "pad_model_print_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "pad_model_replay_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "pad_model_attack_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "centers": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "areas": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "depth_values": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "eye_aspects": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "nose_x_ratios": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "mouth_open_ratios": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "parallax_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "challenge_scores": deque(maxlen=TEMPORAL_WINDOW_SIZE),
            "last_seen": frame_atual,
        }

    tracks_usados.add(melhor_track_id)
    return melhor_track_id


def calcular_movimento_temporal(track):
    depths = list(track["depth_values"])
    eyes = list(track["eye_aspects"])
    noses = list(track.get("nose_x_ratios", []))
    mouths = list(track.get("mouth_open_ratios", []))
    if len(depths) < TEMPORAL_MIN_FRAMES:
        return 0.0

    movimento_profundidade = limitar((max(depths) - min(depths)) / 10.0) if depths else 0.0
    movimento_olhos = limitar((max(eyes) - min(eyes)) / 0.08) if eyes else 0.0
    movimento_nariz = limitar((max(noses) - min(noses)) / 0.10) if noses else 0.0
    movimento_boca = limitar((max(mouths) - min(mouths)) / 0.05) if mouths else 0.0
    return limitar(
        (movimento_profundidade * 0.35)
        + (movimento_olhos * 0.30)
        + (movimento_nariz * 0.25)
        + (movimento_boca * 0.10)
    )


def calcular_paralaxe_temporal(track):
    depths = list(track["depth_values"])
    noses = list(track.get("nose_x_ratios", []))
    areas = list(track.get("areas", []))
    if len(depths) < TEMPORAL_MIN_FRAMES or len(noses) < TEMPORAL_MIN_FRAMES:
        return 0.0

    variacao_profundidade = limitar((max(depths) - min(depths)) / 9.0)
    variacao_nariz = limitar((max(noses) - min(noses)) / 0.10)
    area_media = max(sum(areas) / len(areas), 1.0)
    variacao_area = limitar((max(areas) - min(areas)) / (area_media * 0.16)) if areas else 0.0
    coerencia_3d = 1.0 - min(
        abs(variacao_area - variacao_profundidade),
        abs(variacao_area - variacao_nariz),
    )
    return limitar(
        (variacao_profundidade * 0.40)
        + (variacao_nariz * 0.35)
        + (coerencia_3d * 0.25)
    )


def obter_desafio_atual():
    if not ACTIVE_CHALLENGE_ENABLED:
        return {"codigo": "LIVRE", "texto": "Modo livre", "segundos_restantes": 0}

    periodo = max(ACTIVE_CHALLENGE_SECONDS, 1)
    indice = int(time.time() // periodo) % len(CHALLENGES)
    codigo, texto = CHALLENGES[indice]
    segundos_restantes = periodo - int(time.time() % periodo)
    return {
        "codigo": codigo,
        "texto": texto,
        "segundos_restantes": segundos_restantes,
    }


def avaliar_desafio(track, desafio):
    codigo = desafio["codigo"]
    eyes = list(track.get("eye_aspects", []))
    noses = list(track.get("nose_x_ratios", []))
    areas = list(track.get("areas", []))

    if codigo == "PISQUE" and len(eyes) >= TEMPORAL_MIN_FRAMES:
        score = limitar((max(eyes) - min(eyes)) / 0.09)
    elif codigo == "VIRE_ESQUERDA" and len(noses) >= TEMPORAL_MIN_FRAMES:
        score = limitar((max(noses) - min(noses)) / 0.11)
    elif codigo == "VIRE_DIREITA" and len(noses) >= TEMPORAL_MIN_FRAMES:
        score = limitar((max(noses) - min(noses)) / 0.11)
    elif codigo == "APROXIME" and len(areas) >= TEMPORAL_MIN_FRAMES:
        area_media = max(sum(areas) / len(areas), 1.0)
        score = limitar((max(areas) - min(areas)) / (area_media * 0.22))
    else:
        score = 0.0

    return {
        "codigo": codigo,
        "texto": desafio["texto"],
        "score": score,
        "ok": score >= ACTIVE_CHALLENGE_MIN_SCORE,
    }


def registrar_calibracao(frame_atual, faces_resultado, metricas, desafio):
    if not CALIBRATION_LOG_ENABLED or not faces_resultado:
        return

    campos = [
        "timestamp",
        "frame",
        "track_id",
        "label",
        "tipo_apresentacao",
        "confianca",
        "frames_analisados",
        "estavel",
        "desafio_codigo",
        "desafio_ok",
        "face_detectada",
        "profundidade",
        "textura_natural",
        "cor_natural",
        "movimento_natural",
        "paralaxe_3d",
        "desafio_ativo_score",
        "anti_spoofing",
        "foto_score",
        "tela_score",
        "suporte_plano",
        "escala_face",
        "pad_model_live",
        "pad_model_print",
        "pad_model_replay",
        "pad_model_attack",
        "decode_ms",
        "deteccao_ms",
        "liveness_ms",
        "encode_ms",
        "total_ms",
    ]

    os.makedirs(os.path.dirname(CALIBRATION_LOG_PATH), exist_ok=True)
    arquivo_existe = os.path.exists(CALIBRATION_LOG_PATH)
    with open(CALIBRATION_LOG_PATH, "a", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        if not arquivo_existe:
            writer.writeheader()

        timestamp = datetime.now(timezone.utc).isoformat()
        for face in faces_resultado:
            evidencias = face.get("evidencias", {})
            linha = {
                "timestamp": timestamp,
                "frame": frame_atual,
                "track_id": face.get("id"),
                "label": face.get("label"),
                "tipo_apresentacao": face.get("tipo_apresentacao"),
                "confianca": face.get("confianca"),
                "frames_analisados": face.get("frames_analisados"),
                "estavel": face.get("estavel"),
                "desafio_codigo": desafio.get("codigo"),
                "desafio_ok": face.get("desafio_ok"),
                "decode_ms": metricas.get("decode_ms"),
                "deteccao_ms": metricas.get("deteccao_ms"),
                "liveness_ms": metricas.get("liveness_ms"),
                "encode_ms": metricas.get("encode_ms"),
                "total_ms": metricas.get("total_ms"),
            }
            for chave in campos:
                if chave in evidencias:
                    linha[chave] = evidencias[chave]
            writer.writerow(linha)


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
    track.setdefault("face_scales", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track["face_scales"].append(float(evidencias["escala_face"]))
    track.setdefault("pad_model_live_scores", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track.setdefault("pad_model_print_scores", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track.setdefault("pad_model_replay_scores", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track.setdefault("pad_model_attack_scores", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track["pad_model_live_scores"].append(float(evidencias["pad_model_live"]))
    track["pad_model_print_scores"].append(float(evidencias["pad_model_print"]))
    track["pad_model_replay_scores"].append(float(evidencias["pad_model_replay"]))
    track["pad_model_attack_scores"].append(float(evidencias["pad_model_attack"]))
    track["centers"].append(center)
    track["areas"].append(float(area))
    track["depth_values"].append(float(evidencias["variacao_profundidade"]))
    track["eye_aspects"].append(float(evidencias["eye_aspect_ratio"]))
    track.setdefault("nose_x_ratios", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track.setdefault("mouth_open_ratios", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track.setdefault("parallax_scores", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track.setdefault("challenge_scores", deque(maxlen=TEMPORAL_WINDOW_SIZE))
    track["nose_x_ratios"].append(float(evidencias["nose_x_ratio"]))
    track["mouth_open_ratios"].append(float(evidencias["mouth_open_ratio"]))

    movimento_natural = calcular_movimento_temporal(track)
    paralaxe_3d = calcular_paralaxe_temporal(track)
    desafio = avaliar_desafio(track, obter_desafio_atual())
    track["parallax_scores"].append(float(paralaxe_3d))
    track["challenge_scores"].append(float(desafio["score"]))
    contexto_apresentacao = (
        evidencias["suporte_plano"] >= 0.25
        or evidencias["foto_score"] >= 0.45
        or evidencias["tela_score"] >= 0.45
        or evidencias["escala_face"] < PAD_MIN_REAL_FACE_SCALE
    )
    ataque_modelo_ponderado = (
        evidencias["pad_model_attack"] if contexto_apresentacao else 0.0
    )
    ataque_visual = max(
        evidencias["foto_score"],
        evidencias["tela_score"],
        evidencias["suporte_plano"],
        ataque_modelo_ponderado,
    )
    if evidencias["pad_model_usado"]:
        anti_spoof_instantaneo = limitar(
            (evidencias["pad_model_live"] * 0.20)
            + (evidencias["profundidade"] * 0.22)
            + (evidencias["textura_natural"] * 0.18)
            + (evidencias["cor_natural"] * 0.12)
            + (movimento_natural * 0.12)
            + (paralaxe_3d * 0.08)
            + ((1.0 - ataque_visual) * 0.08)
        )
    else:
        anti_spoof_instantaneo = limitar(
            (evidencias["profundidade"] * 0.25)
            + (evidencias["textura_natural"] * 0.20)
            + (evidencias["cor_natural"] * 0.15)
            + (movimento_natural * 0.17)
            + (paralaxe_3d * 0.08)
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
    escala_face_media = media_deque(track["face_scales"])
    pad_live_media = media_deque(track["pad_model_live_scores"])
    pad_print_media = media_deque(track["pad_model_print_scores"])
    pad_replay_media = media_deque(track["pad_model_replay_scores"])
    pad_attack_media = media_deque(track["pad_model_attack_scores"])
    paralaxe_media = media_deque(track["parallax_scores"])
    desafio_media = media_deque(track["challenge_scores"])
    pad_model_usado = bool(evidencias["pad_model_usado"])
    contexto_apresentacao_media = (
        suporte_plano_media >= 0.25
        or foto_media >= 0.45
        or tela_media >= 0.45
        or escala_face_media < PAD_MIN_REAL_FACE_SCALE
    )
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
        and paralaxe_media >= 0.18
        and max(foto_media, tela_media, suporte_plano_media) < PAD_FLAT_SUPPORT_SCORE
        and escala_face_media >= PAD_MIN_REAL_FACE_SCALE
        and (
            not pad_model_usado
            or (
                pad_live_media >= 0.30
                or not contexto_apresentacao_media
            )
        )
    )
    face_real_grande_com_oclusao = (
        escala_face_media >= PAD_LARGE_REAL_FACE_SCALE
        and profundidade_media >= 0.35
        and textura_media >= 0.30
        and cor_media >= 0.25
        and suporte_plano_media < PAD_FLAT_SUPPORT_SCORE
        and foto_media < 0.80
        and tela_media < 0.80
        and score_medio >= 0.35
        and (paralaxe_media >= 0.12 or desafio_media >= ACTIVE_CHALLENGE_MIN_SCORE)
    )
    evidencia_temporal_forte_real = (
        escala_face_media >= PAD_LARGE_REAL_FACE_SCALE
        and profundidade_media >= 0.70
        and cor_media >= 0.65
        and movimento_natural >= 0.50
        and paralaxe_media >= 0.55
        and desafio_media >= ACTIVE_CHALLENGE_MIN_SCORE
        and suporte_plano_media < 0.20
        and foto_media < 0.35
        and tela_media < 0.35
    )

    label = "SPOOF"
    tipo_apresentacao = "INCERTO"
    confianca = calcular_confianca_incerto(score_medio, frames_analisados)
    cor_borda = (0, 0, 255)
    cor_interface = "#ef4444"

    if not estavel:
        tipo_apresentacao = "INCERTO"
    elif score_medio >= PAD_REAL_THRESHOLD and evidencias_suficientes_real:
        label = "REAL"
        tipo_apresentacao = "PRESENCA_FISICA"
        confianca = score_medio
        cor_borda = (0, 255, 0)
        cor_interface = "#22c55e"
    elif evidencia_temporal_forte_real:
        label = "REAL"
        tipo_apresentacao = "PRESENCA_FISICA_CONFIRMADA"
        confianca = limitar(
            (
                profundidade_media * 0.25
                + cor_media * 0.15
                + movimento_natural * 0.20
                + paralaxe_media * 0.25
                + desafio_media * 0.15
            )
        )
        cor_borda = (0, 255, 0)
        cor_interface = "#22c55e"
    elif face_real_grande_com_oclusao:
        label = "REAL"
        tipo_apresentacao = "PRESENCA_FISICA_PARCIAL"
        confianca = limitar(max(score_medio, 0.68))
        cor_borda = (0, 255, 0)
        cor_interface = "#22c55e"
    elif pad_print_media >= PAD_MODEL_ATTACK_THRESHOLD and contexto_apresentacao_media:
        tipo_apresentacao = "FOTO"
        confianca = pad_print_media
    elif pad_replay_media >= PAD_MODEL_ATTACK_THRESHOLD and contexto_apresentacao_media:
        tipo_apresentacao = "TELA"
        confianca = pad_replay_media
    elif suporte_plano_media >= PAD_FLAT_SUPPORT_SCORE or foto_media >= PAD_STRONG_ATTACK_SCORE:
        tipo_apresentacao = "FOTO"
        confianca = max(foto_media, suporte_plano_media)
    elif tela_media >= PAD_STRONG_ATTACK_SCORE:
        tipo_apresentacao = "TELA"
        confianca = tela_media
    elif score_medio <= PAD_SPOOF_THRESHOLD:
        tipo_apresentacao = "SPOOF"
        confianca = 1.0 - score_medio
    else:
        tipo_apresentacao = "INCERTO"
        confianca = calcular_confianca_incerto(score_medio, frames_analisados)

    return {
        "label": label,
        "tipo_apresentacao": tipo_apresentacao,
        "confianca": confianca,
        "real_score_medio": score_medio,
        "evidencias": {
            "face_detectada": face_media,
            "profundidade": profundidade_media,
            "textura_natural": textura_media,
            "cor_natural": cor_media,
            "movimento_natural": movimento_natural,
            "paralaxe_3d": paralaxe_media,
            "desafio_ativo_score": desafio_media,
            "desafio_ativo_ok": 1.0 if desafio["ok"] else 0.0,
            "anti_spoofing": score_medio,
            "foto_score": foto_media,
            "tela_score": tela_media,
            "suporte_plano": suporte_plano_media,
            "escala_face": escala_face_media,
            "pad_model_live": pad_live_media,
            "pad_model_print": pad_print_media,
            "pad_model_replay": pad_replay_media,
            "pad_model_attack": pad_attack_media,
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

pad_session = None
pad_input_name = None
pad_sessions = []
pad_model_loaded = False
pad_model_provider = None

if PAD_MODEL_ENABLED:
    try:
        pad_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider_ativo == "CUDA"
            else ["CPUExecutionProvider"]
        )

        for caminho_modelo in PAD_MODEL_PATHS:
            if not os.path.exists(caminho_modelo):
                print(f"[-] Modelo PAD nao encontrado: {caminho_modelo}")
                continue

            session = ort.InferenceSession(caminho_modelo, providers=pad_providers)
            input_name = session.get_inputs()[0].name
            pad_sessions.append(
                {
                    "path": caminho_modelo,
                    "session": session,
                    "input_name": input_name,
                    "provider": session.get_providers()[0],
                }
            )

        if not pad_sessions:
            raise FileNotFoundError(
                f"Nenhum modelo PAD carregado em: {', '.join(PAD_MODEL_PATHS)}"
            )

        pad_session = pad_sessions[0]["session"]
        pad_input_name = pad_sessions[0]["input_name"]
        pad_model_provider = pad_sessions[0]["provider"]
        pad_model_loaded = True
        print(f"[*] Modelos PAD carregados: {[m['path'] for m in pad_sessions]}")
        print(f"[*] Provider PAD ativo: {pad_model_provider}")
    except Exception as e:
        print(f"[-] Modelo PAD indisponivel, usando heuristicas: {e}")


@app.get("/health")
async def health():
    return {
        "status": "ok" if model_loaded else "erro",
        "onnx_providers": ort_providers,
        "gpu_enabled": provider_ativo == "CUDA",
        "provider_ativo": provider_ativo,
        "model_loaded": model_loaded,
        "pad_model_enabled": PAD_MODEL_ENABLED,
        "pad_model_loaded": pad_model_loaded,
        "pad_model_path": PAD_MODEL_PATH,
        "pad_model_paths": PAD_MODEL_PATHS,
        "pad_model_provider": pad_model_provider,
        "pad_model_count": len(pad_sessions),
        "pad_model_crop_scales": PAD_MODEL_CROP_SCALES,
        "calibration_log_enabled": CALIBRATION_LOG_ENABLED,
        "calibration_log_path": CALIBRATION_LOG_PATH,
        "active_challenge_enabled": ACTIVE_CHALLENGE_ENABLED,
    }


@app.get("/calibration")
async def calibration(limit: int = Query(20, ge=1, le=200)):
    if not os.path.exists(CALIBRATION_LOG_PATH):
        return {
            "status": "sem_logs",
            "path": CALIBRATION_LOG_PATH,
            "linhas": [],
        }

    with open(CALIBRATION_LOG_PATH, newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    return {
        "status": "ok",
        "path": CALIBRATION_LOG_PATH,
        "total": len(linhas),
        "linhas": linhas[-limit:],
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
        desafio_atual = obter_desafio_atual()

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
                    "tipo_apresentacao": decisao_temporal["tipo_apresentacao"],
                    "desafio_ativo": desafio_atual,
                    "desafio_ok": bool(
                        decisao_temporal["evidencias"]["desafio_ativo_ok"]
                    ),
                    "confianca": round(float(confianca), 3),
                    "score": round(float(confianca), 3),
                    "face_score": round(float(evidencias["face_detectada"]), 3),
                    "anti_spoofing_score": round(
                        float(decisao_temporal["evidencias"]["anti_spoofing"]), 3
                    ),
                    "pad_model_usado": bool(evidencias["pad_model_usado"]),
                    "pad_model_ensemble": int(evidencias["pad_model_ensemble"]),
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

        metricas = {
            "decode_ms": round(tempo_decode_ms, 2),
            "deteccao_ms": round(tempo_deteccao_ms, 2),
            "liveness_ms": round(tempo_liveness_ms, 2),
            "encode_ms": round(tempo_encode_ms, 2),
            "total_ms": round(tempo_total_ms, 2),
        }
        registrar_calibracao(frame_atual, faces_resultado, metricas, desafio_atual)

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
            "desafio_ativo": desafio_atual,
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
                "pad_min_real_face_scale": PAD_MIN_REAL_FACE_SCALE,
                "pad_large_real_face_scale": PAD_LARGE_REAL_FACE_SCALE,
                "pad_model_real_threshold": PAD_MODEL_REAL_THRESHOLD,
                "pad_model_attack_threshold": PAD_MODEL_ATTACK_THRESHOLD,
                "tracks_ativos": len(face_tracks),
            },
            "metricas": metricas,
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
                <div class="status-item"><span class="status-label">Tipo provavel</span><span class="status-value" id="tipo-apresentacao">--</span></div>
                <div class="status-item"><span class="status-label">Desafio</span><span class="status-value" id="desafio-ativo">--</span></div>
                <div class="status-item"><span class="status-label">Desafio status</span><span class="status-value" id="desafio-status">--</span></div>
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
                <div class="status-item"><span class="status-label">Paralaxe 3D</span><span class="status-value" id="score-paralaxe">0%</span></div>
                <div class="status-item"><span class="status-label">Anti-spoofing</span><span class="status-value" id="score-antispoofing">0%</span></div>
                <div class="status-item"><span class="status-label">Suporte plano</span><span class="status-value" id="score-suporte-plano">0%</span></div>
                <div class="status-item"><span class="status-label">Escala da face</span><span class="status-value" id="score-escala-face">0%</span></div>
                <div class="status-item"><span class="status-label">Modelo live</span><span class="status-value" id="score-modelo-live">0%</span></div>
                <div class="status-item"><span class="status-label">Modelo print</span><span class="status-value" id="score-modelo-print">0%</span></div>
                <div class="status-item"><span class="status-label">Modelo replay</span><span class="status-value" id="score-modelo-replay">0%</span></div>
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
            const tipoApresentacaoEl = document.getElementById('tipo-apresentacao');
            const desafioAtivoEl = document.getElementById('desafio-ativo');
            const desafioStatusEl = document.getElementById('desafio-status');
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
            const scoreParalaxeEl = document.getElementById('score-paralaxe');
            const scoreAntispoofingEl = document.getElementById('score-antispoofing');
            const scoreSuportePlanoEl = document.getElementById('score-suporte-plano');
            const scoreEscalaFaceEl = document.getElementById('score-escala-face');
            const scoreModeloLiveEl = document.getElementById('score-modelo-live');
            const scoreModeloPrintEl = document.getElementById('score-modelo-print');
            const scoreModeloReplayEl = document.getElementById('score-modelo-replay');

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
                                    if (data.desafio_ativo) {
                                        desafioAtivoEl.textContent = `${data.desafio_ativo.texto} (${data.desafio_ativo.segundos_restantes}s)`;
                                    }
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
                                        tipoApresentacaoEl.textContent = primeiraFace.tipo_apresentacao || "--";
                                        desafioStatusEl.textContent = primeiraFace.desafio_ok ? "ok" : "pendente";
                                        atualizarClasseStatus(desafioStatusEl, primeiraFace.desafio_ok ? "status-ok" : "status-warn");
                                        framesTemporaisEl.textContent = `${primeiraFace.frames_analisados}/${data.analise_temporal.min_frames}`;
                                        estabilidadeTemporalEl.textContent = primeiraFace.estavel ? "estavel" : "analisando";
                                        atualizarClasseStatus(estabilidadeTemporalEl, primeiraFace.estavel ? "status-ok" : "status-warn");
                                        if (primeiraFace.evidencias) {
                                            scoreFaceEl.textContent = `${(primeiraFace.evidencias.face_detectada * 100).toFixed(0)}%`;
                                            scoreProfundidadeEl.textContent = `${(primeiraFace.evidencias.profundidade * 100).toFixed(0)}%`;
                                            scoreTexturaEl.textContent = `${(primeiraFace.evidencias.textura_natural * 100).toFixed(0)}%`;
                                            scoreMovimentoEl.textContent = `${(primeiraFace.evidencias.movimento_natural * 100).toFixed(0)}%`;
                                            scoreParalaxeEl.textContent = `${(primeiraFace.evidencias.paralaxe_3d * 100).toFixed(0)}%`;
                                            scoreAntispoofingEl.textContent = `${(primeiraFace.evidencias.anti_spoofing * 100).toFixed(0)}%`;
                                            scoreSuportePlanoEl.textContent = `${(primeiraFace.evidencias.suporte_plano * 100).toFixed(0)}%`;
                                            scoreEscalaFaceEl.textContent = `${(primeiraFace.evidencias.escala_face * 100).toFixed(0)}%`;
                                            scoreModeloLiveEl.textContent = `${(primeiraFace.evidencias.pad_model_live * 100).toFixed(0)}%`;
                                            scoreModeloPrintEl.textContent = `${(primeiraFace.evidencias.pad_model_print * 100).toFixed(0)}%`;
                                            scoreModeloReplayEl.textContent = `${(primeiraFace.evidencias.pad_model_replay * 100).toFixed(0)}%`;
                                        }
                                    } else if (data.analise_temporal) {
                                        tipoApresentacaoEl.textContent = "--";
                                        desafioStatusEl.textContent = "--";
                                        atualizarClasseStatus(desafioStatusEl, "");
                                        framesTemporaisEl.textContent = `0/${data.analise_temporal.min_frames}`;
                                        estabilidadeTemporalEl.textContent = "--";
                                        atualizarClasseStatus(estabilidadeTemporalEl, "");
                                        scoreFaceEl.textContent = "0%";
                                        scoreProfundidadeEl.textContent = "0%";
                                        scoreTexturaEl.textContent = "0%";
                                        scoreMovimentoEl.textContent = "0%";
                                        scoreParalaxeEl.textContent = "0%";
                                        scoreAntispoofingEl.textContent = "0%";
                                        scoreSuportePlanoEl.textContent = "0%";
                                        scoreEscalaFaceEl.textContent = "0%";
                                        scoreModeloLiveEl.textContent = "0%";
                                        scoreModeloPrintEl.textContent = "0%";
                                        scoreModeloReplayEl.textContent = "0%";
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
