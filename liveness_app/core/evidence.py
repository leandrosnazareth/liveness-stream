import cv2
import numpy as np

from liveness_app.core.utils import limitar, normalizar_intervalo
from liveness_app.models import inferir_modelo_pad

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


