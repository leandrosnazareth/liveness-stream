from collections import deque

from liveness_app.config import (
    ACTIVE_CHALLENGE_MIN_SCORE,
    PAD_FLAT_SUPPORT_SCORE,
    PAD_LARGE_REAL_FACE_SCALE,
    PAD_MIN_MOTION_SCORE,
    PAD_MIN_REAL_FACE_SCALE,
    PAD_MODEL_ATTACK_THRESHOLD,
    PAD_PARTIAL_REAL_FACE_SCALE,
    PAD_REAL_THRESHOLD,
    PAD_SPOOF_THRESHOLD,
    PAD_STRONG_ATTACK_SCORE,
    TEMPORAL_MATCH_IOU,
    TEMPORAL_MIN_FRAMES,
    TEMPORAL_STABILITY_DELTA,
    TEMPORAL_STALE_FRAMES,
    TEMPORAL_WINDOW_SIZE,
)
from liveness_app.core.challenge import avaliar_desafio, obter_desafio_atual
from liveness_app.core.utils import limitar, media_deque


face_tracks = {}
next_track_id = 1

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
        and textura_media >= 0.24
        and cor_media >= 0.25
        and suporte_plano_media < PAD_FLAT_SUPPORT_SCORE
        and foto_media < 0.80
        and tela_media < 0.80
        and score_medio >= 0.30
        and (paralaxe_media >= 0.12 or desafio_media >= ACTIVE_CHALLENGE_MIN_SCORE)
    )
    face_real_parcial_com_evidencia_temporal = (
        escala_face_media >= PAD_PARTIAL_REAL_FACE_SCALE
        and profundidade_media >= 0.70
        and textura_media >= 0.42
        and cor_media >= 0.62
        and movimento_natural >= 0.42
        and (paralaxe_media >= 0.20 or desafio_media >= 0.18)
        and suporte_plano_media < 0.22
        and foto_media < 0.40
        and tela_media < 0.40
        and score_medio >= 0.40
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
        confianca = limitar(max(score_medio, 0.72))
        cor_borda = (0, 255, 0)
        cor_interface = "#22c55e"
    elif face_real_parcial_com_evidencia_temporal:
        label = "REAL"
        tipo_apresentacao = "PRESENCA_FISICA_TEMPORAL"
        confianca = limitar(
            max(
                score_medio,
                (
                    profundidade_media * 0.22
                    + textura_media * 0.14
                    + cor_media * 0.14
                    + movimento_natural * 0.22
                    + max(paralaxe_media, desafio_media) * 0.18
                    + (1.0 - max(foto_media, tela_media, suporte_plano_media)) * 0.10
                ),
            )
        )
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

