import csv
import os
from datetime import datetime, timezone

from liveness_app.config import CALIBRATION_LOG_ENABLED, CALIBRATION_LOG_PATH

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



def ler_calibracao(limit):
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
