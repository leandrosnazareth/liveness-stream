import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse

from liveness_app.config import (
    ACTIVE_CHALLENGE_ENABLED,
    CALIBRATION_LOG_ENABLED,
    CALIBRATION_LOG_PATH,
    PAD_FLAT_SUPPORT_SCORE,
    PAD_LARGE_REAL_FACE_SCALE,
    PAD_MIN_MOTION_SCORE,
    PAD_MIN_REAL_FACE_SCALE,
    PAD_MODEL_ATTACK_THRESHOLD,
    PAD_MODEL_CROP_SCALES,
    PAD_MODEL_ENABLED,
    PAD_MODEL_PATH,
    PAD_MODEL_PATHS,
    PAD_MODEL_REAL_THRESHOLD,
    PAD_REAL_THRESHOLD,
    PAD_SPOOF_THRESHOLD,
    PAD_STRONG_ATTACK_SCORE,
    TEMPORAL_MIN_FRAMES,
    TEMPORAL_REAL_THRESHOLD,
    TEMPORAL_SPOOF_THRESHOLD,
    TEMPORAL_STABILITY_DELTA,
    TEMPORAL_WINDOW_SIZE,
)
from liveness_app.core.calibration import ler_calibracao, registrar_calibracao
from liveness_app.core.challenge import obter_desafio_atual
from liveness_app.core.evidence import calcular_evidencias
from liveness_app.core.temporal import (
    atualizar_decisao_temporal,
    face_tracks,
    limpar_tracks_antigos,
    obter_track_id,
)
from liveness_app.models import state


app = FastAPI()
frame_sequence = 0
INDEX_HTML_PATH = Path(__file__).parent / "web" / "index.html"


@app.get("/health")
async def health():
    return {
        "status": "ok" if state.model_loaded else "erro",
        "onnx_providers": state.ort_providers,
        "gpu_enabled": state.provider_ativo == "CUDA",
        "provider_ativo": state.provider_ativo,
        "model_loaded": state.model_loaded,
        "pad_model_enabled": PAD_MODEL_ENABLED,
        "pad_model_loaded": state.pad_model_loaded,
        "pad_model_path": PAD_MODEL_PATH,
        "pad_model_paths": PAD_MODEL_PATHS,
        "pad_model_provider": state.pad_model_provider,
        "pad_model_count": len(state.pad_sessions),
        "pad_model_crop_scales": PAD_MODEL_CROP_SCALES,
        "calibration_log_enabled": CALIBRATION_LOG_ENABLED,
        "calibration_log_path": CALIBRATION_LOG_PATH,
        "active_challenge_enabled": ACTIVE_CHALLENGE_ENABLED,
    }


@app.get("/calibration")
async def calibration(limit: int = Query(20, ge=1, le=200)):
    return ler_calibracao(limit)


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
        faces = state.app_face.get(frame)
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
            "provider_ativo": state.provider_ativo,
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
    return INDEX_HTML_PATH.read_text(encoding="utf-8")
