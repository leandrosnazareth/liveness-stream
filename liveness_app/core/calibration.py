"""Optional diagnostics, without frames, embeddings, subject IDs or capture tokens."""
from collections import deque
import csv
from datetime import datetime, timezone
from pathlib import Path
import threading

from liveness_app.config import CALIBRATION_LOG_ENABLED, CALIBRATION_LOG_PATH

LOCK = threading.Lock()
FIELDS = ["timestamp", "session_id", "frame", "estado", "motivo", "score",
          "quantidade_rostos", "frames_analisados", "desafio_codigo", "desafio_ok",
          "pad_model_live", "pad_model_print", "pad_model_replay", "pad_model_attack",
          "decode_ms", "total_ms"]


def registrar_calibracao(frame_atual, faces_resultado, metricas, desafio, *, session_id, result, face_count):
    if not CALIBRATION_LOG_ENABLED:
        return
    evidence = faces_resultado[0].get("evidencias", {}) if faces_resultado else {}
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "session_id": session_id,
           "frame": frame_atual, "estado": result["estado"], "motivo": result["motivo"],
           "score": result.get("score"), "quantidade_rostos": face_count,
           "frames_analisados": result.get("frames_analisados", 0),
           "desafio_codigo": desafio["codigo"], "desafio_ok": desafio["codigo"] == "CONCLUIDO",
           **{key: evidence.get(key) for key in FIELDS if key.startswith("pad_model_")},
           "decode_ms": metricas["decode_ms"], "total_ms": metricas["total_ms"]}
    path = Path(CALIBRATION_LOG_PATH)
    with LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open(newline="", encoding="utf-8") as source:
                if next(csv.reader(source), None) != FIELDS:
                    raise ValueError("Formato de calibracao antigo; configure um novo arquivo")
        with path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def ler_calibracao(limit):
    path = Path(CALIBRATION_LOG_PATH)
    with LOCK:
        if not path.exists():
            return {"status": "sem_logs", "linhas": []}
        with path.open(newline="", encoding="utf-8") as source:
            rows = deque(csv.DictReader(source), maxlen=limit)
    return {"status": "ok", "linhas": list(rows)}
