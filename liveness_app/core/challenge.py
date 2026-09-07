import time

from liveness_app.config import (
    ACTIVE_CHALLENGE_ENABLED,
    ACTIVE_CHALLENGE_MIN_SCORE,
    ACTIVE_CHALLENGE_SECONDS,
    CHALLENGES,
    TEMPORAL_MIN_FRAMES,
)
from liveness_app.core.utils import limitar

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


