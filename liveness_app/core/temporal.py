"""One conservative policy; history belongs to exactly one capture session."""
from collections import deque
from dataclasses import dataclass, field
import math

from liveness_app.config import (
    PAD_FLAT_SUPPORT_SCORE, PAD_MIN_MOTION_SCORE, PAD_MODEL_ATTACK_THRESHOLD,
    PAD_MODEL_REAL_THRESHOLD, PAD_STRONG_ATTACK_SCORE, TEMPORAL_MIN_FRAMES,
    TEMPORAL_MIN_SECONDS, TEMPORAL_STABILITY_DELTA, TEMPORAL_WINDOW_SIZE,
)

TERMINAL = {"APROVADO", "REJEITADO", "INCONCLUSIVO"}


def decision(state, reason, score=None, **extra):
    return {"estado": state, "motivo": reason, "score": score,
            "score_calibrado": False, **extra}


@dataclass
class TemporalState:
    samples: deque = field(default_factory=lambda: deque(maxlen=TEMPORAL_WINDOW_SIZE))
    first_seen: float | None = None

    def observe(self, evidence, now, challenge_ok):
        if evidence.get("pad_model_usado") is not True:
            return decision("INCONCLUSIVO", "PAD_INDISPONIVEL")
        keys = ("pad_model_live", "pad_model_print", "pad_model_replay", "pad_model_attack",
                "foto_score", "tela_score", "suporte_plano", "eye_aspect_ratio", "nose_x_ratio")
        try:
            if not all(math.isfinite(float(evidence[k])) and 0 <= float(evidence[k]) <= 1 for k in keys):
                return decision("INCONCLUSIVO", "EVIDENCIA_INVALIDA")
        except (KeyError, ValueError, TypeError):
            return decision("INCONCLUSIVO", "EVIDENCIA_INVALIDA")
        # An attack in the current frame cannot be diluted by earlier frames.
        if evidence["pad_model_attack"] >= PAD_MODEL_ATTACK_THRESHOLD:
            return decision("REJEITADO", "ATAQUE_PAD", evidence["pad_model_live"])
        if (max(evidence["foto_score"], evidence["tela_score"]) >= PAD_STRONG_ATTACK_SCORE
                or evidence["suporte_plano"] >= PAD_FLAT_SUPPORT_SCORE):
            return decision("REJEITADO", "ATAQUE_VISUAL", evidence["pad_model_live"])
        if evidence.get("qualidade_ok") is not True:
            return decision("INCONCLUSIVO", "QUALIDADE_INSUFICIENTE")
        if self.first_seen is None:
            self.first_seen = now
        self.samples.append(dict(evidence))
        scores = [s["pad_model_live"] for s in self.samples]
        score = sum(scores) / len(scores)
        spread = max(scores) - min(scores)
        eyes = [s["eye_aspect_ratio"] for s in self.samples]
        noses = [s["nose_x_ratio"] for s in self.samples]
        # Motion is a requirement, never proof of physical depth/liveness.
        motion = min(1.0, max((max(eyes) - min(eyes)) / 0.08, (max(noses) - min(noses)) / 0.10))
        details = {"frames_analisados": len(scores), "estabilidade": spread,
                   "movimento_natural": motion, "desafio_ok": challenge_ok}
        if len(scores) < TEMPORAL_MIN_FRAMES or now - self.first_seen < TEMPORAL_MIN_SECONDS:
            return decision("EM_ANALISE", "COLETANDO_EVIDENCIAS", score, **details)
        if min(scores) < PAD_MODEL_REAL_THRESHOLD or spread > TEMPORAL_STABILITY_DELTA:
            return decision("INCONCLUSIVO", "PAD_INSUFICIENTE_OU_INSTAVEL", score, **details)
        if not challenge_ok:
            return decision("EM_ANALISE", "DESAFIO_PENDENTE", score, **details)
        if motion < PAD_MIN_MOTION_SCORE:
            return decision("EM_ANALISE", "MOVIMENTO_INSUFICIENTE", score, **details)
        return decision("APROVADO", "VIVACIDADE_VERIFICADA", score, **details)
