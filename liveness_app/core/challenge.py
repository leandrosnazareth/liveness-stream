"""Session-bound challenges; coordinates refer to an unmirrored camera image."""
import secrets
from dataclasses import dataclass, field

from liveness_app.config import ACTIVE_CHALLENGE_SECONDS, TEMPORAL_MIN_FRAMES

TEXTS = {
    "PISQUE": "Olhe para a camera e pisque",
    "VIRE_ESQUERDA": "Vire a cabeca para a sua esquerda",
    "VIRE_DIREITA": "Vire a cabeca para a sua direita",
    "APROXIME": "Aproxime o rosto lentamente",
}


def avaliar_desafio(samples, code):
    if len(samples) < TEMPORAL_MIN_FRAMES:
        return False
    baseline = samples[:2]
    if code == "PISQUE":
        opened = sum(s["eye_aspect_ratio"] for s in baseline) / 2
        if opened < 0.20:
            return False
        closed = next((i for i, s in enumerate(samples[2:], 2)
                       if s["eye_aspect_ratio"] < opened * 0.65), None)
        return closed is not None and any(
            s["eye_aspect_ratio"] >= opened * 0.90 for s in samples[closed + 1:])
    if code in {"VIRE_ESQUERDA", "VIRE_DIREITA"}:
        origin = sum(s["nose_x_ratio"] for s in baseline) / 2
        direction = 1 if code == "VIRE_ESQUERDA" else -1
        return all(direction * (s["nose_x_ratio"] - origin) >= 0.08 for s in samples[-2:])
    if code == "APROXIME":
        origin = sum(s["area"] for s in baseline) / 2
        return origin > 0 and all(s["area"] >= origin * 1.20 for s in samples[-2:])
    return False


@dataclass
class Challenge:
    started: float
    steps: tuple = field(default_factory=lambda: tuple(secrets.SystemRandom().sample(list(TEXTS), 2)))
    index: int = 0
    samples: list = field(default_factory=list)

    @property
    def complete(self):
        return self.index == len(self.steps)

    def expired(self, now):
        return not self.complete and now - self.started >= ACTIVE_CHALLENGE_SECONDS

    def observe(self, evidence, bbox, now):
        if self.complete or self.expired(now):
            return
        self.samples.append({"eye_aspect_ratio": evidence["eye_aspect_ratio"],
                             "nose_x_ratio": evidence["nose_x_ratio"],
                             "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])})
        if avaliar_desafio(self.samples, self.steps[self.index]):
            self.index += 1
            self.samples.clear()
            self.started = now

    def public(self, now):
        return {
            "codigo": "CONCLUIDO" if self.complete else self.steps[self.index],
            "texto": "Desafios concluidos" if self.complete else TEXTS[self.steps[self.index]],
            "etapa": min(self.index + 1, len(self.steps)), "total_etapas": len(self.steps),
            "segundos_restantes": 0 if self.complete else max(0, int(ACTIVE_CHALLENGE_SECONDS - (now - self.started))),
        }
