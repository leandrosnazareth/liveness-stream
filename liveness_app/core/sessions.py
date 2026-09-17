"""Bounded, single-process session store. Tokens are capture-only capabilities."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import secrets
import threading
import time

import numpy as np

from liveness_app.config import (
    IDENTITY_MIN_COSINE, SESSION_IDLE_SECONDS, SESSION_MAX_COUNT,
    SESSION_MAX_FRAMES, SESSION_MIN_FRAME_SECONDS, SESSION_TTL_SECONDS,
)
from liveness_app.core.challenge import Challenge
from liveness_app.core.temporal import TERMINAL, TemporalState, decision


class SessionError(Exception):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.code, self.status = code, status


@dataclass
class CaptureSession:
    id: str
    token_hash: bytes
    subject_id: str
    transaction_id: str
    created: float
    expires: float
    demo: bool = False
    last_seen: float | None = None
    sequence: int = 0
    busy: bool = False
    consumed: bool = False
    hashes: set = field(default_factory=set)
    identity: np.ndarray | None = None
    temporal: TemporalState = field(default_factory=TemporalState)
    result: dict = field(default_factory=lambda: decision("EM_ANALISE", "AGUARDANDO_FACE"))
    challenge: Challenge = field(init=False)

    def __post_init__(self):
        self.challenge = Challenge(self.created)

    def finish(self, state, reason):
        self.result = decision(state, reason)
        self.clear_biometrics()
        return self.result

    def clear_biometrics(self):
        self.identity = None
        self.temporal.samples.clear()
        self.hashes.clear()
        self.challenge.samples.clear()

    def register_frame(self, pixels):
        digest = hashlib.sha256(pixels).digest()
        if digest in self.hashes:
            self.finish("REJEITADO", "FRAME_REPETIDO")
            return False
        self.hashes.add(digest)
        return True

    def same_identity(self, embedding):
        if embedding is None:
            self.finish("INCONCLUSIVO", "CONTINUIDADE_INDISPONIVEL")
            return False
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.ndim != 1 or vector.size < 16 or not np.isfinite(vector).all():
            self.finish("INCONCLUSIVO", "CONTINUIDADE_INVALIDA")
            return False
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            self.finish("INCONCLUSIVO", "CONTINUIDADE_INVALIDA")
            return False
        vector = vector / norm
        if self.identity is None:
            self.identity = vector.copy()
        elif self.identity.shape != vector.shape or np.dot(self.identity, vector) < IDENTITY_MIN_COSINE:
            self.finish("REJEITADO", "TROCA_DE_FACE")
            return False
        return True


class SessionStore:
    def __init__(self, clock=time.monotonic, capacity=SESSION_MAX_COUNT):
        self.clock, self.capacity = clock, capacity
        self.sessions = {}
        self.lock = threading.RLock()

    def create(self, subject_id, transaction_id, demo=False):
        with self.lock:
            now = self.clock()
            expired = [sid for sid, s in self.sessions.items() if now >= s.expires and not s.busy]
            for sid in expired:
                self.sessions.pop(sid).clear_biometrics()
            if len(self.sessions) >= self.capacity:
                raise SessionError("LIMITE_DE_SESSOES", 429)
            token = secrets.token_urlsafe(32)
            session = CaptureSession(secrets.token_urlsafe(24), hashlib.sha256(token.encode()).digest(),
                                     subject_id, transaction_id, now, now + SESSION_TTL_SECONDS, demo)
            self.sessions[session.id] = session
            return session, token

    def authenticate(self, sid, token):
        session = self.sessions.get(sid)
        digest = hashlib.sha256(token.encode()).digest()
        if session is None or not hmac.compare_digest(session.token_hash, digest):
            raise SessionError("SESSAO_NAO_AUTORIZADA", 401)
        if self.clock() >= session.expires:
            if not session.busy:
                session.finish("INCONCLUSIVO", "SESSAO_EXPIRADA")
            raise SessionError("SESSAO_EXPIRADA", 410)
        return session

    @contextmanager
    def claim(self, sid, token, sequence):
        with self.lock:
            session = self.authenticate(sid, token)
            now = self.clock()
            if session.busy:
                raise SessionError("FRAME_EM_PROCESSAMENTO")
            if session.result["estado"] in TERMINAL:
                raise SessionError("SESSAO_ENCERRADA")
            if sequence != session.sequence + 1:
                session.finish("REJEITADO", "SEQUENCIA_INVALIDA")
                raise SessionError("SEQUENCIA_INVALIDA")
            if session.last_seen is not None:
                if now - session.last_seen > SESSION_IDLE_SECONDS:
                    session.finish("INCONCLUSIVO", "CAPTURA_INTERROMPIDA")
                    raise SessionError("CAPTURA_INTERROMPIDA", 410)
                if now - session.last_seen < SESSION_MIN_FRAME_SECONDS:
                    raise SessionError("CADENCIA_EXCESSIVA", 429)
            if sequence > SESSION_MAX_FRAMES or session.challenge.expired(now):
                session.finish("INCONCLUSIVO", "DESAFIO_EXPIRADO")
                raise SessionError("DESAFIO_EXPIRADO", 410)
            session.busy, session.sequence, session.last_seen = True, sequence, now
        try:
            yield session
        except BaseException:
            # Cancellation/disconnect must not leave a partly processed attempt reusable.
            session.finish("INCONCLUSIVO", "ERRO_NA_CAPTURA")
            raise
        finally:
            with self.lock:
                session.busy = False

    def consume(self, sid, subject_id, transaction_id):
        with self.lock:
            s = self.sessions.get(sid)
            if s is None or (s.subject_id, s.transaction_id) != (subject_id, transaction_id):
                raise SessionError("VINCULO_INVALIDO", 404)
            if self.clock() >= s.expires:
                s.finish("INCONCLUSIVO", "SESSAO_EXPIRADA")
                raise SessionError("SESSAO_EXPIRADA", 410)
            if s.demo or s.busy or s.consumed or s.result["estado"] != "APROVADO":
                raise SessionError("RESULTADO_NAO_CONSUMIVEL")
            s.consumed = True
            return {"session_id": s.id, "subject_id": s.subject_id,
                    "transaction_id": s.transaction_id, "resultado_pad": dict(s.result),
                    "autoriza_transacao": False}
