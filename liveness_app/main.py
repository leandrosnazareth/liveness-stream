import base64
from contextlib import asynccontextmanager
import hmac
import io
import logging
from pathlib import Path
import secrets
import time

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from liveness_app.config import (
    DEMO_MODE, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, SERVICE_API_KEY,
    SESSION_IDLE_SECONDS, SESSION_TTL_SECONDS, TEMPORAL_MIN_FRAMES, TEMPORAL_WINDOW_SIZE,
)
from liveness_app.core.calibration import ler_calibracao, registrar_calibracao
from liveness_app.core.evidence import calcular_evidencias
from liveness_app.core.sessions import SessionError, SessionStore
from liveness_app.core.temporal import TERMINAL, decision
from liveness_app.models import state

logger = logging.getLogger(__name__)
sessions = SessionStore()


@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(state.load)
    yield
    for session in sessions.sessions.values():
        session.clear_biometrics()
    sessions.sessions.clear()


app = FastAPI(lifespan=lifespan)
INDEX_HTML_PATH = Path(__file__).parent / "web" / "index.html"


@app.exception_handler(SessionError)
async def session_error(request, exc):
    return JSONResponse(status_code=exc.status, content={"detail": exc.code},
                        headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


def service_auth(x_api_key: str = Header(default="")):
    if len(SERVICE_API_KEY) < 32:
        raise HTTPException(503, "AUTENTICACAO_DO_SERVICO_NAO_CONFIGURADA")
    if not hmac.compare_digest(x_api_key.encode(), SERVICE_API_KEY.encode()):
        raise HTTPException(401, "SERVICO_NAO_AUTORIZADO")


class Binding(BaseModel):
    subject_id: str = Field(min_length=1, max_length=128, pattern=r"^\S+$")
    transaction_id: str = Field(min_length=1, max_length=128, pattern=r"^\S+$")


def create_session(subject_id, transaction_id, demo=False):
    if not state.ready:
        raise HTTPException(503, "MODELOS_NAO_VALIDADOS")
    session, token = sessions.create(subject_id, transaction_id, demo)
    return {"session_id": session.id, "capture_token": token, "next_sequence": 1,
            "expires_in": SESSION_TTL_SECONDS, "demo": demo,
            "desafio_ativo": session.challenge.public(sessions.clock())}


@app.post("/sessions", dependencies=[Depends(service_auth)], status_code=201)
def start_session(binding: Binding):
    # This binding is supplied by the trusted banking backend, not the capture client.
    return create_session(binding.subject_id, binding.transaction_id)


@app.post("/demo/sessions", status_code=201)
def start_demo():
    if not DEMO_MODE:
        raise HTTPException(404, "DEMO_DESABILITADA")
    return create_session("demo", "demo-" + secrets.token_urlsafe(16), demo=True)


@app.post("/sessions/{session_id}/consume", dependencies=[Depends(service_auth)])
def consume_result(session_id: str, binding: Binding):
    if not state.ready:
        raise HTTPException(503, "MODELOS_NAO_VALIDADOS")
    return sessions.consume(session_id, binding.subject_id, binding.transaction_id)


@app.get("/health")
def health():
    return JSONResponse(status_code=200 if state.ready else 503, content={
        "status": "ok" if state.ready else "indisponivel", "ready": state.ready,
        "model_loaded": state.model_loaded, "pad_model_loaded": state.pad_model_loaded,
        "pad_model_count": len(state.pad_sessions), "provider_ativo": state.provider_ativo,
        "pad_model_provider": state.pad_model_provider, "demo_mode": DEMO_MODE,
    })


@app.get("/calibration", dependencies=[Depends(service_auth)])
def calibration(limit: int = Query(20, ge=1, le=200)):
    return ler_calibracao(limit)


def decode_frame(content):
    try:
        # Check dimensions before allocating a decoded image in OpenCV.
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            if image.format != "JPEG" or min(width, height) < 160 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Dimensoes/formato invalidos")
            image.verify()
        frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (height, width):
            raise ValueError("Imagem invalida")
        return frame
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "FRAME_INVALIDO") from exc


def analyze_frame(session, frame, received):
    """All model use is serialized; session ownership is reserved across awaits."""
    with state.lock:
        if not state.ready:
            session.finish("INCONCLUSIVO", "PAD_INDISPONIVEL")
            return [], 0
        faces = state.app_face.get(frame)
        if len(faces) != 1:
            if len(faces) > 1:
                session.finish("REJEITADO", "MULTIPLAS_FACES")
            elif session.identity is not None:
                session.finish("INCONCLUSIVO", "FACE_PERDIDA")
            return [], len(faces)
        face = faces[0]
        landmarks = np.asarray(face.landmark_3d_68)
        bbox = np.asarray(face.bbox)
        if (landmarks.shape != (68, 3) or not np.isfinite(landmarks).all()
                or bbox.shape != (4,) or not np.isfinite(bbox).all()):
            session.finish("INCONCLUSIVO", "LANDMARKS_INVALIDOS")
            return [], 1
        bbox = bbox.astype(int).tolist()
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
            session.finish("INCONCLUSIVO", "ENQUADRAMENTO_INVALIDO")
            return [], 1
        evidence = calcular_evidencias(face, frame, bbox, landmarks, abs(landmarks[30, 2] - landmarks[0, 2]))
        if not session.same_identity(getattr(face, "embedding", None)):
            return [], 1
        # Evaluate hard gates BEFORE accepting challenge samples.
        result = session.temporal.observe(evidence, received, session.challenge.complete)
        if result["estado"] not in TERMINAL:
            session.challenge.observe(evidence, bbox, received)
            # Approval can happen on the next frame, with no duplicated temporal sample.
        session.result = result
        public_evidence = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 5))
                           for k, v in evidence.items() if np.isfinite(v)}
        return [{"id": 1, "bbox": bbox, **result, "evidencias": public_evidence,
                 "desafio_ok": session.challenge.complete}], 1


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  modo: str = Query("metadata", pattern="^(metadata|imagem)$"),
                  authorization: str = Header(default=""),
                  x_session_id: str = Header(default=""),
                  x_frame_sequence: int = Header(default=0, ge=0)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "TOKEN_DE_CAPTURA_OBRIGATORIO")
    with sessions.claim(x_session_id, authorization[7:], x_frame_sequence) as session:
        start = time.perf_counter()
        received = sessions.clock()
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                session.finish("INCONCLUSIVO", "FRAME_MUITO_GRANDE")
                raise HTTPException(413, "FRAME_MUITO_GRANDE")
            frame = await run_in_threadpool(decode_frame, content)
            decode_ms = (time.perf_counter() - start) * 1000
            faces, count = [], 0
            if session.register_frame(frame.tobytes()):
                faces, count = await run_in_threadpool(analyze_frame, session, frame, received)
            now = sessions.clock()
            if now >= session.expires or session.challenge.expired(now):
                session.finish("INCONCLUSIVO", "SESSAO_OU_DESAFIO_EXPIRADO")
                faces = []
            elif now - received > SESSION_IDLE_SECONDS:
                session.finish("INCONCLUSIVO", "PROCESSAMENTO_EXCEDEU_PRAZO")
                faces = []
            challenge = session.challenge.public(now)
            metrics = {"decode_ms": round(decode_ms, 2),
                       "total_ms": round((time.perf_counter() - start) * 1000, 2)}
            registrar_calibracao(session.sequence, faces, metrics, challenge,
                                 session_id=session.id, result=session.result, face_count=count)
            response = {"status": "sucesso", "session_id": session.id, "sequence": session.sequence,
                        "decisao": session.result, "encerrada": session.result["estado"] in TERMINAL,
                        "quantidade_rostos": count, "faces": faces, "desafio_ativo": challenge,
                        "provider_ativo": state.provider_ativo,
                        "resolucao": {"largura": frame.shape[1], "altura": frame.shape[0]},
                        "analise_temporal": {"janela": TEMPORAL_WINDOW_SIZE, "min_frames": TEMPORAL_MIN_FRAMES},
                        "metricas": metrics, "autoriza_transacao": False}
            if modo == "imagem":
                for face in faces:
                    x1, y1, x2, y2 = face["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if face["estado"] == "APROVADO" else (0, 160, 255), 2)
                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError("Falha ao codificar imagem")
                response["imagem_processada"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
            if response["encerrada"]:
                session.clear_biometrics()
            return response
        except HTTPException:
            raise
        except Exception:
            logger.exception("Erro ao processar captura")
            raise HTTPException(503, "PROCESSAMENTO_INDISPONIVEL")
        finally:
            await file.close()


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/capture.js")
def capture_script():
    return Response(INDEX_HTML_PATH.with_name("capture.js").read_text(encoding="utf-8"),
                    media_type="text/javascript")
