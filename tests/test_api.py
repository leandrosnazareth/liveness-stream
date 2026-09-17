from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from liveness_app import main
from liveness_app.core.sessions import SessionStore

KEY = "integration-test-service-key-123456789"


@pytest.fixture
def api(monkeypatch):
    now = [0.]
    store = SessionStore(clock=lambda: now[0])
    face = SimpleNamespace(bbox=np.array([100, 100, 350, 350]),
                           landmark_3d_68=np.ones((68, 3)), embedding=np.ones(32))
    faces = [face]
    values = {"pad_model_usado": True, "pad_model_live": .98, "pad_model_print": .01,
              "pad_model_replay": .01, "pad_model_attack": .02, "foto_score": .1,
              "tela_score": .1, "suporte_plano": 0., "eye_aspect_ratio": .3,
              "nose_x_ratio": .5, "qualidade_ok": True}
    monkeypatch.setattr(main, "sessions", store)
    monkeypatch.setattr(main, "SERVICE_API_KEY", KEY)
    monkeypatch.setattr(main, "DEMO_MODE", False)
    monkeypatch.setattr(main.state, "load", lambda: None)
    monkeypatch.setattr(main.state, "model_loaded", True)
    monkeypatch.setattr(main.state, "pad_model_loaded", True)
    monkeypatch.setattr(main.state, "app_face", SimpleNamespace(get=lambda frame: faces))
    monkeypatch.setattr(main, "calcular_evidencias", lambda *args: dict(values))
    with TestClient(main.app) as client:
        yield SimpleNamespace(client=client, now=now, store=store, face=face, faces=faces, values=values)


def create(api, subject="a", tx="tx"):
    response = api.client.post("/sessions", json={"subject_id": subject, "transaction_id": tx},
                               headers={"X-API-Key": KEY})
    assert response.status_code == 201
    return response.json()


def send(api, session, sequence, content=None, **headers):
    if content is None:
        frame = np.full((480, 640, 3), 50 + sequence, np.uint8)
        content = cv2.imencode(".jpg", frame)[1].tobytes()
    return api.client.post("/predict", files={"file": ("frame.jpg", content, "image/jpeg")},
        headers={"Authorization": "Bearer " + session["capture_token"],
                 "X-Session-ID": session["session_id"], "X-Frame-Sequence": str(sequence), **headers})


def test_authorization_readiness_and_demo_disabled(api, monkeypatch):
    assert api.client.post("/sessions", json={"subject_id": "a", "transaction_id": "t"}).status_code == 401
    assert api.client.get("/calibration").status_code == 401
    assert api.client.post("/demo/sessions").status_code == 404
    assert api.client.post("/predict", files={"file": ("x", b"bad")}).status_code == 401
    monkeypatch.setattr(main.state, "pad_model_loaded", False)
    assert api.client.get("/health").status_code == 503
    assert api.client.post("/sessions", headers={"X-API-Key": KEY},
                           json={"subject_id": "a", "transaction_id": "t"}).status_code == 503


def test_sessions_are_isolated_even_at_identical_coordinates(api):
    a, b = create(api), create(api, "b", "tx2")
    assert send(api, a, 1, Authorization="Bearer " + b["capture_token"]).status_code == 401
    result = send(api, a, 1)
    assert result.status_code == 200
    assert result.json()["decisao"]["estado"] == "EM_ANALISE"
    assert len(api.store.sessions[a["session_id"]].temporal.samples) == 1
    assert len(api.store.sessions[b["session_id"]].temporal.samples) == 0
    assert result.headers["cache-control"] == "no-store"


def test_replay_is_terminal_on_first_frame_without_screen_border(api):
    session = create(api)
    api.values.update(pad_model_live=0., pad_model_replay=1., pad_model_attack=1.)
    result = send(api, session, 1).json()
    assert result["decisao"]["motivo"] == "ATAQUE_PAD"
    assert result["encerrada"] and not result["autoriza_transacao"]
    api.now[0] = .25
    api.values.update(pad_model_live=.98, pad_model_replay=.01, pad_model_attack=.02)
    assert send(api, session, 2).status_code == 409


@pytest.mark.parametrize("case,reason", [("multi", "MULTIPLAS_FACES"), ("swap", "TROCA_DE_FACE"), ("lost", "FACE_PERDIDA")])
def test_face_continuity_guards(api, case, reason):
    session = create(api)
    assert send(api, session, 1).status_code == 200
    api.now[0] = .25
    if case == "multi":
        api.faces.append(api.face)
    elif case == "lost":
        api.faces.clear()
    else:
        api.face.embedding = -np.ones(32)
    response = send(api, session, 2).json()
    assert response["encerrada"] and response["decisao"]["motivo"] == reason


def test_duplicate_pixels_and_out_of_order_frames(api):
    session = create(api)
    jpg = cv2.imencode(".jpg", np.full((480, 640, 3), 100, np.uint8))[1].tobytes()
    send(api, session, 1, jpg)
    api.now[0] = .25
    assert send(api, session, 2, jpg).json()["decisao"]["motivo"] == "FRAME_REPETIDO"
    other = create(api, "b", "tx2")
    assert send(api, other, 2).status_code == 409


def test_malformed_upload_invalidates_attempt(api):
    session = create(api)
    assert send(api, session, 1, b"not an image").status_code == 422
    assert api.store.sessions[session["session_id"]].result["estado"] == "INCONCLUSIVO"


def test_runtime_pad_loss_and_processing_timeout_cannot_approve(api, monkeypatch):
    session = create(api)
    monkeypatch.setattr(main.state, "pad_model_loaded", False)
    response = send(api, session, 1)
    assert response.json()["decisao"]["motivo"] == "PAD_INDISPONIVEL"
    monkeypatch.setattr(main.state, "pad_model_loaded", True)
    session = create(api, "b", "tx2")
    original = main.calcular_evidencias
    def delayed(*args):
        api.now[0] += 4
        return original(*args)
    monkeypatch.setattr(main, "calcular_evidencias", delayed)
    response = send(api, session, 1).json()
    assert response["decisao"]["motivo"] == "PROCESSAMENTO_EXCEDEU_PRAZO"
    assert response["faces"] == [] and response["encerrada"]


def test_oversized_image_rejected_before_opencv_decode(api, monkeypatch):
    session = create(api)
    monkeypatch.setattr(main, "MAX_IMAGE_PIXELS", 160 * 160)
    def unexpected_decode(*args):
        raise AssertionError("OpenCV must not allocate oversized image")
    jpg = cv2.imencode(".jpg", np.ones((480, 640, 3), np.uint8))[1].tobytes()
    monkeypatch.setattr(cv2, "imdecode", unexpected_decode)
    assert send(api, session, 1, jpg).status_code == 422


def test_pipeline_can_complete_and_backend_consumes_once(api):
    session = create(api)
    internal = api.store.sessions[session["session_id"]]
    internal.challenge.steps = ("VIRE_ESQUERDA", "VIRE_DIREITA")
    noses = [.5, .5, .55, .60, .60, .60, .60, .55, .50, .50, .50]
    for i, nose in enumerate(noses, 1):
        api.now[0] = i * .25
        api.values["nose_x_ratio"] = nose
        response = send(api, session, i)
        assert response.status_code == 200, response.text
    assert response.json()["decisao"]["estado"] == "APROVADO"
    assert internal.identity is None and not internal.temporal.samples
    path = f'/sessions/{session["session_id"]}/consume'
    binding = {"subject_id": "a", "transaction_id": "tx"}
    assert api.client.post(path, json=binding).status_code == 401
    assert api.client.post(path, headers={"X-API-Key": KEY},
                           json={**binding, "transaction_id": "other"}).status_code == 404
    result = api.client.post(path, json=binding, headers={"X-API-Key": KEY})
    assert result.status_code == 200 and result.json()["autoriza_transacao"] is False
    assert api.client.post(path, json=binding, headers={"X-API-Key": KEY}).status_code == 409


def test_demo_result_cannot_be_consumed(api, monkeypatch):
    monkeypatch.setattr(main, "DEMO_MODE", True)
    response = api.client.post("/demo/sessions")
    assert response.status_code == 201
    session = api.store.sessions[response.json()["session_id"]]
    session.result = {"estado": "APROVADO"}
    assert api.client.post(f"/sessions/{session.id}/consume", headers={"X-API-Key": KEY},
        json={"subject_id": session.subject_id, "transaction_id": session.transaction_id}).status_code == 409
