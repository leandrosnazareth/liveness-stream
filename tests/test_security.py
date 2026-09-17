from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from liveness_app.core.challenge import Challenge, avaliar_desafio
from liveness_app.core.sessions import SessionError, SessionStore
from liveness_app.core.temporal import TemporalState
from liveness_app.core.pad import PadAdapter, PadContract, PadContractError, decode_output, preparar_entrada_pad, sha256


def evidence(**changes):
    return dict({"pad_model_usado": True, "pad_model_live": .98, "pad_model_print": .01,
                 "pad_model_replay": .01, "pad_model_attack": .02, "foto_score": .1,
                 "tela_score": .1, "suporte_plano": 0., "eye_aspect_ratio": .3,
                 "nose_x_ratio": .5, "qualidade_ok": True, "escala_face": .55}, **changes)


@pytest.mark.parametrize("changes,state,reason", [
    ({"pad_model_live": 0., "pad_model_replay": 1., "pad_model_attack": 1.}, "REJEITADO", "ATAQUE_PAD"),
    ({"pad_model_usado": False}, "INCONCLUSIVO", "PAD_INDISPONIVEL"),
    ({"pad_model_live": float("nan")}, "INCONCLUSIVO", "EVIDENCIA_INVALIDA"),
    ({"pad_model_attack": float("inf")}, "INCONCLUSIVO", "EVIDENCIA_INVALIDA"),
    ({"qualidade_ok": False}, "INCONCLUSIVO", "QUALIDADE_INSUFICIENTE"),
    ({"tela_score": .9}, "REJEITADO", "ATAQUE_VISUAL"),
])
def test_hard_gates_override_large_face_and_history(changes, state, reason):
    temporal = TemporalState()
    for i in range(12):
        temporal.observe(evidence(nose_x_ratio=.5 + .01 * i), i * .25, False)
    result = temporal.observe(evidence(**changes), 3.25, True)
    assert (result["estado"], result["motivo"]) == (state, reason)


def test_static_face_and_unfulfilled_challenge_never_approve():
    for completed in (False, True):
        temporal = TemporalState()
        for i in range(30):
            result = temporal.observe(evidence(), i * .25, completed)
            assert result["estado"] != "APROVADO"
        assert result["movimento_natural"] == 0
        assert result["score"] == pytest.approx(.98)
        assert result["score_calibrado"] is False


def test_valid_sequence_can_approve_only_after_time_and_challenge():
    temporal = TemporalState()
    for i in range(5):
        result = temporal.observe(evidence(nose_x_ratio=.5 + .03 * i), i * .1, True)
        assert result["estado"] == "EM_ANALISE"
    result = temporal.observe(evidence(nose_x_ratio=.65), 1.1, False)
    assert result["motivo"] == "DESAFIO_PENDENTE"
    result = temporal.observe(evidence(nose_x_ratio=.65), 1.3, True)
    assert result["estado"] == "APROVADO"


def test_gray_zone_does_not_approve():
    temporal = TemporalState()
    for i in range(5):
        result = temporal.observe(evidence(pad_model_live=.7, pad_model_attack=.3), i * .3, True)
    assert result["estado"] == "INCONCLUSIVO"


def samples(noses=None, eyes=None, areas=None):
    return [{"nose_x_ratio": n, "eye_aspect_ratio": e, "area": a}
            for n, e, a in zip(noses or [.5]*5, eyes or [.3]*5, areas or [100]*5)]


def test_direction_and_blink_order():
    rightward = samples(noses=[.5, .5, .55, .60, .60])
    assert avaliar_desafio(rightward, "VIRE_ESQUERDA")
    assert not avaliar_desafio(rightward, "VIRE_DIREITA")
    assert avaliar_desafio(samples(eyes=[.3, .3, .1, .2, .3]), "PISQUE")
    assert not avaliar_desafio(samples(eyes=[.1, .1, .3, .3, .3]), "PISQUE")
    assert not avaliar_desafio(samples(eyes=[.3, .3, .3, .1, .1]), "PISQUE")
    assert not avaliar_desafio(samples(areas=[150, 150, 130, 100, 100]), "APROXIME")


def test_challenge_steps_do_not_reuse_evidence_and_expire():
    challenge = Challenge(0, steps=("VIRE_ESQUERDA", "VIRE_DIREITA"))
    for i, sample in enumerate(samples(noses=[.5, .5, .55, .60, .60])):
        challenge.observe(sample, [0, 0, 100, 100], i * .25)
    assert challenge.index == 1 and challenge.samples == [] and not challenge.complete
    challenge.observe(samples()[0], [0, 0, 100, 100], 100)
    assert challenge.expired(100) and challenge.samples == []


def store_fixture():
    now = [0.]
    return SessionStore(clock=lambda: now[0]), now


def test_session_ownership_history_replay_and_terminal_state():
    store, now = store_fixture()
    a, ta = store.create("a", "tx-a")
    b, tb = store.create("b", "tx-b")
    with pytest.raises(SessionError, match="NAO_AUTORIZADA"):
        with store.claim(a.id, tb, 1):
            pass
    with store.claim(a.id, ta, 1):
        a.temporal.observe(evidence(), now[0], False)
        assert a.register_frame(b"pixels")
    assert len(b.temporal.samples) == 0 and b.sequence == 0
    now[0] = .25
    with store.claim(a.id, ta, 2):
        assert not a.register_frame(b"pixels")
    assert a.result["estado"] == "REJEITADO"
    now[0] = .50
    with pytest.raises(SessionError, match="ENCERRADA"):
        with store.claim(a.id, ta, 3):
            pass


@pytest.mark.parametrize("delay,sequence,reason", [(4, 2, "INTERROMPIDA"), (46, 2, "EXPIRADA"), (.2, 3, "SEQUENCIA_INVALIDA")])
def test_expiry_and_order_before_reusing_history(delay, sequence, reason):
    store, now = store_fixture()
    s, token = store.create("a", "tx")
    with store.claim(s.id, token, 1):
        s.temporal.observe(evidence(), 0, False)
    now[0] = delay
    with pytest.raises(SessionError, match=reason):
        with store.claim(s.id, token, sequence):
            pass
    assert not s.temporal.samples


def test_concurrent_claim_is_rejected_and_capacity_bounded():
    store, now = store_fixture()
    store.capacity = 1
    s, token = store.create("a", "tx")
    with pytest.raises(SessionError, match="LIMITE"):
        store.create("b", "tx2")
    with store.claim(s.id, token, 1):
        with pytest.raises(SessionError, match="PROCESSAMENTO"):
            with store.claim(s.id, token, 2):
                pass


def test_face_swap_and_missing_embedding():
    store, _ = store_fixture()
    s, _ = store.create("a", "tx")
    anchor = np.zeros(16)
    anchor[0] = 1
    assert s.same_identity(anchor)
    assert not s.same_identity(np.roll(anchor, 1))
    assert s.result["motivo"] == "TROCA_DE_FACE"
    s, _ = store.create("b", "tx2")
    assert not s.same_identity(None)


def test_cancelled_capture_invalidates_session():
    store, _ = store_fixture()
    s, token = store.create("a", "tx")
    with pytest.raises(KeyboardInterrupt):
        with store.claim(s.id, token, 1):
            raise KeyboardInterrupt()
    assert not s.busy and s.result["estado"] == "INCONCLUSIVO"


def test_any_failed_pad_member_disables_entire_ensemble(monkeypatch):
    from liveness_app import models
    class Broken:
        def predict(self, frame, bbox):
            raise RuntimeError("synthetic inference failure")
    class Healthy:
        def predict(self, frame, bbox):
            return {"live": .99, "print": .005, "replay": .005}
    monkeypatch.setattr(models.state, "pad_model_loaded", True)
    monkeypatch.setattr(models.state, "pad_sessions", [Healthy(), Broken()])
    result = models.inferir_modelo_pad(None, None)
    assert not result["pad_model_usado"] and not models.state.pad_model_loaded


def test_attack_from_one_model_cannot_be_averaged_away(monkeypatch):
    from liveness_app import models
    class Model:
        def __init__(self, live):
            self.live = live
        def predict(self, frame, bbox):
            return {"live": self.live, "print": 0., "replay": 1 - self.live}
    monkeypatch.setattr(models.state, "pad_model_loaded", True)
    monkeypatch.setattr(models.state, "pad_sessions", [Model(.99), Model(.01)])
    result = models.inferir_modelo_pad(None, None)
    assert result["pad_model_live"] == .01 and result["pad_model_attack"] == .99


def test_consumption_is_bound_single_use_and_demo_cannot_be_consumed():
    store, _ = store_fixture()
    s, _ = store.create("a", "tx")
    s.result = {"estado": "APROVADO"}
    with pytest.raises(SessionError, match="VINCULO"):
        store.consume(s.id, "a", "different")
    assert store.consume(s.id, "a", "tx")["autoriza_transacao"] is False
    with pytest.raises(SessionError, match="CONSUMIVEL"):
        store.consume(s.id, "a", "tx")
    d, _ = store.create("demo", "demo-tx", demo=True)
    d.result = {"estado": "APROVADO"}
    with pytest.raises(SessionError, match="CONSUMIVEL"):
        store.consume(d.id, "demo", "demo-tx")


def contract(**changes):
    base = PadContract("a"*64, "input", "output", ("print", "live", "replay"),
                       "BGR", 255., 2.7, "upstream_rect", "logits", "reference.npz", "b"*64, "test-only")
    return replace(base, **changes)


@pytest.mark.parametrize("raw", [np.array([[np.nan, 1, 0]]), np.ones((1, 4)), np.ones(3), np.ones((2, 3))])
def test_malformed_model_output_fails_closed(raw):
    with pytest.raises(PadContractError):
        decode_output(raw, contract())


def test_explicit_logits_mapping_even_when_logits_sum_to_one():
    result = decode_output(np.array([[0., 1., 0.]]), contract())
    assert result["live"] == pytest.approx(np.e / (np.e + 2))
    assert result["print"] > 0
    with pytest.raises(PadContractError):
        decode_output(np.array([[-1., 2., 0.]]), contract(output_kind="probabilities"))


def test_preprocessing_contract_controls_channels_range_and_crop():
    frame = np.full((100, 120, 3), [10, 20, 30], dtype=np.uint8)
    tensor = preparar_entrada_pad(frame, [30, 30, 60, 70], contract(divisor=1., color_order="RGB"))
    assert tensor.shape == (1, 3, 80, 80) and tensor.dtype == np.float32
    assert tensor[0, :, 0, 0].tolist() == [30, 20, 10]
    with pytest.raises(PadContractError):
        preparar_entrada_pad(frame, [-1, 30, 60, 70], contract())


class FakeOnnx:
    def get_inputs(self):
        return [SimpleNamespace(name="input", type="tensor(float)", shape=[1, 3, 80, 80])]

    def get_outputs(self):
        return [SimpleNamespace(name="output", type="tensor(float)", shape=[1, 3])]

    def run(self, names, feeds):
        index = int(round(float(feeds["input"][0, 0, 0, 0])))
        output = np.zeros((1, 3), dtype=np.float32)
        output[0, index] = 1.
        return [output]


def test_reference_detects_class_inversion_normalization_and_hash(tmp_path):
    # Synthetic fixtures exercise validation mechanics, not real model accuracy.
    c = contract(divisor=1., output_kind="probabilities")
    frames = np.stack([np.full((100, 100, 3), i, np.uint8) for i in range(3)])
    boxes = np.tile([20, 20, 80, 80], (3, 1))
    inputs = np.stack([preparar_entrada_pad(f, b, c)[0] for f, b in zip(frames, boxes)])
    path = tmp_path / "reference.npz"
    np.savez(path, frames=frames, boxes=boxes, inputs=inputs,
             probabilities=[[0, 1, 0], [1, 0, 0], [0, 0, 1]], labels=["print", "live", "replay"])
    c = replace(c, reference_sha256=sha256(path))
    PadAdapter(FakeOnnx(), c).validate_reference(tmp_path / "contract.json")
    for bad in (replace(c, classes=("live", "print", "replay")), replace(c, divisor=255.),
                replace(c, reference_sha256="0"*64)):
        with pytest.raises(PadContractError):
            PadAdapter(FakeOnnx(), bad).validate_reference(tmp_path / "contract.json")
