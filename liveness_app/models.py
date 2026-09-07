import os

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from liveness_app.config import (
    PAD_MODEL_CROP_SCALES,
    PAD_MODEL_ENABLED,
    PAD_MODEL_PATHS,
)
from liveness_app.core.utils import limitar, softmax


class ModelState:
    def __init__(self):
        self.ort_providers = ort.get_available_providers()
        self.model_loaded = False
        self.app_face = None
        self.provider_ativo = "CPU"
        self.pad_session = None
        self.pad_input_name = None
        self.pad_sessions = []
        self.pad_model_loaded = False
        self.pad_model_provider = None


state = ModelState()
print(f"[*] ONNXRuntime providers disponiveis: {state.ort_providers}")

try:
    if "CUDAExecutionProvider" not in state.ort_providers:
        raise RuntimeError("CUDAExecutionProvider indisponivel no ONNXRuntime")

    state.app_face = FaceAnalysis(
        allowed_modules=["detection", "landmark_3d_68"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    state.app_face.prepare(ctx_id=0, det_size=(640, 640))
    state.provider_ativo = "CUDA"
    state.model_loaded = True
    print("[*] Modelo Multi-Face Neural carregado com sucesso!")
except Exception as e:
    print(f"[-] Erro ao carregar na GPU, usando modo de compatibilidade: {e}")
    state.app_face = FaceAnalysis(allowed_modules=["detection", "landmark_3d_68"])
    state.app_face.prepare(ctx_id=-1, det_size=(640, 640))
    state.provider_ativo = "CPU"
    state.model_loaded = True

if PAD_MODEL_ENABLED:
    try:
        pad_providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if state.provider_ativo == "CUDA"
            else ["CPUExecutionProvider"]
        )

        for caminho_modelo in PAD_MODEL_PATHS:
            if not os.path.exists(caminho_modelo):
                print(f"[-] Modelo PAD nao encontrado: {caminho_modelo}")
                continue

            session = ort.InferenceSession(caminho_modelo, providers=pad_providers)
            input_name = session.get_inputs()[0].name
            state.pad_sessions.append(
                {
                    "path": caminho_modelo,
                    "session": session,
                    "input_name": input_name,
                    "provider": session.get_providers()[0],
                }
            )

        if not state.pad_sessions:
            raise FileNotFoundError(
                f"Nenhum modelo PAD carregado em: {', '.join(PAD_MODEL_PATHS)}"
            )

        state.pad_session = state.pad_sessions[0]["session"]
        state.pad_input_name = state.pad_sessions[0]["input_name"]
        state.pad_model_provider = state.pad_sessions[0]["provider"]
        state.pad_model_loaded = True
        print(f"[*] Modelos PAD carregados: {[m['path'] for m in state.pad_sessions]}")
        print(f"[*] Provider PAD ativo: {state.pad_model_provider}")
    except Exception as e:
        print(f"[-] Modelo PAD indisponivel, usando heuristicas: {e}")

def recortar_face_expandida(frame, bbox, escala=2.7):
    altura, largura = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    largura_face = max(1, x2 - x1)
    altura_face = max(1, y2 - y1)
    centro_x = (x1 + x2) / 2.0
    centro_y = (y1 + y2) / 2.0
    lado = max(largura_face, altura_face) * escala

    nx1 = int(max(0, centro_x - lado / 2.0))
    ny1 = int(max(0, centro_y - lado / 2.0))
    nx2 = int(min(largura, centro_x + lado / 2.0))
    ny2 = int(min(altura, centro_y + lado / 2.0))
    crop = frame[ny1:ny2, nx1:nx2]
    return crop if crop.size else None


def preparar_entrada_pad(crop):
    entrada = cv2.resize(crop, (80, 80)).astype(np.float32) / 255.0
    entrada = np.transpose(entrada, (2, 0, 1))[None, :, :, :]
    return entrada


def inferir_sessao_pad(sessao, nome_entrada, entrada):
    saida = sessao.run(None, {nome_entrada: entrada})[0]
    probabilidades = np.asarray(saida).reshape(-1)
    if probabilidades.size < 3:
        return None

    if not np.isclose(float(np.sum(probabilidades[:3])), 1.0, atol=0.08):
        probabilidades = softmax(probabilidades[:3])
    else:
        probabilidades = probabilidades[:3]
    return probabilidades


def inferir_modelo_pad(frame, bbox):
    if not state.pad_model_loaded or not state.pad_sessions:
        return {
            "pad_model_live": 0.0,
            "pad_model_print": 0.0,
            "pad_model_replay": 0.0,
            "pad_model_attack": 0.0,
            "pad_model_ensemble": 0,
            "pad_model_usado": False,
        }

    resultados = []
    for escala in PAD_MODEL_CROP_SCALES:
        crop = recortar_face_expandida(frame, bbox, escala=escala)
        if crop is None:
            continue
        entrada = preparar_entrada_pad(crop)
        for modelo in state.pad_sessions:
            probabilidades = inferir_sessao_pad(
                modelo["session"], modelo["input_name"], entrada
            )
            if probabilidades is not None:
                resultados.append(probabilidades)

    if not resultados:
        return {
            "pad_model_live": 0.0,
            "pad_model_print": 0.0,
            "pad_model_replay": 0.0,
            "pad_model_attack": 0.0,
            "pad_model_ensemble": 0,
            "pad_model_usado": False,
        }

    probabilidades = np.mean(np.stack(resultados), axis=0)
    live = float(probabilidades[0])
    print_attack = float(probabilidades[1])
    replay_attack = float(probabilidades[2])
    return {
        "pad_model_live": limitar(live),
        "pad_model_print": limitar(print_attack),
        "pad_model_replay": limitar(replay_attack),
        "pad_model_attack": limitar(print_attack + replay_attack),
        "pad_model_ensemble": len(resultados),
        "pad_model_usado": True,
    }


