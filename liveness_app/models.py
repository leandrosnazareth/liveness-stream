"""Models load at application startup, never as an import side effect."""
import logging
import threading

from liveness_app.config import PAD_MODEL_ENABLED, PAD_MODEL_PATHS
from liveness_app.core.pad import PadAdapter, PadContract, PadContractError, sha256

logger = logging.getLogger(__name__)


class ModelState:
    def __init__(self):
        self.model_loaded = False
        self.app_face = None
        self.provider_ativo = "CPU"
        self.pad_sessions = []
        self.pad_model_loaded = False
        self.pad_model_provider = None
        self.ort_providers = []
        self.lock = threading.Lock()

    @property
    def ready(self):
        return self.model_loaded and self.pad_model_loaded

    def load(self):
        self.model_loaded = self.pad_model_loaded = False
        self.pad_sessions = []
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            self.ort_providers = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in self.ort_providers else ["CPUExecutionProvider"])
            self.app_face = FaceAnalysis(
                allowed_modules=["detection", "landmark_3d_68", "recognition"], providers=providers)
            self.app_face.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1,
                                  det_size=(640, 640))
            if not {"detection", "landmark_3d_68", "recognition"}.issubset(self.app_face.models):
                raise RuntimeError("Modelos faciais obrigatorios ausentes")
            self.model_loaded = True
            self.provider_ativo = "CUDA" if providers[0] == "CUDAExecutionProvider" else "CPU"
            if not PAD_MODEL_ENABLED or not PAD_MODEL_PATHS:
                raise PadContractError("PAD obrigatorio desabilitado")
            adapters = []
            for path in PAD_MODEL_PATHS:
                contract_path = path + ".json"
                contract = PadContract.load(contract_path)
                if sha256(path) != contract.model_sha256:
                    raise PadContractError("SHA256 do modelo PAD diverge")
                session = ort.InferenceSession(path, providers=providers)
                adapter = PadAdapter(session, contract)
                adapter.validate_reference(contract_path)
                adapters.append(adapter)
            # Commit only when EVERY configured model passes its contract/reference.
            self.pad_sessions = adapters
            self.pad_model_provider = adapters[0].session.get_providers()[0]
            self.pad_model_loaded = True
        except Exception:
            logger.exception("Modelos indisponiveis para aprovacao de vivacidade")


state = ModelState()


def inferir_modelo_pad(frame, bbox):
    unavailable = {"pad_model_live": 0.0, "pad_model_print": 0.0, "pad_model_replay": 0.0,
                   "pad_model_attack": 0.0, "pad_model_ensemble": 0, "pad_model_usado": False}
    if not state.pad_model_loaded or not state.pad_sessions:
        return unavailable
    try:
        results = [adapter.predict(frame, bbox) for adapter in state.pad_sessions]
    except Exception:
        state.pad_model_loaded = False
        logger.exception("Falha PAD: novas aprovacoes bloqueadas ate reinicializacao/validacao")
        return unavailable
    return {
        "pad_model_live": min(p["live"] for p in results),
        "pad_model_print": max(p["print"] for p in results),
        "pad_model_replay": max(p["replay"] for p in results),
        "pad_model_attack": max(p["print"] + p["replay"] for p in results),
        "pad_model_ensemble": len(results), "pad_model_usado": True,
    }
