"""Security defaults; thresholds require validation on the target cameras."""
import os


def integer(name, default, minimum=1):
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} deve ser >= {minimum}")
    return value


def number(name, default, minimum=0.0, maximum=1.0):
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} fora do intervalo permitido")
    return value


TEMPORAL_WINDOW_SIZE = integer("TEMPORAL_WINDOW_SIZE", 12, 5)
TEMPORAL_MIN_FRAMES = integer("TEMPORAL_MIN_FRAMES", 5, 5)
if TEMPORAL_MIN_FRAMES > TEMPORAL_WINDOW_SIZE:
    raise ValueError("TEMPORAL_MIN_FRAMES excede TEMPORAL_WINDOW_SIZE")
TEMPORAL_STABILITY_DELTA = number("TEMPORAL_STABILITY_DELTA", 0.18)
TEMPORAL_MIN_SECONDS = number("TEMPORAL_MIN_SECONDS", 1.0, 0.5, 30)
PAD_MODEL_ENABLED = os.getenv("PAD_MODEL_ENABLED", "true").lower() == "true"
PAD_MODEL_PATH = os.getenv("PAD_MODEL_PATH", "/app/models/minifasnet_v2.onnx")
PAD_MODEL_PATHS = [p.strip() for p in os.getenv("PAD_MODEL_PATHS", PAD_MODEL_PATH).split(",") if p.strip()]
PAD_MODEL_REAL_THRESHOLD = number("PAD_MODEL_REAL_THRESHOLD", 0.90, 0.5)
PAD_MODEL_ATTACK_THRESHOLD = number("PAD_MODEL_ATTACK_THRESHOLD", 0.55, 0.01)
if PAD_MODEL_REAL_THRESHOLD <= 1.0 - PAD_MODEL_ATTACK_THRESHOLD:
    raise ValueError("Limiares PAD devem manter uma zona inconclusiva")
PAD_MIN_MOTION_SCORE = number("PAD_MIN_MOTION_SCORE", 0.18, 0.01)
PAD_STRONG_ATTACK_SCORE = number("PAD_STRONG_ATTACK_SCORE", 0.62, 0.01)
PAD_FLAT_SUPPORT_SCORE = number("PAD_FLAT_SUPPORT_SCORE", 0.48, 0.01)
PAD_MIN_REAL_FACE_SCALE = number("PAD_MIN_REAL_FACE_SCALE", 0.23, 0.01)
IDENTITY_MIN_COSINE = number("IDENTITY_MIN_COSINE", 0.65, 0.1)
SESSION_TTL_SECONDS = integer("SESSION_TTL_SECONDS", 45)
SESSION_IDLE_SECONDS = number("SESSION_IDLE_SECONDS", 3.0, 0.5, 30)
SESSION_MIN_FRAME_SECONDS = number("SESSION_MIN_FRAME_SECONDS", 0.08, 0.01, 1)
SESSION_MAX_FRAMES = integer("SESSION_MAX_FRAMES", 300)
SESSION_MAX_COUNT = integer("SESSION_MAX_COUNT", 1000)
ACTIVE_CHALLENGE_SECONDS = integer("ACTIVE_CHALLENGE_SECONDS", 15)
MAX_UPLOAD_BYTES = integer("MAX_UPLOAD_BYTES", 2 * 1024 * 1024)
MAX_IMAGE_PIXELS = integer("MAX_IMAGE_PIXELS", 1920 * 1080)
SERVICE_API_KEY = os.getenv("LIVENESS_SERVICE_API_KEY", "")
DEMO_MODE = os.getenv("LIVENESS_DEMO_MODE", "false").lower() == "true"
CALIBRATION_LOG_ENABLED = os.getenv("CALIBRATION_LOG_ENABLED", "false").lower() == "true"
CALIBRATION_LOG_PATH = os.getenv("CALIBRATION_LOG_PATH", "/app/logs/liveness_calibration.csv")
