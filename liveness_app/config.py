import os
TEMPORAL_WINDOW_SIZE = int(os.getenv("TEMPORAL_WINDOW_SIZE", "12"))
TEMPORAL_MIN_FRAMES = int(os.getenv("TEMPORAL_MIN_FRAMES", "5"))
TEMPORAL_REAL_THRESHOLD = float(os.getenv("TEMPORAL_REAL_THRESHOLD", "0.70"))
TEMPORAL_SPOOF_THRESHOLD = float(os.getenv("TEMPORAL_SPOOF_THRESHOLD", "0.45"))
TEMPORAL_STABILITY_DELTA = float(os.getenv("TEMPORAL_STABILITY_DELTA", "0.18"))
TEMPORAL_MATCH_IOU = float(os.getenv("TEMPORAL_MATCH_IOU", "0.30"))
TEMPORAL_STALE_FRAMES = int(os.getenv("TEMPORAL_STALE_FRAMES", "20"))
PAD_REAL_THRESHOLD = float(os.getenv("PAD_REAL_THRESHOLD", "0.72"))
PAD_SPOOF_THRESHOLD = float(os.getenv("PAD_SPOOF_THRESHOLD", "0.42"))
PAD_MIN_MOTION_SCORE = float(os.getenv("PAD_MIN_MOTION_SCORE", "0.18"))
PAD_STRONG_ATTACK_SCORE = float(os.getenv("PAD_STRONG_ATTACK_SCORE", "0.62"))
PAD_FLAT_SUPPORT_SCORE = float(os.getenv("PAD_FLAT_SUPPORT_SCORE", "0.48"))
PAD_MIN_REAL_FACE_SCALE = float(os.getenv("PAD_MIN_REAL_FACE_SCALE", "0.23"))
PAD_LARGE_REAL_FACE_SCALE = float(os.getenv("PAD_LARGE_REAL_FACE_SCALE", "0.30"))
PAD_PARTIAL_REAL_FACE_SCALE = float(os.getenv("PAD_PARTIAL_REAL_FACE_SCALE", "0.16"))
PAD_MODEL_PATH = os.getenv("PAD_MODEL_PATH", "/app/models/minifasnet_v2.onnx")
PAD_MODEL_PATHS = [
    caminho.strip()
    for caminho in os.getenv("PAD_MODEL_PATHS", PAD_MODEL_PATH).split(",")
    if caminho.strip()
]
PAD_MODEL_ENABLED = os.getenv("PAD_MODEL_ENABLED", "true").lower() == "true"
PAD_MODEL_REAL_THRESHOLD = float(os.getenv("PAD_MODEL_REAL_THRESHOLD", "0.72"))
PAD_MODEL_ATTACK_THRESHOLD = float(os.getenv("PAD_MODEL_ATTACK_THRESHOLD", "0.55"))
PAD_MODEL_CROP_SCALES = [
    float(valor.strip())
    for valor in os.getenv("PAD_MODEL_CROP_SCALES", "2.0,2.7,3.4").split(",")
    if valor.strip()
]
CALIBRATION_LOG_ENABLED = os.getenv("CALIBRATION_LOG_ENABLED", "true").lower() == "true"
CALIBRATION_LOG_PATH = os.getenv(
    "CALIBRATION_LOG_PATH", "/app/logs/liveness_calibration.csv"
)
ACTIVE_CHALLENGE_ENABLED = (
    os.getenv("ACTIVE_CHALLENGE_ENABLED", "true").lower() == "true"
)
ACTIVE_CHALLENGE_SECONDS = int(os.getenv("ACTIVE_CHALLENGE_SECONDS", "12"))
ACTIVE_CHALLENGE_MIN_SCORE = float(os.getenv("ACTIVE_CHALLENGE_MIN_SCORE", "0.45"))

CHALLENGES = [
    ("PISQUE", "Pisque"),
    ("VIRE_ESQUERDA", "Vire a cabeca para a esquerda"),
    ("VIRE_DIREITA", "Vire a cabeca para a direita"),
    ("APROXIME", "Aproxime o rosto"),
]

