# Attention U-Net: build the architecture or load its completed best checkpoint.
from pathlib import Path

from kssd.models import build_model

MODEL_NAME = "Attention U-Net"
ARCHITECTURE = "attention_unet"
RUN_NAME = "attention256_50epochs_20261014_v2"
STRAIN_ROOT = Path(__file__).resolve().parents[1]
RESULTS = STRAIN_ROOT / "results" / RUN_NAME / ARCHITECTURE
BEST_CHECKPOINT = (
    STRAIN_ROOT / "checkpoints" / RUN_NAME / "attention_unet.best.weights.h5"
)


def build(input_shape=(256, 256, 1)):
    return build_model(ARCHITECTURE, input_shape)


def load_best():
    if not BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Local trained weights are missing: {BEST_CHECKPOINT}")
    model = build()
    model.load_weights(BEST_CHECKPOINT)
    return model
