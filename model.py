import torch
import numpy as np
from torch import nn

device = "cpu"

_model = None

def load_models_once():
    global _model

    if _model is not None:
        return

    print("Loading lightweight classifier...")

    _model = nn.Sequential(
        nn.Linear(40, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

    state_dict = torch.load(
        "classifier.pt",
        map_location=device
    )

    _model.load_state_dict(state_dict)
    _model.eval()

    print("Lightweight classifier loaded")


def predict_ai(audio_np: np.ndarray) -> float:
    if _model is None:
        load_models_once()

    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    with torch.no_grad():
        x = torch.tensor(audio_np, dtype=torch.float32)
        logits = _model(x)
        prob = torch.sigmoid(logits).item()

    return prob


def generate_explanation(prob: float) -> str:
    if prob >= 0.7:
        return "Strong synthetic speech patterns detected"
    elif prob >= 0.5:
        return "Some AI-generated speech artifacts detected"
    else:
        return "Natural human voice characteristics detected"


def detect_language(_: np.ndarray) -> str:
    return "Auto-detected"
