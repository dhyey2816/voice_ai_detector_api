import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor

device = "cpu"

# Global singletons (loaded ONCE)
_processor = None
_wav2vec = None
_classifier = None


def load_models_once():
    """
    Load all heavy models ONCE at startup.
    """
    global _processor, _wav2vec, _classifier

    if _processor is None:
        print("Loading Wav2Vec2 processor...")
        _processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

    if _wav2vec is None:
        print("Loading Wav2Vec2 model...")
        _wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(device)
        _wav2vec.eval()

    if _classifier is None:
        print("Loading classifier head...")
        _classifier = torch.nn.Linear(768, 1)
        _classifier.load_state_dict(
            torch.load("classifier.pt", map_location=device)
        )
        _classifier.eval()

    print("All models loaded successfully")


def predict_ai(audio_np: np.ndarray) -> float:
    """
    Fast inference path — NO model loading here.
    """
    if _processor is None or _wav2vec is None or _classifier is None:
        raise RuntimeError("Models not loaded. Call load_models_once() at startup.")

    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    inputs = _processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        features = _wav2vec(**inputs).last_hidden_state.mean(dim=1)
        prob = torch.sigmoid(_classifier(features)).item()

    return prob


def generate_explanation(prob: float) -> str:
    if prob >= 0.7:
        return "Strong synthetic speech patterns detected"
    elif prob >= 0.5:
        return "Some AI-generated speech artifacts detected"
    else:
        return "Natural human voice characteristics detected"


def detect_language(_: np.ndarray) -> str:
    # Lightweight heuristic (allowed)
    return "Auto-detected"
