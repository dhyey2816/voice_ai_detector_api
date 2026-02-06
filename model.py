import numpy as np
import torch
import librosa

device = "cpu"
_model = None


def load_models_once():
    global _model
    if _model is None:
        print("Loading lightweight classifier...")
        _model = torch.load("classifier.pt", map_location=device)
        _model.eval()
        print("Lightweight classifier loaded")


def extract_features(audio_np, sr=16000):
    mfcc = librosa.feature.mfcc(
        y=audio_np,
        sr=sr,
        n_mfcc=20
    )
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.var(axis=1)])
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)


def predict_ai(audio_np):
    if _model is None:
        raise RuntimeError("Model not loaded")

    features = extract_features(audio_np)

    with torch.no_grad():
        prob = torch.sigmoid(_model(features)).item()

    return prob


def generate_explanation(prob: float) -> str:
    if prob >= 0.7:
        return "Strong synthetic speech artifacts detected"
    elif prob >= 0.5:
        return "Some AI-generated speech patterns detected"
    else:
        return "Natural human voice characteristics detected"


def detect_language(_):
    return "Auto-detected"
