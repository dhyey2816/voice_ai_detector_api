import torch
import numpy as np
import whisper
from transformers import Wav2Vec2Model, Wav2Vec2Processor

device = "cpu"  # Render free tier safe

# --------- Lazy loaded globals ----------
_whisper_model = None
_wav2vec_model = None
_processor = None
_classifier = None

SUPPORTED_LANGS = {
    "ta": "Tamil",
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu"
}

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny")
    return _whisper_model

def get_wav2vec():
    global _wav2vec_model, _processor
    if _wav2vec_model is None:
        _processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        _wav2vec_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )
        _wav2vec_model.eval()
    return _processor, _wav2vec_model

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = torch.nn.Linear(768, 1)
        _classifier.load_state_dict(
            torch.load("classifier.pt", map_location="cpu")
        )
        _classifier.eval()
    return _classifier

def detect_language(audio_np):
    audio_np = audio_np.astype(np.float32)
    model = get_whisper_model()
    result = model.transcribe(audio_np)
    return SUPPORTED_LANGS.get(result["language"], "Unknown")

def predict_ai(audio_np):
    if len(audio_np.shape) == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    processor, wav2vec = get_wav2vec()
    classifier = get_classifier()

    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        emb = wav2vec(**inputs).last_hidden_state.mean(dim=1)
        prob = torch.sigmoid(classifier(emb)).item()

    return prob

def generate_explanation(prob):
    if prob >= 0.7:
        return "Strong synthetic voice patterns detected"
    elif prob >= 0.5:
        return "Possible AI-generated speech artifacts"
    else:
        return "Natural human voice characteristics detected"
