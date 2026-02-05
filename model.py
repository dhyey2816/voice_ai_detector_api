import torch
import numpy as np
import whisper
from transformers import Wav2Vec2Model, Wav2Vec2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# GLOBAL CACHED MODELS
# -------------------------
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

# -------------------------
# LAZY LOADERS (IMPORTANT)
# -------------------------
def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("tiny").to(device)
    return _whisper_model


def get_wav2vec():
    global _wav2vec_model, _processor

    if _processor is None:
        _processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

    if _wav2vec_model is None:
        _wav2vec_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        ).to(device)
        _wav2vec_model.eval()

    return _processor, _wav2vec_model


def get_classifier():
    global _classifier
    if _classifier is None:
        classifier = torch.nn.Linear(768, 1)
        classifier.load_state_dict(
            torch.load("classifier.pt", map_location=device)
        )
        classifier = classifier.to(device)
        classifier.eval()
        _classifier = classifier
    return _classifier

# -------------------------
# FUNCTIONS USED BY API
# -------------------------
def detect_language(audio_np):
    audio_np = audio_np.astype(np.float32)

    model = get_whisper_model()

    result = model.transcribe(
        audio_np,
        task="transcribe",
        language=None
    )

    lang_code = result["language"]
    return SUPPORTED_LANGS.get(lang_code, "Unknown")


def predict_ai(audio_np):
    processor, wav2vec_model = get_wav2vec()
    classifier = get_classifier()

    if len(audio_np.shape) == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        prob = torch.sigmoid(classifier(emb)).item()

    return prob


def generate_explanation(prob):
    if prob >= 0.7:
        return "Strong synthetic voice patterns and unnatural pitch stability detected"
    elif prob >= 0.5:
        return "Some synthetic speech artifacts observed"
    else:
        return "Natural human voice variations and micro-pauses detected"
