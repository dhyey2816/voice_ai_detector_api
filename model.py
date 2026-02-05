import torch
import numpy as np
import whisper
from transformers import Wav2Vec2Model, Wav2Vec2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load once (speed boost)
whisper_model = whisper.load_model("tiny").to(device)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
wav2vec_model.eval()

classifier = torch.nn.Linear(768, 1)
classifier.load_state_dict(torch.load("classifier.pt", map_location=device))
classifier = classifier.to(device)
classifier.eval()

SUPPORTED_LANGS = {
    "ta": "Tamil",
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu"
}

def detect_language(audio_np):
    # Whisper expects float32
    audio_np = audio_np.astype(np.float32)

    # Whisper expects 1D, so keep it 1D
    result = whisper_model.transcribe(audio_np, task="transcribe", language=None)
    lang_code = result["language"]

    return SUPPORTED_LANGS.get(lang_code, "Unknown")

def predict_ai(audio_np):
    # 🔧 Make it batched: [1, N]
    if len(audio_np.shape) == 1:
        audio_np = np.expand_dims(audio_np, axis=0)

    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", padding=True)
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
