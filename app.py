import os
import base64
import tempfile
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from audio_utils import load_audio
from model import predict_ai, generate_explanation, detect_language

API_KEY = os.getenv("API_KEY")

app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class VoiceRequest(BaseModel):
    audioFormat: str
    audioBase64: str

    class Config:
        extra = "ignore"


@app.post("/api/voice-detection")
def detect_voice(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    # Decode Base64 → MP3
    try:
        audio_bytes = base64.b64decode(data.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, sr = load_audio(tmp_path)
        prob = predict_ai(audio)
    finally:
        os.remove(tmp_path)

    return {
        "status": "success",
        "language": detect_language(audio),
        "classification": "AI_GENERATED" if prob >= 0.5 else "HUMAN",
        "confidenceScore": round(prob, 2),
        "explanation": generate_explanation(prob)
    }
