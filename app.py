import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from audio_utils import base64_to_wav, load_audio
from model import predict_ai, load_models

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

# 🔥 LOAD MODELS AT STARTUP
@app.on_event("startup")
def startup_event():
    load_models()   # blocks until models are loaded


class VoiceRequest(BaseModel):
    audioFormat: str
    audioBase64: str

    class Config:
        extra = "ignore"


@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    wav_io = base64_to_wav(data.audioBase64)
    audio = load_audio(wav_io)

    prob_ai = predict_ai(audio)

    return {
        "status": "success",
        "classification": "AI_GENERATED" if prob_ai >= 0.5 else "HUMAN",
        "confidenceScore": round(prob_ai, 2)
    }
