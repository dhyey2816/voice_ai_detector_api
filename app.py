import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from audio_utils import base64_to_wav, load_audio
from model import predict_ai, generate_explanation, detect_language

API_KEY = os.getenv("API_KEY")  # ← IMPORTANT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceRequest(BaseModel):
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
def detect_voice(data: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    wav_io = base64_to_wav(data.audioBase64)
    audio = load_audio(wav_io)

    language = detect_language(audio)
    prob_ai = predict_ai(audio)

    return {
        "status": "success",
        "language": language,
        "classification": "AI_GENERATED" if prob_ai >= 0.5 else "HUMAN",
        "confidenceScore": round(prob_ai, 2),
        "explanation": generate_explanation(prob_ai)
    }
