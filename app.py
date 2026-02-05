import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from audio_utils import base64_to_wav, load_audio
from model import predict_ai, generate_explanation, detect_language

# ================================
# API KEY (from Render / cloud env)
# ================================
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set")

# ================================
# FastAPI App
# ================================
app = FastAPI()

# ================================
# CORS (safe for hackathon testing)
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# Health / Root Endpoint
# ================================
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Voice AI Detector API is live"
    }

# ================================
# Request Model
# ================================
class VoiceRequest(BaseModel):
    audioFormat: str
    audioBase64: str

    class Config:
        extra = "ignore"  # allows judge UI extra fields (like Language)

# ================================
# Voice Detection Endpoint
# (supports both / and no /)
# ================================
@app.post("/api/voice-detection")
@app.post("/api/voice-detection/")
def detect_voice(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # ---- API Key Validation ----
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ---- Format Validation ----
    if data.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    try:
        # ---- Decode & Load Audio ----
        wav_io = base64_to_wav(data.audioBase64)
        audio = load_audio(wav_io)

        # ---- Inference ----
        language = detect_language(audio)
        prob_ai = predict_ai(audio)

        classification = "AI_GENERATED" if prob_ai >= 0.5 else "HUMAN"

        return {
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": round(prob_ai, 2),
            "explanation": generate_explanation(prob_ai)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
