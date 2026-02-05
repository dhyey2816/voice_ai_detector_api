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

# ✅ ADD THIS BLOCK HERE
@app.on_event("startup")
def startup_event():
    from model import load_models_once
    load_models_once()

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
    ...
