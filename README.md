# AI-Generated Voice Detection API

This project implements a REST API that determines whether a given voice sample is AI-generated or human-generated.
The system supports Tamil, English, Hindi, Malayalam, and Telugu automatically using language detection.

## Features
- Accepts Base64 encoded MP3 audio
- Automatically detects spoken language
- Classifies voice as AI_GENERATED or HUMAN
- Returns confidence score and explanation
- Secure API with API key
- Real-time inference using pretrained models

## Tech Stack
- Python
- FastAPI
- PyTorch
- Wav2Vec2 (feature extraction)
- Whisper (language detection)
- Librosa, Pydub

## API Endpoint
POST /api/voice-detection

Headers:
x-api-key: sk_test_123456789  
Content-Type: application/json

Request Body:
{
  "audioFormat": "mp3",
  "audioBase64": "..."
}

Response:
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.73,
  "explanation": "Strong synthetic voice patterns detected"
}

## Run Locally
pip install -r requirements.txt  
python -m uvicorn app:app --reload  

Open demo_ui.html in browser to test.

## Notes
- No external detection APIs are used
- Works offline after model download
- Designed for hackathon evaluation
