import base64
import io
from pydub import AudioSegment
import librosa
import numpy as np

def base64_to_wav(audio_base64: str):
    audio_base64 = audio_base64.strip()
    audio_base64 = audio_base64.replace("\n", "").replace(" ", "")

    # Fix padding
    missing_padding = len(audio_base64) % 4
    if missing_padding:
        audio_base64 += "=" * (4 - missing_padding)

    audio_bytes = base64.b64decode(audio_base64)
    return io.BytesIO(audio_bytes)

def load_audio(wav_io):
    y, sr = librosa.load(wav_io, sr=16000)
    return y

