import base64
import io
from pydub import AudioSegment
import librosa
import numpy as np

def base64_to_wav(audio_base64: str):
    audio_bytes = base64.b64decode(audio_base64)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def load_audio(wav_io):
    y, sr = librosa.load(wav_io, sr=16000)
    return y
