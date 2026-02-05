import base64
import os
import subprocess
import tempfile
import librosa
import numpy as np


TARGET_SR = 16000


class AudioError(Exception):
    pass


def load_audio(path, return_base64=False):
    """
    If return_base64=True  -> returns base64 MP3 (transport)
    If return_base64=False -> returns (audio_array, sample_rate)
    """

    if not os.path.exists(path):
        raise AudioError("File not found")

    # ---- Base64 ONLY (no decoding) ----
    if return_base64:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ---- Inference path ----
    tmp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(TARGET_SR), tmp_wav],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    audio, sr = librosa.load(tmp_wav, sr=TARGET_SR, mono=True)
    os.remove(tmp_wav)

    if audio is None or len(audio) == 0:
        raise AudioError("Invalid audio")

    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    return audio.astype(np.float32), sr
