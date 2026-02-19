import base64
import io
import logging
import os
import tempfile
import wave
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch
from fastapi import Body, FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from TTS.api import TTS

logger = logging.getLogger("xtts")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DEFAULT_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "en"


class XTTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = DEFAULT_LANGUAGE
    speaker: Optional[str] = None
    speaker_wav_base64: Optional[str] = None
    split_sentences: bool = True


app = FastAPI(title="XTTS v2 Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _wave_bytes(waveform, sample_rate: int) -> bytes:
    if isinstance(waveform, torch.Tensor):
        audio = waveform.detach().cpu().numpy()
    else:
        audio = waveform
    if audio.ndim > 1:
        audio = audio.squeeze(0)
    audio = (audio * 32767.0).clip(-32768, 32767).astype("int16")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


@lru_cache(maxsize=1)
def _load_tts():
    use_gpu = torch.cuda.is_available()
    tts = TTS(DEFAULT_MODEL_ID, gpu=use_gpu)
    return tts


def _get_sample_rate(tts: TTS) -> int:
    if hasattr(tts, "synthesizer") and hasattr(tts.synthesizer, "output_sample_rate"):
        return int(tts.synthesizer.output_sample_rate)
    return 24000


def _list_speakers(tts: TTS) -> List[str]:
    speakers = []
    if hasattr(tts, "speakers") and tts.speakers:
        speakers = list(tts.speakers)
    return speakers


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/voices")
def voices():
    tts = _load_tts()
    speakers = _list_speakers(tts)
    return {
        "model_id": DEFAULT_MODEL_ID,
        "language": DEFAULT_LANGUAGE,
        "speakers": speakers,
    }


@app.post("/v1/text-to-speech/{voice_id}")
def text_to_speech(
    voice_id: str = Path(..., description="Voice identifier"),
    req: XTTSRequest = Body(...),
):
    tts = _load_tts()
    sample_rate = _get_sample_rate(tts)

    speaker_wav_path = None
    if req.speaker_wav_base64:
        try:
            raw = base64.b64decode(req.speaker_wav_base64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 audio") from exc
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        speaker_wav_path = tmp.name

    try:
        waveform = tts.tts(
            text=req.text,
            language=req.language,
            speaker=req.speaker,
            speaker_wav=[speaker_wav_path] if speaker_wav_path else None,
            split_sentences=req.split_sentences,
        )
    finally:
        if speaker_wav_path and os.path.exists(speaker_wav_path):
            try:
                os.unlink(speaker_wav_path)
            except OSError:
                logger.warning("Failed to delete temp speaker wav: %s", speaker_wav_path)

    audio_bytes = _wave_bytes(waveform, sample_rate)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.get("/")
def root():
    return {
        "name": "XTTS v2 Backend",
        "model": DEFAULT_MODEL_ID,
        "endpoints": {
            "tts": "/v1/text-to-speech/{voice_id}",
            "voices": "/v1/voices",
            "health": "/health",
        },
        "notes": "Voice cloning requires speaker_wav_base64 and language.",
    }
