import io
import logging
import os
import wave
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from fastapi import Body, FastAPI, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, VitsModel, set_seed

logger = logging.getLogger("mms_tts")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DEFAULT_MODEL_ID = "facebook/mms-tts-amh"
DEFAULT_VOICE_ID = "amh-default"
VOICE_CATALOG = [
    {
        "voice_id": "amh-default",
        "name": "Amharic Default",
        "model_id": DEFAULT_MODEL_ID,
        "language_code": "amh",
        "description": "Single-speaker MMS Amharic voice.",
    },
    {
        "voice_id": "eng-default",
        "name": "English Default",
        "model_id": "facebook/mms-tts-eng",
        "language_code": "eng",
        "description": "Single-speaker MMS English voice.",
    },
]


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_id: Optional[str] = None
    language_code: Optional[str] = None
    voice_settings: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None


app = FastAPI(title="Amharic TTS Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _uromanize(text: str) -> str:
    """
    MMS-TTS Amharic checkpoint expects Latin (uroman) input.
    If uroman is unavailable, return text unchanged and log a warning.
    """
    try:
        from uroman import uroman  # type: ignore

        return uroman.uroman(text)
    except Exception:
        logger.warning("uroman not available; using raw text input")
        return text


def _wave_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    if waveform.ndim > 1:
        waveform = waveform.squeeze(0)
    audio = waveform.detach().cpu().numpy()
    audio = (audio * 32767.0).clip(-32768, 32767).astype("int16")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def _chunk_bytes(data: bytes, chunk_size: int):
    for idx in range(0, len(data), chunk_size):
        yield data[idx : idx + chunk_size]


@lru_cache(maxsize=1)
def _load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/voices")
def voices():
    return {"voices": VOICE_CATALOG}


@app.post("/v1/text-to-speech/{voice_id}")
def text_to_speech(
    voice_id: str = Path(..., description="Voice identifier"),
    output_format: str = Query("wav", description="Audio output format"),
    req: TTSRequest = Body(...),
):
    if voice_id != DEFAULT_VOICE_ID:
        logger.info("Unknown voice_id '%s'; using default voice", voice_id)

    selected = next((v for v in VOICE_CATALOG if v["voice_id"] == voice_id), None)
    model_id = req.model_id or (selected["model_id"] if selected else DEFAULT_MODEL_ID)
    tokenizer, model, device = _load_model(model_id)

    if req.seed is not None:
        set_seed(req.seed)

    text = _uromanize(req.text)
    inputs = tokenizer(text, return_tensors="pt")
    logger.warning("Original input",req.text)
    logger.warning("\n Romanized Text", text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs).waveform

    audio_bytes = _wave_bytes(output, model.config.sampling_rate)

    if output_format not in ("wav", "wav_22050", "wav_16000"):
        logger.info("Unsupported output_format '%s'; defaulting to wav", output_format)

    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/v1/text-to-speech/{voice_id}/stream")
def text_to_speech_stream(
    voice_id: str = Path(..., description="Voice identifier"),
    output_format: str = Query("wav", description="Audio output format"),
    chunk_ms: int = Query(250, ge=20, le=2000, description="Chunk size in milliseconds"),
    req: TTSRequest = Body(...),
):
    if voice_id != DEFAULT_VOICE_ID:
        logger.info("Unknown voice_id '%s'; using default voice", voice_id)

    selected = next((v for v in VOICE_CATALOG if v["voice_id"] == voice_id), None)
    model_id = req.model_id or (selected["model_id"] if selected else DEFAULT_MODEL_ID)
    tokenizer, model, device = _load_model(model_id)

    if req.seed is not None:
        set_seed(req.seed)

    text = _uromanize(req.text)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs).waveform

    audio_bytes = _wave_bytes(output, model.config.sampling_rate)

    if output_format not in ("wav", "wav_22050", "wav_16000"):
        logger.info("Unsupported output_format '%s'; defaulting to wav", output_format)

    bytes_per_second = model.config.sampling_rate * 2
    chunk_size = max(1024, int(bytes_per_second * (chunk_ms / 1000.0)))
    return StreamingResponse(
        _chunk_bytes(audio_bytes, chunk_size),
        media_type="audio/wav",
    )


@app.get("/")
def root():
    return {
        "name": "Amharic TTS Backend",
        "model": DEFAULT_MODEL_ID,
        "voice_id": DEFAULT_VOICE_ID,
        "endpoints": {
            "tts": "/v1/text-to-speech/{voice_id}",
            "tts_stream": "/v1/text-to-speech/{voice_id}/stream",
            "health": "/health",
        },
        "notes": "This is an ElevenLabs-style TTS endpoint backed by facebook/mms-tts-amh.",
    }
