import io
import logging
import os
import tempfile
import wave
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
from fastapi import Body, FastAPI, File, HTTPException, Path, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, VitsModel, set_seed

logger = logging.getLogger("mms_tts")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DEFAULT_MODEL_ID = "facebook/mms-tts-amh"
DEFAULT_VOICE_ID = "amh-default"
DEFAULT_STT_MODEL_ID = "b1n1yam/shook-medium-amharic-2k"
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


def _load_audio(path: str) -> Tuple[torch.Tensor, int]:
    try:
        speech_array, sampling_rate = torchaudio.load(path)
        if speech_array.ndim > 1:
            speech_array = speech_array.mean(dim=0)
        return speech_array, int(sampling_rate)
    except Exception as exc:
        try:
            import soundfile as sf  # type: ignore
        except Exception as import_exc:
            raise RuntimeError(
                "Audio decode failed. Install `torchcodec` or `soundfile` to enable decoding."
            ) from import_exc
        data, sampling_rate = sf.read(path, always_2d=True)
        data = data.mean(axis=1)
        return torch.from_numpy(data), int(sampling_rate)


@lru_cache(maxsize=1)
def _load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


@lru_cache(maxsize=1)
def _load_stt_model(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return processor, model, device


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/voices")
def voices():
    return {"voices": VOICE_CATALOG}


@app.post("/v1/speech-to-text")
async def speech_to_text(
    audio: UploadFile = File(..., description="Audio file (wav, mp3, flac, etc.)"),
    model_id: str = Query(DEFAULT_STT_MODEL_ID, description="Speech-to-text model id"),
):
    processor, model, device = _load_stt_model(model_id)

    raw = await audio.read()
    if not raw:
        return {"text": "", "model_id": model_id}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename or "")[1] or ".wav")
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()
        
        try:
            speech_tensor, sampling_rate = _load_audio(tmp.name)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        speech_array = speech_tensor.numpy()

        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": text, "model_id": model_id}
    finally:
        if os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except OSError:
                logger.warning("Failed to delete temp audio file: %s", tmp.name)


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
            "stt": "/v1/speech-to-text",
            "health": "/health",
        },
        "notes": "This is an ElevenLabs-style TTS endpoint backed by facebook/mms-tts-amh.",
    }
