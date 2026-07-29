import os
import tempfile
import traceback

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import model as m
import scoring as s

DATA_DIR = os.environ.get("VERDICT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))

app = FastAPI(title="VERDICT Audio Deepfake Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    m.load_model()
    calibrated = m.try_startup_calibration(DATA_DIR)
    if calibrated is None:
        print(f"No calibration sample found in {DATA_DIR} — using default REAL_INDEX="
              f"{m.get_real_index()}. Test with a known-real clip and call /calibrate if it's backwards.")
    else:
        print(f"Startup calibration complete — REAL_INDEX = {calibrated}")


def _load_uploaded_wav(raw_bytes: bytes) -> torch.Tensor:
    """Decode arbitrary uploaded audio bytes to a 16kHz mono waveform tensor."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        wav, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.mean(dim=0)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(400, "Empty file")

    try:
        wav = _load_uploaded_wav(raw_bytes)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio file: {e}")

    duration_sec = round(len(wav) / 16000, 2)
    real_index = m.get_real_index()

    try:
        # Stage 2 — classification
        probs = m.get_scores(wav)
        confidence_real = float(probs[real_index])
        verdict = "real" if confidence_real >= 0.5 else "synthetic"

        # Stage 3 — explainability (SHAP)
        E, top_chunk, attribution = m.explain_clip(wav, real_index, nsamples=40)
        chunk_start = round(top_chunk * (duration_sec / m.N_CHUNKS if duration_sec > 0 else 0.2), 2)
        chunk_end = round((top_chunk + 1) * (duration_sec / m.N_CHUNKS if duration_sec > 0 else 0.2), 2)

        # Stage 4 — Robustness (R) and Legal-Alignment (L)
        robustness = s.robustness_score(wav)
        legal = s.legal_alignment_score(wav, raw_bytes)

        return {
            "duration_sec": duration_sec,
            "verdict": verdict,
            "confidence_real": round(confidence_real, 4),
            "real_index_used": real_index,
            "scores": {
                "robustness_R": robustness["mean_robustness_R"],
                "explainability_E": round(E, 2),
                "legal_alignment_L": legal["legal_alignment_score_L"],
            },
            "explainability_detail": {
                "top_contributing_chunk": top_chunk,
                "top_chunk_time_window_sec": [chunk_start, chunk_end],
                "note": (
                    "This shows WHERE in the clip the model's decision concentrated, not WHY "
                    "that region looks suspicious — the paper's deeper explanation (comparing "
                    "against a paired real recording) needs a known-genuine reference clip, "
                    "which isn't available for an arbitrary upload."
                ),
            },
            "robustness_detail": robustness["per_snr_level"],
            "legal_alignment_detail": legal,
            "disclaimer": (
                "Research pilot output, not a certified forensic or legal determination. "
                "See the VERDICT paper for methodology and known limitations."
            ),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {e}")


@app.post("/calibrate")
async def calibrate(real_file: UploadFile = File(...)):
    """Fix REAL_INDEX live using a clip you know is genuine speech —
    use this once after deploying if verdicts come out backwards."""
    raw_bytes = await real_file.read()
    wav = _load_uploaded_wav(raw_bytes)
    idx = m.calibrate_from_pair(wav, wav)
    return {"real_index_now": idx}


@app.get("/health")
def health():
    return {"status": "ok", "device": m.DEVICE, "real_index": m.get_real_index()}


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
