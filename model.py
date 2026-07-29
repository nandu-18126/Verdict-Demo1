"""
AASIST loading + inference + SHAP explainability.

Ported directly from the VERDICT pilot notebook (cells 14-19 for loading/
inference, cell 27 for the SHAP explainability method). Logic is kept as
close to the original as possible so results are consistent with what's
reported in the paper.

NOTE ON CALIBRATION (judgment call — please verify after deploying):
The notebook calibrated REAL_INDEX empirically from a batch of known
real/fake clips. A live single-upload demo has no such batch on every
request, so this module calibrates ONCE at startup using two tiny bundled
reference clips (data/calib_real.wav, data/calib_fake.wav) and caches the
result. If you don't supply those files, it falls back to REAL_INDEX = 1
(the common ASVspoof/AASIST convention: 0 = spoof, 1 = bonafide) — but
you should confirm this is correct for your checkpoint the first time you
test the deployed app with a known-real recording. If the verdict comes
out backwards, call POST /calibrate with a known-real clip to fix it live.
"""

import json
import os
import sys
import threading

import numpy as np
import torch
import torchaudio

AASIST_DIR = os.environ.get("AASIST_DIR", "/app/aasist_repo")
CKPT_PATH = os.path.join(AASIST_DIR, "models", "weights", "AASIST.pth")
CONFIG_PATH = os.path.join(AASIST_DIR, "config", "AASIST.conf")

CUT = 64600  # ~4.04s at 16kHz — AASIST's fixed input length
N_CHUNKS = 20  # temporal chunks used for the SHAP masking scheme

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_lock = threading.Lock()
_model = None
_real_index = 1  # default convention; overwritten by calibration if available


def load_model():
    """Load AASIST once at process startup."""
    global _model
    if _model is not None:
        return _model

    if AASIST_DIR not in sys.path:
        sys.path.insert(0, AASIST_DIR)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model_config = config["model_config"]

    from models.AASIST import Model as AASISTModel  # noqa: import after sys.path insert

    model = AASISTModel(model_config).to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return _model


def pad_or_trim(wav: torch.Tensor, cut: int = CUT) -> torch.Tensor:
    n = wav.shape[0]
    if n >= cut:
        return wav[:cut]
    reps = int(cut / n) + 1
    return wav.repeat(reps)[:cut]


def load_wav_16k_mono(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.mean(dim=0)


@torch.no_grad()
def get_scores(wav_1d: torch.Tensor) -> np.ndarray:
    """Run AASIST on a 1D 16kHz mono waveform tensor. Returns softmax probs, shape (2,)."""
    model = load_model()
    wav = pad_or_trim(wav_1d).unsqueeze(0).to(DEVICE)
    out = model(wav)
    logits = out[1] if isinstance(out, (tuple, list)) else out
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs


@torch.no_grad()
def get_scores_from_path(path: str) -> np.ndarray:
    return get_scores(load_wav_16k_mono(path))


def confidence_real(wav_1d: torch.Tensor) -> float:
    return float(get_scores(wav_1d)[_real_index])


def get_real_index() -> int:
    return _real_index


def calibrate_from_pair(real_wav: torch.Tensor, fake_wav: torch.Tensor) -> int:
    """Same logic as the notebook: whichever output index scores higher on
    the known-real clip is 'real'. Called at startup with bundled samples
    (if present) and can be re-called via POST /calibrate."""
    global _real_index
    with _lock:
        probs_real = get_scores(real_wav)
        idx = 0 if probs_real[0] > probs_real[1] else 1
        _real_index = idx
    return idx


def try_startup_calibration(data_dir: str):
    real_path = os.path.join(data_dir, "calib_real.wav")
    fake_path = os.path.join(data_dir, "calib_fake.wav")
    if os.path.exists(real_path):
        real_wav = load_wav_16k_mono(real_path)
        fake_wav = load_wav_16k_mono(fake_path) if os.path.exists(fake_path) else None
        idx = calibrate_from_pair(real_wav, fake_wav if fake_wav is not None else real_wav)
        return idx
    return None


# ---------------- SHAP-based explainability (ported from cell 27) ----------------

def make_predict_fn(base_wav: torch.Tensor):
    chunk_len = len(base_wav) // N_CHUNKS
    model = load_model()

    def predict(mask_batch):
        outs = []
        for mask in mask_batch:
            wav = base_wav.clone()
            for i, keep in enumerate(mask):
                if keep == 0:
                    start = i * chunk_len
                    end = start + chunk_len if i < N_CHUNKS - 1 else len(wav)
                    wav[start:end] = 0.0
            with torch.no_grad():
                inp = wav.unsqueeze(0).to(DEVICE)
                out = model(inp)
                logits = out[1] if isinstance(out, (tuple, list)) else out
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            outs.append(probs)
        return np.array(outs)

    return predict


def explain_clip(wav_1d: torch.Tensor, real_index: int, nsamples: int = 100):
    """Returns (explainability_score_E, top_chunk_index, attribution_array)."""
    import shap  # local import: slow to import, only needed for this call

    wav = pad_or_trim(wav_1d)
    predict_fn = make_predict_fn(wav)
    background = np.zeros((1, N_CHUNKS))
    explainer = shap.KernelExplainer(predict_fn, background)
    all_ones = np.ones((1, N_CHUNKS))
    shap_values = explainer.shap_values(all_ones, nsamples=nsamples, silent=True)

    if isinstance(shap_values, list):
        attribution = np.abs(shap_values[real_index]).flatten()
    else:
        attribution = np.abs(shap_values[:, real_index]).flatten()

    p = attribution / (attribution.sum() + 1e-8)
    p = np.clip(p, 1e-12, 1)
    H = -np.sum(p * np.log(p))
    H_norm = H / np.log(N_CHUNKS)
    E = 100 * (1 - H_norm)
    top_chunk = int(np.argmax(attribution))
    return float(E), top_chunk, attribution
