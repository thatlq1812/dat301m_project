# utils.py (refined for consistent normalization and SDR evaluation)
import os
import torch
import soundfile as sf
import torchaudio
from museval.metrics import bss_eval_sources

def load_audio(path, sr=44100):
    waveform, _ = torchaudio.load(path)
    return waveform

def save_audio(path, waveform, sr=44100):
    # Normalize to [-1, 1] for consistent loudness
    waveform = waveform / (waveform.abs().max() + 1e-6)
    sf.write(path, waveform.squeeze().numpy().T, sr)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def normalize_rms(audio, target_rms=0.1):
    """
    Normalize audio signal to a fixed RMS value for consistent loudness.
    """
    rms = audio.pow(2).mean().sqrt()
    return audio * (target_rms / (rms + 1e-6))

def compute_sdr(pred, target):
    """
    Compute SDR between predicted and target signals.
    Both signals are normalized to target RMS before evaluation.
    pred, target: torch.Tensor with shape [1, samples]
    """
    pred = normalize_rms(pred).squeeze().cpu().numpy()
    target = normalize_rms(target).squeeze().cpu().numpy()

    sdr, _, _, _ = bss_eval_sources(target, pred)
    return float(sdr.mean())

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
