# evaluate.py
import os
import torch
from utils import load_audio, save_audio, compute_sdr
from model import Demucs

# === Setting ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
song_name = "Angels In Amplifiers - I'm Alright"  # <-- Đặt tên track cần test
song_path = f"data/musdb18/test/{song_name}"
segment_duration = 10  # seconds
sample_rate = 44100
segment_samples = segment_duration * sample_rate

# === Find newest model checkpoint in the checkpoints directory ===
model_dir = "checkpoints"
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
if not model_files:
    raise FileNotFoundError("No model checkpoints found in the 'checkpoints' directory.")
model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
model_path = os.path.join(model_dir, model_files[0])
print(f"Using model: {model_path}")

# === Concatenate audio segments ===
def segment_audio(audio, segment_samples=441000):
    segments = []
    total_len = audio.shape[-1]
    for start in range(0, total_len, segment_samples):
        end = min(start + segment_samples, total_len)
        segment = audio[..., start:end]
        if segment.shape[-1] < segment_samples:
            pad = segment_samples - segment.shape[-1]
            segment = torch.nn.functional.pad(segment, (0, pad))
        segments.append(segment)
    return segments

# === RMS nomeralization ===
def normalize_energy(audio, target_rms=0.1):
    rms = audio.pow(2).mean().sqrt()
    return audio * (target_rms / (rms + 1e-6))

# === Load model ===
model = Demucs()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# === Load mixture ===
mix = load_audio(os.path.join(song_path, "mixture.wav"))  # [1, samples]
mix = mix.mean(dim=0, keepdim=True)  # Mono: [1, samples]
segments = segment_audio(mix, segment_samples)
print(f"Segmented input into {len(segments)} chunks")

# === Predict outputs ===
all_outputs = {k: [] for k in model.instruments}
with torch.no_grad():
    for seg in segments:
        seg = seg.unsqueeze(0).to(device)  # [1, 1, samples]
        est = model(seg)
        for k in est:
            all_outputs[k].append(est[k].squeeze(0).cpu())

# === Merge outputs ===
estimates = {k: torch.cat(v, dim=-1) for k, v in all_outputs.items()}

# === Normalize and save outputs ===
output_dir = f"outputs/{song_name.replace(' ', '_')}"
os.makedirs(output_dir, exist_ok=True)
print("\n[Output RMS energy per source]")

for name, audio in estimates.items():
    audio = normalize_energy(audio)
    save_audio(os.path.join(output_dir, f"{name}.wav"), audio)
    rms = audio.pow(2).mean().sqrt().item()
    print(f"[{name:>7}] RMS Energy: {rms:.6f}")

# === Load ground-truth and calculate SDR ===
gt = {}
for inst in ["vocals", "drums", "bass", "other"]:
    path = os.path.join(song_path, f"{inst}.wav")
    if os.path.exists(path):
        audio = load_audio(path).mean(dim=0, keepdim=True)  # Mono
        gt[inst] = audio
    else:
        print(f"Warning: Missing ground truth for '{inst}', skipping...")

# === Compare predictions with ground truth ===
sdr_scores = {}
for k in gt:
    if k in estimates:
        pred = estimates[k]
        target = gt[k]

        # Cut to the same length
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        sdr = compute_sdr(pred, target)
        sdr_scores[k] = sdr

# === SDR Result ===
print("\nSDR Scores per instrument:")
for k, v in sdr_scores.items():
    print(f"{k:>8}: {v:.2f} dB")

if sdr_scores:
    avg_sdr = sum(sdr_scores.values()) / len(sdr_scores)
    print(f"\nAverage SDR: {avg_sdr:.2f} dB")
