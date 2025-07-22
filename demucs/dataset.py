# dataset.py
import os
import torchaudio
import torch
from torch.utils.data import Dataset
import random

class MUSDBDataset(Dataset):
    def __init__(self, root="data/musdb18/train", segment=10, sample_rate=44100, instruments=["vocals", "drums", "bass", "other"], config=None, split='train'):
        self.segment = config["segment"] if config else segment
        self.sample_rate = config["sample_rate"] if config else sample_rate
        self.instruments = config["target_instruments"] if config else instruments

        self.root = os.path.join(root if root else f"data/musdb18/{split}")
        self.songs = [
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ]
        self.segment_samples = self.segment * self.sample_rate

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        song_path = os.path.join(self.root, self.songs[idx])
        mix, _ = torchaudio.load(os.path.join(song_path, "mixture.wav"))
        mix = mix.mean(dim=0, keepdim=True)  # [1, samples] ← chuyển từ stereo [2, samples] → mono
        sources = {}
        for inst in self.instruments:
            sources[inst], _ = torchaudio.load(os.path.join(song_path, f"{inst}.wav"))
            sources[inst] = sources[inst].mean(dim=0, keepdim=True)  # convert to mono

        # Cut a random segment of N seconds
        total_len = mix.shape[1]
        if total_len > self.segment_samples:
            start = random.randint(0, total_len - self.segment_samples)
            mix = mix[:, start:start + self.segment_samples]
            for inst in self.instruments:
                sources[inst] = sources[inst][:, start:start + self.segment_samples]
        else:
            # Pad if too short
            pad = self.segment_samples - total_len
            mix = torch.nn.functional.pad(mix, (0, pad))
            for inst in self.instruments:
                sources[inst] = torch.nn.functional.pad(sources[inst], (0, pad))

        return mix, sources
