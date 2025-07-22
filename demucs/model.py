# model.py
import torch
import torch.nn as nn

class Demucs(nn.Module):
    def __init__(self, hidden=64, target_instruments=["vocals", "drums", "bass", "other"]):
        super().__init__()
        self.instruments = target_instruments

        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1 if i == 0 else hidden * (2 ** (i - 1)), hidden * (2 ** i), kernel_size=8, stride=4, padding=2),
                nn.BatchNorm1d(hidden * (2 ** i)),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in range(4)
        ])

        # LSTM bottleneck
        self.bottleneck_lstm = nn.LSTM(
            input_size=hidden * (2 ** 3),  # deepest encoder output
            hidden_size=hidden * (2 ** 3),
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.bottleneck_linear = nn.Linear(hidden * (2 ** 3) * 2, hidden * (2 ** 3))  # reduce from Bi-LSTM back to match decoder

        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    hidden * (2 ** (i + 1)),  # input channels
                    hidden * (2 ** i),        # output channels
                    kernel_size=8, stride=4, padding=2),
                nn.BatchNorm1d(hidden * (2 ** i)),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in reversed(range(3))
        ])

        # Output heads per instrument
        self.output_layers = nn.ModuleDict({
            inst: nn.ConvTranspose1d(hidden, 1, kernel_size=8, stride=4, padding=2)
            for inst in self.instruments
        })

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)

        # LSTM bottleneck
        z = skip_connections.pop()  # deepest feature map [B, C, T]
        z = z.permute(0, 2, 1)  # [B, T, C] for LSTM
        z, _ = self.bottleneck_lstm(z)
        z = self.bottleneck_linear(z)
        z = z.permute(0, 2, 1)  # [B, C, T] back

        # Decoder with skip connections
        for i, dec in enumerate(self.decoder):
            z = dec(z)
            skip = skip_connections.pop()
            if z.shape[-1] > skip.shape[-1]:
                z = z[..., :skip.shape[-1]]  # crop if needed
            elif skip.shape[-1] > z.shape[-1]:
                skip = skip[..., :z.shape[-1]]
            z = z + skip  # skip connection

        # Output heads per instrument
        out = {}
        for inst in self.instruments:
            out[inst] = self.output_layers[inst](z)

        return out