"""
1D-conv autoencoder over windowed multivariate time series, migrated from
`~/Projects/FL-bench`'s `src/utils/models.ConvAutoEncoder` (itself a
FL-bench-framework rebuild of `auto_encoder/train_autoencoder.py`'s
standalone `ConvAutoencoder`, with zero dependency on that external repo).
Used by `benchmark/run_ifcaae_baseline.py`'s clustered-FL reconstruction
model -- see that file's module docstring for the algorithm (IFCAAE:
unsupervised, reconstruction-based adaptation of IFCA).

Simplified from the FL-bench version: no `DecoupledModel`/hydra
integration, no `[B, 1, T, F]` 4D convention (this repo's other models --
`fl_model.py`, `gdn_model.py` -- all use `[B, T, N]` directly, matched
here for consistency).
"""
import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    conv_channels = [64, 128, 256]
    kernel_sizes = [3, 3, 3]

    def __init__(self, num_nodes: int, window_size: int):
        super().__init__()
        encoder_layers = []
        in_channels = num_nodes
        for i, (out_channels, kernel_size) in enumerate(zip(self.conv_channels, self.kernel_sizes)):
            encoder_layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2) if i < len(self.conv_channels) - 1 else nn.AdaptiveMaxPool1d(8),
            ]
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        reversed_channels = self.conv_channels[::-1]
        decode_out_channels = reversed_channels[1:] + [num_nodes]
        decoder_layers = []
        for i, (out_channels, kernel_size) in enumerate(zip(decode_out_channels, self.kernel_sizes[::-1])):
            if i < len(reversed_channels) - 1:
                decoder_layers += [
                    nn.ConvTranspose1d(reversed_channels[i], out_channels, kernel_size,
                                        stride=2, padding=kernel_size // 2, output_padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            else:
                decoder_layers.append(
                    nn.ConvTranspose1d(reversed_channels[i], out_channels, kernel_size, padding=kernel_size // 2)
                )
        self.decoder = nn.Sequential(*decoder_layers)
        self.size_adjuster = nn.AdaptiveAvgPool1d(window_size)

    def forward(self, x):  # x: [B, T, N] -> recon [B, T, N]
        sequence = x.transpose(1, 2)  # [B, N, T]
        encoded = self.encoder(sequence)
        decoded = self.decoder(encoded)
        decoded = self.size_adjuster(decoded)
        return decoded.transpose(1, 2)  # [B, T, N]
