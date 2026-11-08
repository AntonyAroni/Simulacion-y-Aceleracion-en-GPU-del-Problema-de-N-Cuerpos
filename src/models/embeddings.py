"""
Tokenización del Espacio de Fases y Activación Espectral Wavelet-Fourier.
Referencias:
  - Zhao et al. (2023), "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks"
  - Brantner et al. (2024), "Structure-Preserving Transformers for Learning Parametrized Hamiltonian Systems"
"""

import math
import torch
import torch.nn as nn


class WaveletFourierActivation(nn.Module):
    """
    Activación Wavelet-Fourier aprendible introducida en PINNsFormer (Zhao et al., 2023):
        sigma(x) = w_1 * sin(x) + w_2 * cos(x) + w_3 * SiLU(x)
    Evita el sesgo espectral (spectral bias) de los MLPs estándar en oscilaciones no lineales.
    """

    def __init__(self, features: int) -> None:
        super().__init__()
        self.w_sin = nn.Parameter(torch.full((1, 1, features), 0.35))
        self.w_cos = nn.Parameter(torch.full((1, 1, features), 0.35))
        self.w_silu = nn.Parameter(torch.full((1, 1, features), 0.50))
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_sin * torch.sin(x) + self.w_cos * torch.cos(x) + self.w_silu * self.silu(x)


class PhaseSpaceTokenizer(nn.Module):
    """
    Proyecta una ventana histórica de estados del espacio de fases:
      history_positions: [B, S, N, 3]
      history_velocities: [B, S, N, 3]
      masses: [B, N, 1]
    en tokens latentes por partícula [B, N, d_model] enriquecidos con características de Fourier.
    """

    def __init__(self, seq_history: int, d_model: int) -> None:
        super().__init__()
        self.seq_history = seq_history
        self.d_model = d_model

        # Por cada cuerpo tenemos: S * (q_x, q_y, q_z, v_x, v_y, v_z) + masa = 6*S + 1
        in_dim = 6 * seq_history + 1
        fourier_dim = d_model // 2

        self.linear_in = nn.Linear(in_dim, fourier_dim)
        self.proj = nn.Linear(in_dim + 2 * fourier_dim, d_model)
        self.activation = WaveletFourierActivation(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        history_positions: torch.Tensor,
        history_velocities: torch.Tensor,
        masses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retorna los tokens latentes de las N partículas: [B, N, d_model].
        """
        # Reordenar de [B, S, N, 3] a [B, N, S, 3] y aplanar el eje temporal S
        b_size, _, n_bodies, _ = history_positions.shape
        pos_flat = history_positions.permute(0, 2, 1, 3).reshape(b_size, n_bodies, -1)
        vel_flat = history_velocities.permute(0, 2, 1, 3).reshape(b_size, n_bodies, -1)

        raw_features = torch.cat([pos_flat, vel_flat, masses], dim=-1)  # [B, N, 6*S + 1]
        freq_proj = math.pi * self.linear_in(raw_features)
        fourier_features = torch.cat([raw_features, torch.sin(freq_proj), torch.cos(freq_proj)], dim=-1)

        tokens = self.proj(fourier_features)
        tokens = self.activation(tokens)
        return self.norm(tokens)
