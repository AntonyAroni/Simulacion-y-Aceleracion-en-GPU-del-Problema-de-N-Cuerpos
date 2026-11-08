"""
Bloque de Autoatención Guiada por la Física con Sesgo Relacional Gravitatorio.
Referencias:
  - Varghese, Zhang, Karniadakis (2024/2025), "SympGNNs: Symplectic Graph Neural Networks"
  - Wu et al. (ICML 2024), "Transolver: A Fast Transformer Solver for PDEs"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import WaveletFourierActivation


class PhysicsGuidedAttentionBlock(nn.Module):
    """
    Bloque Transformer Equivariante que modula los pesos de autoatención entre partículas
    mediante el potencial gravitacional inverso de la distancia:
        Bias_{ij} = MLP( [ 1 / sqrt(||q_i - q_j||^2 + eps^2), ||q_i - q_j||, m_i * m_j ] )
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_gravity_bias: bool = True,
        softening: float = 1e-4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible por n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_gravity_bias = use_gravity_bias
        self.softening_sq = softening ** 2

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if self.use_gravity_bias:
            self.gravity_bias_net = nn.Sequential(
                nn.Linear(3, n_heads * 2),
                nn.SiLU(),
                nn.Linear(n_heads * 2, n_heads),
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveletFourierActivation(d_ff),
            nn.Linear(d_ff, d_model),
        )

    def _compute_gravity_bias(self, latest_pos: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """
        Calcula el tensor de sesgo relacional gravitatorio [B, n_heads, N, N].
        """
        diff = latest_pos.unsqueeze(1) - latest_pos.unsqueeze(2)  # [B, N, N, 3]
        dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True) + self.softening_sq  # [B, N, N, 1]
        inv_dist = torch.rsqrt(dist_sq)
        dist = torch.sqrt(dist_sq)
        mass_prod = torch.bmm(masses, masses.transpose(1, 2)).unsqueeze(-1)  # [B, N, N, 1]

        pair_feats = torch.cat([inv_dist, dist, mass_prod], dim=-1)  # [B, N, N, 3]
        bias = self.gravity_bias_net(pair_feats)  # [B, N, N, n_heads]
        return bias.permute(0, 3, 1, 2)  # [B, n_heads, N, N]

    def forward(
        self,
        tokens: torch.Tensor,
        latest_pos: torch.Tensor,
        masses: torch.Tensor,
    ) -> torch.Tensor:
        b_size, n_bodies, _ = tokens.shape
        normed = self.norm1(tokens)

        qkv = self.qkv_proj(normed).reshape(b_size, n_bodies, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
        if self.use_gravity_bias:
            scores = scores + self._compute_gravity_bias(latest_pos, masses)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # [B, H, N, D_h]
        attn_out = attn_out.transpose(1, 2).reshape(b_size, n_bodies, self.d_model)

        tokens = tokens + self.out_proj(attn_out)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens
