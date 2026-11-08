"""
Kernel Vectorizado de Interacción Gravitatoria Directa O(N^2) en GPU.
Implementa IForceKernel con doble ruta de optimización HPC:
  1. Ruta Exacta de Diferencias Finitas Par-a-Par (N <= 32): Precisión FP64 máxima sin cancelación.
  2. Ruta GEMM / Tensor-Core (N > 32): Reformula la ley de gravitación universal como multiplicaciones
     matriciales por lotes (torch.bmm) eliminando el tensor intermedio 4D [B, N, N, 3]:
         a_i = sum_j W_{ij} q_j - (sum_j W_{ij}) q_i
     reduciendo en 3x la huella de VRAM y explotando las unidades matriciales de la RTX 4050.
"""

import torch
from ..core.interfaces import IForceKernel


class DirectGravityKernel(IForceKernel):
    """
    Calcula aceleraciones y energía potencial gravitacional para N cuerpos en paralelo:
        a_i = G * sum_{j != i} m_j * (q_j - q_i) / (||q_j - q_i||^2 + eps^2)^(3/2)
    """

    def __init__(self, g_const: float = 1.0, softening: float = 1e-4, gemm_threshold: int = 32) -> None:
        self.g_const = float(g_const)
        self.softening_sq = float(softening) ** 2
        self.gemm_threshold = gemm_threshold
        self.total_force_calls: int = 0

    def compute_acceleration(self, positions: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """
        Calcula el tensor de aceleraciones [B, N, 3] en GPU.
        """
        self.total_force_calls += 1
        n_bodies = positions.shape[1]

        if n_bodies <= self.gemm_threshold:
            # Ruta exacta para pocos cuerpos (órbita coreográfica Figura-8 en FP64)
            diff = positions.unsqueeze(1) - positions.unsqueeze(2)  # [B, N, N, 3]
            dist_sq = torch.sum(diff * diff, dim=-1) + self.softening_sq  # [B, N, N]
            diag_mask = ~torch.eye(n_bodies, dtype=torch.bool, device=positions.device).unsqueeze(0)
            inv_dist_cubed = torch.where(diag_mask, torch.rsqrt(dist_sq) / dist_sq, torch.zeros_like(dist_sq))
            weights = self.g_const * masses.transpose(1, 2) * inv_dist_cubed  # [B, N, N]
            return torch.sum(diff * weights.unsqueeze(-1), dim=2)

        # Ruta HPC GEMM / Tensor-Core para cúmulos masivos (N > 32): Cero tensores 4D
        pos_norm_sq = torch.sum(positions * positions, dim=-1, keepdim=True)  # [B, N, 1]
        dist_sq = pos_norm_sq + pos_norm_sq.transpose(1, 2) - 2.0 * torch.bmm(positions, positions.transpose(1, 2))
        dist_sq = torch.clamp(dist_sq, min=0.0) + self.softening_sq

        inv_dist = torch.rsqrt(dist_sq)
        inv_dist_cubed = inv_dist / dist_sq
        # Anular diagonal principal i == j
        inv_dist_cubed.diagonal(dim1=-2, dim2=-1).zero_()

        # Matriz de pesos gravitacionales W_{ij} = G * m_j / r_{ij}^3 -> [B, N, N]
        weights = self.g_const * inv_dist_cubed * masses.transpose(1, 2)

        # Identidad algebraica: a_i = (W @ Q)_i - (sum_j W_{ij}) * q_i
        wq = torch.bmm(weights, positions)  # [B, N, 3]
        row_sum = torch.sum(weights, dim=-1, keepdim=True)  # [B, N, 1]
        return wq - row_sum * positions

    def compute_potential_energy(self, positions: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """
        Calcula la energía potencial gravitacional exacta V(q) para cada sistema del lote [B].
        """
        n_bodies = positions.shape[1]
        if n_bodies <= self.gemm_threshold:
            diff = positions.unsqueeze(1) - positions.unsqueeze(2)
            dist_sq = torch.sum(diff * diff, dim=-1) + self.softening_sq
        else:
            pos_norm_sq = torch.sum(positions * positions, dim=-1, keepdim=True)
            dist_sq = pos_norm_sq + pos_norm_sq.transpose(1, 2) - 2.0 * torch.bmm(positions, positions.transpose(1, 2))
            dist_sq = torch.clamp(dist_sq, min=0.0) + self.softening_sq

        diag_mask = ~torch.eye(n_bodies, dtype=torch.bool, device=positions.device).unsqueeze(0)
        inv_dist = torch.where(diag_mask, torch.rsqrt(dist_sq), torch.zeros_like(dist_sq))
        mass_matrix = torch.bmm(masses, masses.transpose(1, 2))
        return -0.5 * self.g_const * torch.sum(mass_matrix * inv_dist, dim=(1, 2))

    def estimate_flops(self, batch_size: int, num_bodies: int, num_calls: int | None = None) -> float:
        calls = num_calls if num_calls is not None else self.total_force_calls
        return float(calls) * float(batch_size) * (float(num_bodies) ** 2) * 20.0
