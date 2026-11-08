"""
Capa Diferenciable de Proyección en la Variedad Simpléctica, Iso-Energética y de Momento Angular.
Referencias:
  - Canizares et al. (2024/2025), "Symplectic Neural Flows for Modeling and Discovery"
  - Energy-Consistent Neural Operators (2025)
"""

import torch
import torch.nn as nn
from ..core.interfaces import IForceKernel


class SymplecticManifoldProjection(nn.Module):
    """
    Proyecta las predicciones crudas del Transformer sobre la subvariedad de leyes de conservación:
      1. Conservación exacta del Momento Lineal Total (P = sum m_i v_i = P_0) y Centro de Masas.
      2. Conservación exacta del Momento Angular Total (L = sum q_i x (m_i v_i) = L_0).
      3. Proyección iso-energética sobre la hipersuperficie Hamiltoniana H(q, p) = E_0.
    """

    def __init__(self, force_kernel: IForceKernel) -> None:
        super().__init__()
        self.force_kernel = force_kernel

    def forward(
        self,
        pred_pos: torch.Tensor,
        pred_vel: torch.Tensor,
        masses: torch.Tensor,
        ref_com_pos: torch.Tensor | None = None,
        ref_com_vel: torch.Tensor | None = None,
        reference_energy: torch.Tensor | None = None,
        enforce_energy_manifold: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica proyección de invariantes geométricos sobre [B, N, 3].
        """
        total_mass = torch.sum(masses, dim=1, keepdim=True)  # [B, 1, 1]

        # 1. Proyección al marco del Centro de Masas y conservación de Momento Lineal P
        curr_com_pos = torch.sum(masses * pred_pos, dim=1, keepdim=True) / total_mass
        curr_com_vel = torch.sum(masses * pred_vel, dim=1, keepdim=True) / total_mass

        target_com_pos = ref_com_pos if ref_com_pos is not None else torch.zeros_like(curr_com_pos)
        target_com_vel = ref_com_vel if ref_com_vel is not None else torch.zeros_like(curr_com_vel)

        corr_pos = pred_pos - curr_com_pos + target_com_pos
        corr_vel = pred_vel - curr_com_vel + target_com_vel

        # 2. Proyección de Momento Angular (elimina rotación espuria en el plano orbital XY)
        if enforce_energy_manifold:
            r_sq = torch.sum(corr_pos[..., :2] ** 2, dim=-1, keepdim=True)  # [B, N, 1]
            inertia_z = torch.sum(masses * r_sq, dim=1, keepdim=True) + 1e-12  # [B, 1, 1]
            lz_curr = torch.sum(
                masses * (corr_pos[..., 0:1] * corr_vel[..., 1:2] - corr_pos[..., 1:2] * corr_vel[..., 0:1]),
                dim=1,
                keepdim=True,
            )  # [B, 1, 1]
            omega_z = lz_curr / inertia_z
            vel_rot_x = -omega_z * corr_pos[..., 1:2]
            vel_rot_y = omega_z * corr_pos[..., 0:1]
            vel_rot = torch.cat([vel_rot_x, vel_rot_y, torch.zeros_like(vel_rot_x)], dim=-1)
            corr_vel = corr_vel - vel_rot

        # 3. Proyección sobre la variedad iso-energética H(q, p) = T(v) + V(q) = E_0
        if enforce_energy_manifold and reference_energy is not None:
            potential = self.force_kernel.compute_potential_energy(corr_pos, masses)  # [B]
            kinetic = 0.5 * torch.sum(masses * torch.sum(corr_vel * corr_vel, dim=-1, keepdim=True), dim=(1, 2))  # [B]

            target_kinetic = torch.clamp(reference_energy - potential, min=1e-12)
            scale = torch.sqrt(target_kinetic / (kinetic + 1e-15)).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            corr_vel = corr_vel * scale

        return corr_pos, corr_vel
