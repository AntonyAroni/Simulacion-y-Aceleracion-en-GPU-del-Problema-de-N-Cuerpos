"""
Función de Pérdida Física Auto-Adaptativa (Homoscedastic Uncertainty / Self-Adaptive PINN Loss).
Referencias:
  - Kendall et al. (2018), Multi-Task Learning Using Uncertainty to Weigh Losses
  - McClenny & Braga-Neto (2023), Self-Adaptive Physics-Informed Neural Networks
  - Wang et al. (NeurIPS 2025), Gradient Alignment in Physics-Informed Neural Networks
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..physics.invariants import HamiltonianAuditor


class SelfAdaptivePhysicsLoss(nn.Module):
    """
    Combina cuatro objetivos físicos diferenciables:
      1. Error de Posición en el espacio de configuración: L_pos = MSE(q_pred, q_gt)
      2. Error de Velocidad / Momento canónico: L_vel = MSE(v_pred, v_gt)
      3. Conservación del Hamiltoniano: L_H = MSE(H(q_pred, p_pred), H(q_gt, p_gt))
      4. Conservación del Momento Angular: L_L = MSE(L(q_pred, p_pred), L(q_gt, p_gt))

    Los pesos relativos lambda_k = exp(-s_k) se aprenden automáticamente durante el entrenamiento
    para equilibrar escalas y evitar conflictos de gradiente.
    """

    def __init__(self, auditor: HamiltonianAuditor, use_self_adaptive: bool = True) -> None:
        super().__init__()
        self.auditor = auditor
        self.use_self_adaptive = use_self_adaptive

        # Log-varianzas aprendibles s_k = log(sigma_k^2) inicializadas para favorecer precisión física
        self.log_vars = nn.Parameter(torch.tensor([0.0, 0.0, 0.5, 0.5], dtype=torch.float64))

    def forward(
        self,
        pred_pos: torch.Tensor,
        pred_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        masses: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_pos = F.mse_loss(pred_pos, target_pos)
        loss_vel = F.mse_loss(pred_vel, target_vel)

        # Invariantes físicos predichos vs ground-truth
        kin_pred = self.auditor.compute_kinetic_energy(pred_vel, masses)
        pot_pred = self.auditor.force_kernel.compute_potential_energy(pred_pos, masses)
        h_pred = kin_pred + pot_pred

        kin_gt = self.auditor.compute_kinetic_energy(target_vel, masses)
        pot_gt = self.auditor.force_kernel.compute_potential_energy(target_pos, masses)
        h_gt = kin_gt + pot_gt

        loss_hamiltonian = F.mse_loss(h_pred, h_gt)

        ang_pred = self.auditor.compute_angular_momentum(pred_pos, pred_vel, masses)
        ang_gt = self.auditor.compute_angular_momentum(target_pos, target_vel, masses)
        loss_angular = F.mse_loss(ang_pred, ang_gt)

        raw_losses = torch.stack([loss_pos, loss_vel, loss_hamiltonian, loss_angular])

        if self.use_self_adaptive:
            # Limitar log_vars en rango estable [-6, 4]
            clamped_s = torch.clamp(self.log_vars.to(raw_losses.dtype), min=-6.0, max=4.0)
            precision = torch.exp(-clamped_s)
            total_loss = torch.sum(0.5 * precision * raw_losses + 0.5 * clamped_s)
        else:
            weights = torch.tensor([1.0, 1.0, 0.25, 0.25], device=raw_losses.device, dtype=raw_losses.dtype)
            total_loss = torch.sum(weights * raw_losses)

        return {
            "total": total_loss,
            "pos": loss_pos.detach(),
            "vel": loss_vel.detach(),
            "hamiltonian": loss_hamiltonian.detach(),
            "angular": loss_angular.detach(),
            "adaptive_weights": torch.exp(-torch.clamp(self.log_vars.detach(), -6.0, 4.0)),
        }
