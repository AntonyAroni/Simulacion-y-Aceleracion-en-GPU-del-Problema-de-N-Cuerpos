"""
Auditor de Invariantes Físicos y Leyes de Conservación (Hamiltoniano, Momento Lineal y Angular).
Cumple con la exigencia de rigor numérico del Sílabo UNSA (Unidades I y II).
"""

from typing import Dict
import torch
from ..core.interfaces import IForceKernel, IPhysicsAuditor
from ..core.state import PhaseSpaceState


class HamiltonianAuditor(IPhysicsAuditor):
    """
    Evalúa las cantidades conservadas del sistema dinámico de N-cuerpos:
      1. Energía Cinética T(v)
      2. Energía Potencial V(q)
      3. Hamiltoniano Total H(q, p) = T + V
      4. Momento Lineal Total P = sum m_i v_i
      5. Momento Angular Total L = sum q_i x (m_i v_i)
      6. Razón del Teorema del Virial: 2T / |V|
    """

    def __init__(self, force_kernel: IForceKernel) -> None:
        self.force_kernel = force_kernel

    def compute_kinetic_energy(self, velocities: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Calcula T = 0.5 * sum_i m_i ||v_i||^2 para cada lote [B]."""
        vel_sq = torch.sum(velocities * velocities, dim=-1, keepdim=True)  # [B, N, 1]
        return 0.5 * torch.sum(masses * vel_sq, dim=(1, 2))  # [B]

    def compute_linear_momentum(self, velocities: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Calcula el vector momento lineal total P [B, 3]."""
        return torch.sum(masses * velocities, dim=1)

    def compute_angular_momentum(
        self, positions: torch.Tensor, velocities: torch.Tensor, masses: torch.Tensor
    ) -> torch.Tensor:
        """Calcula el vector momento angular total L = sum q_i x p_i [B, 3]."""
        momenta = masses * velocities  # [B, N, 3]
        angular = torch.linalg.cross(positions, momenta, dim=-1)  # [B, N, 3]
        return torch.sum(angular, dim=1)  # [B, 3]

    def evaluate_invariants(self, state: PhaseSpaceState) -> Dict[str, torch.Tensor]:
        """Evalúa todos los invariantes geométricos y físicos del estado actual."""
        kinetic = self.compute_kinetic_energy(state.velocities, state.masses)
        potential = self.force_kernel.compute_potential_energy(state.positions, state.masses)
        total_energy = kinetic + potential
        linear_mom = self.compute_linear_momentum(state.velocities, state.masses)
        angular_mom = self.compute_angular_momentum(state.positions, state.velocities, state.masses)
        angular_mag = torch.linalg.norm(angular_mom, dim=-1)
        virial_ratio = (2.0 * kinetic) / (torch.abs(potential) + 1e-12)

        return {
            "kinetic": kinetic,
            "potential": potential,
            "total_energy": total_energy,
            "linear_momentum": linear_mom,
            "angular_momentum": angular_mom,
            "angular_momentum_norm": angular_mag,
            "virial_ratio": virial_ratio,
        }

    @staticmethod
    def compute_relative_drift(current_val: torch.Tensor, initial_val: torch.Tensor) -> torch.Tensor:
        """Calcula la deriva relativa |(X(t) - X(0)) / X(0)|."""
        return torch.abs(current_val - initial_val) / (torch.abs(initial_val) + 1e-15)
