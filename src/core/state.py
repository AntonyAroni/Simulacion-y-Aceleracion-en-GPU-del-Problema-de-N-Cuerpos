"""
Contenedor del Estado en el Espacio de Fases (PhaseSpaceState).
Soporta ejecución individual [N, D] y por lotes paralelos [B, N, D] en GPU.
"""

from dataclasses import dataclass
import torch


@dataclass
class PhaseSpaceState:
    """
    Representa el estado físico instantáneo de un sistema de N-cuerpos.

    Atributos:
        positions (torch.Tensor): Coordenadas espaciales q de forma [B, N, 3] o [N, 3].
        velocities (torch.Tensor): Velocidades v de forma [B, N, 3] o [N, 3].
        masses (torch.Tensor): Masas m de forma [B, N, 1] o [N, 1].
        time (float): Tiempo físico acumulado t.
    """
    positions: torch.Tensor
    velocities: torch.Tensor
    masses: torch.Tensor
    time: float = 0.0

    def __post_init__(self) -> None:
        if self.positions.ndim == 2:
            self.positions = self.positions.unsqueeze(0)
        if self.velocities.ndim == 2:
            self.velocities = self.velocities.unsqueeze(0)
        if self.masses.ndim == 1:
            self.masses = self.masses.unsqueeze(0).unsqueeze(-1)
        elif self.masses.ndim == 2:
            self.masses = self.masses.unsqueeze(-1)

    @property
    def batch_size(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_bodies(self) -> int:
        return int(self.positions.shape[1])

    @property
    def spatial_dim(self) -> int:
        return int(self.positions.shape[2])

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.positions.dtype

    @property
    def momenta(self) -> torch.Tensor:
        """Momento canónico conjugado p_i = m_i * v_i."""
        return self.masses * self.velocities

    def clone(self) -> "PhaseSpaceState":
        """Crea una copia profunda en memoria de GPU/CPU."""
        return PhaseSpaceState(
            positions=self.positions.clone(),
            velocities=self.velocities.clone(),
            masses=self.masses.clone(),
            time=float(self.time),
        )

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> "PhaseSpaceState":
        """Transfiere el estado al dispositivo y precisión especificados."""
        target_dtype = dtype if dtype is not None else self.dtype
        return PhaseSpaceState(
            positions=self.positions.to(device=device, dtype=target_dtype),
            velocities=self.velocities.to(device=device, dtype=target_dtype),
            masses=self.masses.to(device=device, dtype=target_dtype),
            time=float(self.time),
        )
