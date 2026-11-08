"""
Integradores de Primer Orden: Euler Explícito y Euler-Cromer (Tema 07 del Sílabo UNSA).
"""

from ..core.interfaces import IForceKernel, IIntegrator
from ..core.state import PhaseSpaceState


class EulerIntegrator(IIntegrator):
    """Método de Euler explícito estándar (Orden 1, No Simpléctico)."""

    def __init__(self, force_kernel: IForceKernel) -> None:
        self.force_kernel = force_kernel

    @property
    def name(self) -> str:
        return "Euler Explícito"

    @property
    def order(self) -> int:
        return 1

    @property
    def is_symplectic(self) -> bool:
        return False

    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        acc = self.force_kernel.compute_acceleration(state.positions, state.masses)
        new_pos = state.positions + dt * state.velocities
        new_vel = state.velocities + dt * acc
        return PhaseSpaceState(
            positions=new_pos,
            velocities=new_vel,
            masses=state.masses,
            time=state.time + dt,
        )


class EulerCromerIntegrator(IIntegrator):
    """Método de Euler-Cromer semi-implícito (Orden 1, Simpléctico de 1.º orden)."""

    def __init__(self, force_kernel: IForceKernel) -> None:
        self.force_kernel = force_kernel

    @property
    def name(self) -> str:
        return "Euler-Cromer"

    @property
    def order(self) -> int:
        return 1

    @property
    def is_symplectic(self) -> bool:
        return True

    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        acc = self.force_kernel.compute_acceleration(state.positions, state.masses)
        new_vel = state.velocities + dt * acc
        new_pos = state.positions + dt * new_vel
        return PhaseSpaceState(
            positions=new_pos,
            velocities=new_vel,
            masses=state.masses,
            time=state.time + dt,
        )
