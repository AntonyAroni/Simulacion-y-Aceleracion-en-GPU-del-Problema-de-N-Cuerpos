"""
Métodos de Runge-Kutta Explícitos: RK2 (Punto Medio) y RK4 Clásico (Temas 08 y 10 del Sílabo UNSA).
"""

from ..core.interfaces import IForceKernel, IIntegrator
from ..core.state import PhaseSpaceState


class RK2Integrator(IIntegrator):
    """Método de Runge-Kutta de 2.º Orden (Punto Medio Explícito)."""

    def __init__(self, force_kernel: IForceKernel) -> None:
        self.force_kernel = force_kernel

    @property
    def name(self) -> str:
        return "Runge-Kutta 2 (RK2)"

    @property
    def order(self) -> int:
        return 2

    @property
    def is_symplectic(self) -> bool:
        return False

    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        q0, v0, m = state.positions, state.velocities, state.masses

        a1 = self.force_kernel.compute_acceleration(q0, m)
        q_mid = q0 + 0.5 * dt * v0
        v_mid = v0 + 0.5 * dt * a1

        a2 = self.force_kernel.compute_acceleration(q_mid, m)
        q_next = q0 + dt * v_mid
        v_next = v0 + dt * a2

        return PhaseSpaceState(positions=q_next, velocities=v_next, masses=m, time=state.time + dt)


class RK4Integrator(IIntegrator):
    """
    Método Clásico de Runge-Kutta de 4.º Orden (RK4).
    Error de truncamiento local O(dt^5) y global O(dt^4).
    Requiere 4 evaluaciones del kernel de fuerzas por paso temporal.
    No es simpléctico: presenta deriva secular de energía en horizontes orbitales largos.
    """

    def __init__(self, force_kernel: IForceKernel) -> None:
        self.force_kernel = force_kernel

    @property
    def name(self) -> str:
        return "Runge-Kutta 4 (RK4)"

    @property
    def order(self) -> int:
        return 4

    @property
    def is_symplectic(self) -> bool:
        return False

    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        q0, v0, m = state.positions, state.velocities, state.masses

        # Etapa k1
        k1_q = v0
        k1_v = self.force_kernel.compute_acceleration(q0, m)

        # Etapa k2
        q2 = q0 + 0.5 * dt * k1_q
        k2_q = v0 + 0.5 * dt * k1_v
        k2_v = self.force_kernel.compute_acceleration(q2, m)

        # Etapa k3
        q3 = q0 + 0.5 * dt * k2_q
        k3_q = v0 + 0.5 * dt * k2_v
        k3_v = self.force_kernel.compute_acceleration(q3, m)

        # Etapa k4
        q4 = q0 + dt * k3_q
        k4_q = v0 + dt * k3_v
        k4_v = self.force_kernel.compute_acceleration(q4, m)

        q_next = q0 + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        v_next = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        return PhaseSpaceState(positions=q_next, velocities=v_next, masses=m, time=state.time + dt)
