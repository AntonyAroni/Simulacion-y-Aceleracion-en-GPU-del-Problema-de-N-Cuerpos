"""
Integradores Simplécticos Geométricos: Velocity Verlet (Orden 2) y Yoshida (Orden 4).
Referencia Fundamental:
  H. Yoshida (1990), "Construction of higher order symplectic integrators",
  Physics Letters A, Vol. 150, Issues 5-7, pp. 262-268.
"""

from ..core.interfaces import IForceKernel, IIntegrator
from ..core.state import PhaseSpaceState


class VelocityVerletIntegrator(IIntegrator):
    """Integrador Simpléctico Velocity-Verlet / Leapfrog de 2.º Orden."""

    def __init__(self, force_kernel: IForceKernel) -> None:
        self.force_kernel = force_kernel

    @property
    def name(self) -> str:
        return "Velocity Verlet (Orden 2)"

    @property
    def order(self) -> int:
        return 2

    @property
    def is_symplectic(self) -> bool:
        return True

    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        q0, v0, m = state.positions, state.velocities, state.masses
        a0 = self.force_kernel.compute_acceleration(q0, m)
        v_half = v0 + 0.5 * dt * a0
        q_next = q0 + dt * v_half
        a_next = self.force_kernel.compute_acceleration(q_next, m)
        v_next = v_half + 0.5 * dt * a_next

        return PhaseSpaceState(positions=q_next, velocities=v_next, masses=m, time=state.time + dt)


class Yoshida4Integrator(IIntegrator):
    """
    Integrador Simpléctico de 4.º Orden de Yoshida (1990).
    Compone tres pasos simétricos de Leapfrog mediante el método de Baker-Campbell-Hausdorff (BCH)
    para anular los conmutadores de error de orden 2 y 3.
    Ventaja HPC crucial: Solo requiere 3 evaluaciones del kernel de fuerzas por paso (frente a 4 en RK4),
    preservando exactamente el volumen en el espacio de fases (Teorema de Liouville).
    """

    def __init__(self, force_kernel: IForceKernel) -> None:
        self.force_kernel = force_kernel
        cbrt_2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt_2)
        w0 = -cbrt_2 * w1

        # Coeficientes de deriva (Drift: posiciones) y patada (Kick: velocidades)
        self.c1 = w1 / 2.0
        self.c2 = (w0 + w1) / 2.0
        self.c3 = (w0 + w1) / 2.0
        self.c4 = w1 / 2.0

        self.d1 = w1
        self.d2 = w0
        self.d3 = w1

    @property
    def name(self) -> str:
        return "Yoshida Simpléctico 4.º Orden"

    @property
    def order(self) -> int:
        return 4

    @property
    def is_symplectic(self) -> bool:
        return True

    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        q, v, m = state.positions, state.velocities, state.masses

        # Sub-paso 1
        q = q + self.c1 * dt * v
        a1 = self.force_kernel.compute_acceleration(q, m)
        v = v + self.d1 * dt * a1

        # Sub-paso 2
        q = q + self.c2 * dt * v
        a2 = self.force_kernel.compute_acceleration(q, m)
        v = v + self.d2 * dt * a2

        # Sub-paso 3
        q = q + self.c3 * dt * v
        a3 = self.force_kernel.compute_acceleration(q, m)
        v = v + self.d3 * dt * a3

        # Sub-paso 4 (Cierre simétrico sin evaluación adicional de fuerza)
        q = q + self.c4 * dt * v

        return PhaseSpaceState(positions=q, velocities=v, masses=m, time=state.time + dt)
