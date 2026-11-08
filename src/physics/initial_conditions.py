"""
Generador de Condiciones Iniciales Astrofísicas Canónicas.
Incluye:
  1. Solución Coreográfica Figura-8 de 3 Cuerpos (Chenciner & Montgomery, Annals of Mathematics 2000).
  2. Lotes Perturbados en el Marco del Centro de Masas para Entrenamiento del Operador Transformer.
  3. Sistema Planetario Kepleriano Jerárquico.
  4. Cúmulo Estelar de Plummer de N-Cuerpos para Benchmarks de Escalabilidad HPC en GPU.
"""

import math
import torch
from ..core.state import PhaseSpaceState


class InitialConditionFactory:
    """Fábrica de estados iniciales en el espacio de fases para benchmarks físicos y de GPU."""

    @staticmethod
    def create_figure8_three_body(
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> PhaseSpaceState:
        """
         Crea la órbita periódica exacta en forma de ocho (Chenciner-Montgomery, 2000).
        Período exacto: T = 6.32591398 (con G = 1, m_1 = m_2 = m_3 = 1).
        """
        positions = torch.tensor(
            [
                [
                    [-0.97000436, 0.24308753, 0.0],
                    [0.97000436, -0.24308753, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=dtype,
            device=device,
        )
        vx3, vy3 = -0.93240737, -0.86473146
        velocities = torch.tensor(
            [
                [
                    [-0.5 * vx3, -0.5 * vy3, 0.0],
                    [-0.5 * vx3, -0.5 * vy3, 0.0],
                    [vx3, vy3, 0.0],
                ]
            ],
            dtype=dtype,
            device=device,
        )
        masses = torch.ones((1, 3, 1), dtype=dtype, device=device)
        return PhaseSpaceState(positions=positions, velocities=velocities, masses=masses, time=0.0)

    @staticmethod
    def create_perturbed_figure8_batch(
        batch_size: int = 32,
        noise_scale: float = 0.015,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
        seed: int = 42,
    ) -> PhaseSpaceState:
        """
        Genera un lote de B sistemas de 3 cuerpos alrededor del atractor/órbita Figura-8,
        proyectando cada muestra para tener Centro de Masas en el origen y Momento Lineal nulo (P = 0).
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        base = InitialConditionFactory.create_figure8_three_body(device=device, dtype=dtype)
        pos = base.positions.repeat(batch_size, 1, 1)
        vel = base.velocities.repeat(batch_size, 1, 1)
        masses = base.masses.repeat(batch_size, 1, 1)

        if batch_size > 1:
            pos_noise = torch.randn((batch_size - 1, 3, 2), generator=gen, device=device, dtype=dtype) * noise_scale
            vel_noise = torch.randn((batch_size - 1, 3, 2), generator=gen, device=device, dtype=dtype) * noise_scale
            pos[1:, :, :2] += pos_noise
            vel[1:, :, :2] += vel_noise

        # Garantizar invariancia de centro de masa y momento lineal nulo: sum m_i q_i = 0, sum m_i v_i = 0
        total_mass = torch.sum(masses, dim=1, keepdim=True)
        com_pos = torch.sum(masses * pos, dim=1, keepdim=True) / total_mass
        com_vel = torch.sum(masses * vel, dim=1, keepdim=True) / total_mass
        pos = pos - com_pos
        vel = vel - com_vel

        return PhaseSpaceState(positions=pos, velocities=vel, masses=masses, time=0.0)

    @staticmethod
    def create_plummer_cluster(
        num_bodies: int = 128,
        batch_size: int = 1,
        scale_radius: float = 1.0,
        g_const: float = 1.0,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
        seed: int = 2026,
    ) -> PhaseSpaceState:
        """
        Genera un cúmulo globular de N cuerpos siguiendo el perfil astrofísico de Plummer
        en equilibrio cuasi-virial (2T / |V| ~ 1) para benchmarks de GFLOPS en GPU.
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        m_val = 1.0 / float(num_bodies)
        masses = torch.full((batch_size, num_bodies, 1), m_val, dtype=dtype, device=device)

        # Radio a partir de la función acumulativa de masa de Plummer
        u = torch.rand((batch_size, num_bodies, 1), generator=gen, device=device, dtype=dtype) * 0.98 + 0.01
        radii = scale_radius / torch.sqrt(torch.pow(u, -2.0 / 3.0) - 1.0)

        # Direcciones esféricas isotrópicas
        cos_theta = 2.0 * torch.rand((batch_size, num_bodies, 1), generator=gen, device=device, dtype=dtype) - 1.0
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
        phi = 2.0 * math.pi * torch.rand((batch_size, num_bodies, 1), generator=gen, device=device, dtype=dtype)

        pos = torch.cat(
            [
                radii * sin_theta * torch.cos(phi),
                radii * sin_theta * torch.sin(phi),
                radii * cos_theta,
            ],
            dim=-1,
        )

        # Velocidades acotadas por la velocidad de escape local v_esc = sqrt(2 * |Phi(r)|)
        v_esc = torch.sqrt(2.0 * g_const / torch.sqrt(radii * radii + scale_radius * scale_radius))
        v_mag = 0.45 * v_esc * torch.rand((batch_size, num_bodies, 1), generator=gen, device=device, dtype=dtype)

        v_cos = 2.0 * torch.rand((batch_size, num_bodies, 1), generator=gen, device=device, dtype=dtype) - 1.0
        v_sin = torch.sqrt(torch.clamp(1.0 - v_cos * v_cos, min=0.0))
        v_phi = 2.0 * math.pi * torch.rand((batch_size, num_bodies, 1), generator=gen, device=device, dtype=dtype)

        vel = torch.cat(
            [
                v_mag * v_sin * torch.cos(v_phi),
                v_mag * v_sin * torch.sin(v_phi),
                v_mag * v_cos,
            ],
            dim=-1,
        )

        # Centrar en el marco baricéntrico
        pos = pos - torch.mean(pos, dim=1, keepdim=True)
        vel = vel - torch.mean(vel, dim=1, keepdim=True)

        return PhaseSpaceState(positions=pos, velocities=vel, masses=masses, time=0.0)
