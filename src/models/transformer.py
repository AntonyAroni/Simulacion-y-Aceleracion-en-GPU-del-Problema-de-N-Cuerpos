"""
Operador Neural Basado en Transformer con Estructura Simpléctica (SymplecticTransformerOperator)
y Envoltorio Liskov-Compatible (SurrogateTransformerIntegrator).
"""

import torch
import torch.nn as nn
from ..core.config import TransformerConfig
from ..core.interfaces import IForceKernel, IIntegrator, ISurrogateModel
from ..core.state import PhaseSpaceState
from .embeddings import PhaseSpaceTokenizer, WaveletFourierActivation
from .attention import PhysicsGuidedAttentionBlock
from .projection import SymplecticManifoldProjection


class SymplecticTransformerOperator(nn.Module, ISurrogateModel):
    """
    Arquitectura Híbrida Transformer para Dinámica Gravitacional de N-Cuerpos.
    Combina:
      1. Tokenización del espacio de fases con características espectrales Wavelet-Fourier.
      2. Capas de autoatención guiadas por el potencial gravitacional par-a-par.
      3. Decodificador acoplado tipo Leapfrog/Verlet en el espacio latente:
         v_{t+M} = v_t + Delta_T * (a(q_t) + phi_v(Tokens))
         q_{t+M} = q_t + 0.5 * Delta_T * (v_t + v_{t+M}) + Delta_T * phi_q(Tokens)
      4. Proyección diferenciable sobre la variedad de conservación (Momento Lineal y Hamiltoniano).
    """

    def __init__(
        self,
        config: TransformerConfig,
        force_kernel: IForceKernel,
        macro_dt: float = 0.02,
    ) -> None:
        super().__init__()
        self.config = config
        self.force_kernel = force_kernel
        self.macro_dt = float(macro_dt)

        self.tokenizer = PhaseSpaceTokenizer(
            seq_history=config.seq_history,
            d_model=config.d_model,
        )

        self.blocks = nn.ModuleList(
            [
                PhysicsGuidedAttentionBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    use_gravity_bias=config.use_gravity_bias,
                )
                for _ in range(config.n_layers)
            ]
        )

        self.vel_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            WaveletFourierActivation(config.d_model // 2),
            nn.Linear(config.d_model // 2, 3),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(config.d_model + 3, config.d_model // 2),
            WaveletFourierActivation(config.d_model // 2),
            nn.Linear(config.d_model // 2, 3),
        )

        # Inicialización de baja varianza en las capas de salida (estilo DiT / ControlNet)
        nn.init.normal_(self.vel_head[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.vel_head[-1].bias)
        nn.init.normal_(self.pos_head[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.pos_head[-1].bias)

        self.manifold_proj = SymplecticManifoldProjection(force_kernel=force_kernel)

    def forward(
        self,
        history_positions: torch.Tensor,
        history_velocities: torch.Tensor,
        masses: torch.Tensor,
        reference_energy: torch.Tensor | None = None,
        enforce_energy_manifold: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        history_positions: [B, S, N, 3]
        history_velocities: [B, S, N, 3]
        masses: [B, N, 1]
        """
        q_last = history_positions[:, -1]  # [B, N, 3]
        v_last = history_velocities[:, -1]  # [B, N, 3]

        # Aceleración física instantánea como ancla inductiva (Physics-Informed Anchor)
        acc_last = self.force_kernel.compute_acceleration(q_last, masses)

        tokens = self.tokenizer(history_positions, history_velocities, masses)
        for block in self.blocks:
            tokens = block(tokens, q_last, masses)

        # Predicción acoplada tipo Verlet + corrección neuronal de orden superior
        delta_v_neural = self.vel_head(tokens)
        v_pred = v_last + self.macro_dt * (acc_last + delta_v_neural)

        pos_input = torch.cat([tokens, v_pred], dim=-1)
        delta_q_neural = self.pos_head(pos_input)
        q_pred = q_last + 0.5 * self.macro_dt * (v_last + v_pred) + self.macro_dt * delta_q_neural

        if self.config.use_manifold_projection:
            q_pred, v_pred = self.manifold_proj(
                pred_pos=q_pred,
                pred_vel=v_pred,
                masses=masses,
                reference_energy=reference_energy,
                enforce_energy_manifold=enforce_energy_manifold,
            )

        return q_pred, v_pred

    def predict_macro_step(
        self,
        history_positions: torch.Tensor,
        history_velocities: torch.Tensor,
        masses: torch.Tensor,
        reference_energy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            history_positions=history_positions,
            history_velocities=history_velocities,
            masses=masses,
            reference_energy=reference_energy,
            enforce_energy_manifold=reference_energy is not None,
        )


class SurrogateTransformerIntegrator(IIntegrator):
    """
    Adaptador que permite usar el SymplecticTransformerOperator bajo el mismo contrato IIntegrator
    que RK4 y Yoshida (Principio de Sustitución de Liskov - LSP).
    Utiliza un integrador numérico semilla para inicializar las primeras (S-1) muestras de historia
    y posteriormente avanza mediante saltos macro-temporales Delta_T = M * dt en una sola inferencia.
    """

    def __init__(
        self,
        model: SymplecticTransformerOperator,
        warmup_integrator: IIntegrator,
        auditor,
        macro_step_jump: int = 10,
    ) -> None:
        self.model = model
        self.warmup_integrator = warmup_integrator
        self.auditor = auditor
        self.macro_step_jump = macro_step_jump
        self.pos_buffer: list[torch.Tensor] = []
        self.vel_buffer: list[torch.Tensor] = []
        self.reference_energy: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return f"Symplectic Transformer (Salto {self.macro_step_jump}x)"

    @property
    def order(self) -> int:
        return 4

    @property
    def is_symplectic(self) -> bool:
        return True

    def reset_history(self, initial_state: PhaseSpaceState) -> None:
        self.pos_buffer = [initial_state.positions.clone()]
        self.vel_buffer = [initial_state.velocities.clone()]
        inv = self.auditor.evaluate_invariants(initial_state)
        self.reference_energy = inv["total_energy"].detach()

    @torch.no_grad()
    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        """
        Ejecuta un salto macro-temporal completo de M * dt cuando la ventana histórica está llena.
        """
        seq_len = self.model.config.seq_history
        if not self.pos_buffer:
            self.reset_history(state)

        # Llenar ventana inicial S si aún no está completa
        current = state
        while len(self.pos_buffer) < seq_len:
            for _ in range(self.macro_step_jump):
                current = self.warmup_integrator.step(current, dt)
            self.pos_buffer.append(current.positions.clone())
            self.vel_buffer.append(current.velocities.clone())

        hist_q = torch.stack(self.pos_buffer[-seq_len:], dim=1)  # [B, S, N, 3]
        hist_v = torch.stack(self.vel_buffer[-seq_len:], dim=1)  # [B, S, N, 3]

        q_next, v_next = self.model.predict_macro_step(
            history_positions=hist_q,
            history_velocities=hist_v,
            masses=state.masses,
            reference_energy=self.reference_energy,
        )

        # Actualizar ventana deslizante con el nuevo estado macro-temporal
        self.pos_buffer.append(q_next.clone())
        self.vel_buffer.append(v_next.clone())
        if len(self.pos_buffer) > seq_len:
            self.pos_buffer.pop(0)
            self.vel_buffer.pop(0)

        macro_dt = dt * self.macro_step_jump
        return PhaseSpaceState(
            positions=q_next,
            velocities=v_next,
            masses=state.masses,
            time=state.time + macro_dt,
        )
