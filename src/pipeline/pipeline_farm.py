"""
Motor Esquelético Pipeline-Farm sobre CUDA Streams.
Implementa tres etapas desacopladas (Principio de Inversión de Dependencias - DIP):
  1. EmitterStage: Particionado de lotes y preparación de tensores en memoria de dispositivo.
  2. GPUWorkerFarmStage: Pool de trabajadores concurrentes sobre múltiples CUDA Streams.
  3. CollectorStage: Reducción de invariantes físicos (E, P, L) y telemetría de rendimiento (GFLOPS, ms/paso).
"""

from dataclasses import dataclass
import time
from typing import List, Dict, Any
import torch
from ..core.config import PipelineConfig
from ..core.interfaces import IIntegrator, IPhysicsAuditor, IPipelineStage
from ..core.state import PhaseSpaceState


@dataclass
class SimulationResult:
    """Resultado consolidado por la etapa Collector del Pipeline-Farm."""
    integrator_name: str
    is_symplectic: bool
    order: int
    times: torch.Tensor
    positions_history: torch.Tensor
    velocities_history: torch.Tensor
    energy_history: torch.Tensor
    relative_energy_drift: torch.Tensor
    angular_momentum_drift: torch.Tensor
    virial_history: torch.Tensor
    elapsed_ms: float
    steps_executed: int
    gflops: float


class EmitterStage(IPipelineStage):
    """
    Etapa 1 (Emitter): Prepara y clona el estado inicial hacia memoria contigua en GPU
    para cada trabajador de la granja (Farm).
    """

    def execute(self, payload: Dict[str, Any], stream: torch.cuda.Stream | None = None) -> Dict[str, Any]:
        initial_state: PhaseSpaceState = payload["initial_state"]
        if stream is not None and initial_state.device.type == "cuda":
            with torch.cuda.stream(stream):
                cloned = initial_state.clone()
        else:
            cloned = initial_state.clone()
        payload["working_state"] = cloned
        return payload


class GPUWorkerFarmStage(IPipelineStage):
    """
    Etapa 2 (Worker Farm): Ejecuta el bucle de integración temporal o inferencia del Transformer
    dentro de un CUDA Stream dedicado, registrando tiempos exactos de hardware.
    """

    def execute(self, payload: Dict[str, Any], stream: torch.cuda.Stream | None = None) -> Dict[str, Any]:
        integrator: IIntegrator = payload["integrator"]
        state: PhaseSpaceState = payload["working_state"]
        dt: float = payload["dt"]
        num_steps: int = payload["num_steps"]
        record_interval: int = payload["record_interval"]

        pos_list: List[torch.Tensor] = [state.positions.clone()]
        vel_list: List[torch.Tensor] = [state.velocities.clone()]
        time_list: List[float] = [float(state.time)]

        use_cuda_timer = state.device.type == "cuda" and torch.cuda.is_available()

        if use_cuda_timer:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream())
            with ctx:
                start_evt.record()
                for step_idx in range(1, num_steps + 1):
                    state = integrator.step(state, dt)
                    if step_idx % record_interval == 0 or step_idx == num_steps:
                        pos_list.append(state.positions.clone())
                        vel_list.append(state.velocities.clone())
                        time_list.append(float(state.time))
                end_evt.record()
            end_evt.synchronize()
            elapsed_ms = float(start_evt.elapsed_time(end_evt))
        else:
            t0 = time.perf_counter()
            for step_idx in range(1, num_steps + 1):
                state = integrator.step(state, dt)
                if step_idx % record_interval == 0 or step_idx == num_steps:
                    pos_list.append(state.positions.clone())
                    vel_list.append(state.velocities.clone())
                    time_list.append(float(state.time))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        payload["positions_history"] = torch.stack(pos_list, dim=0)  # [T_rec, B, N, 3]
        payload["velocities_history"] = torch.stack(vel_list, dim=0)
        payload["times"] = torch.tensor(time_list, device=state.device, dtype=state.dtype)
        payload["elapsed_ms"] = elapsed_ms
        payload["steps_executed"] = num_steps
        return payload


class CollectorStage(IPipelineStage):
    """
    Etapa 3 (Collector): Audita la trayectoria completa calculando el Hamiltoniano H(t),
    deriva energética relativa |Delta E / E_0|, momento angular |Delta L / L_0| y GFLOPS.
    """

    def __init__(self, auditor: IPhysicsAuditor, flops_per_step_fn) -> None:
        self.auditor = auditor
        self.flops_per_step_fn = flops_per_step_fn

    def execute(self, payload: Dict[str, Any], stream: torch.cuda.Stream | None = None) -> SimulationResult:
        integrator: IIntegrator = payload["integrator"]
        pos_hist: torch.Tensor = payload["positions_history"]  # [T, B, N, 3]
        vel_hist: torch.Tensor = payload["velocities_history"]
        times: torch.Tensor = payload["times"]
        initial_state: PhaseSpaceState = payload["initial_state"]
        elapsed_ms: float = payload["elapsed_ms"]
        steps_executed: int = payload["steps_executed"]

        t_steps, b_size, n_bodies, _ = pos_hist.shape
        energies = []
        ang_norms = []
        virials = []

        for t_idx in range(t_steps):
            snap = PhaseSpaceState(
                positions=pos_hist[t_idx],
                velocities=vel_hist[t_idx],
                masses=initial_state.masses,
                time=float(times[t_idx].item()),
            )
            inv = self.auditor.evaluate_invariants(snap)
            energies.append(inv["total_energy"][0])
            ang_norms.append(inv["angular_momentum_norm"][0])
            virials.append(inv["virial_ratio"][0])

        energy_tensor = torch.stack(energies)
        ang_tensor = torch.stack(ang_norms)
        virial_tensor = torch.stack(virials)

        e0 = energy_tensor[0]
        l0 = ang_tensor[0]
        rel_e_drift = torch.abs(energy_tensor - e0) / (torch.abs(e0) + 1e-15)
        denom_l = l0 if float(torch.abs(l0).item()) > 1e-6 else torch.tensor(1.0, device=l0.device, dtype=l0.dtype)
        rel_l_drift = torch.abs(ang_tensor - l0) / denom_l

        total_flops = self.flops_per_step_fn(integrator, b_size, n_bodies, steps_executed)
        gflops = (total_flops / 1e9) / max(elapsed_ms / 1000.0, 1e-9)

        return SimulationResult(
            integrator_name=integrator.name,
            is_symplectic=integrator.is_symplectic,
            order=integrator.order,
            times=times.detach().cpu(),
            positions_history=pos_hist[:, 0].detach().cpu(),
            velocities_history=vel_hist[:, 0].detach().cpu(),
            energy_history=energy_tensor.detach().cpu(),
            relative_energy_drift=rel_e_drift.detach().cpu(),
            angular_momentum_drift=rel_l_drift.detach().cpu(),
            virial_history=virial_tensor.detach().cpu(),
            elapsed_ms=elapsed_ms,
            steps_executed=steps_executed,
            gflops=gflops,
        )


class PipelineFarmEngine:
    """
    Orquestador Maestro del Esqueleto Pipeline-Farm en GPU.
    Coordina EmitterStage -> GPUWorkerFarmStage (Multi-Stream) -> CollectorStage.
    """

    def __init__(self, config: PipelineConfig, auditor: IPhysicsAuditor) -> None:
        self.config = config
        self.auditor = auditor
        self.emitter = EmitterStage()
        self.farm = GPUWorkerFarmStage()
        self.collector = CollectorStage(auditor=auditor, flops_per_step_fn=self._estimate_integrator_flops)

        if torch.cuda.is_available() and config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.streams: List[torch.cuda.Stream | None] = (
            [torch.cuda.Stream() for _ in range(config.num_cuda_streams)]
            if torch.cuda.is_available()
            else [None]
        )

    @staticmethod
    def _estimate_integrator_flops(
        integrator: IIntegrator, batch_size: int, num_bodies: int, steps: int
    ) -> float:
        name_lower = integrator.name.lower()
        if "rk4" in name_lower:
            evals_per_step = 4.0
        elif "yoshida" in name_lower:
            evals_per_step = 3.0
        elif "verlet" in name_lower or "rk2" in name_lower:
            evals_per_step = 2.0
        elif "transformer" in name_lower:
            # Evaluación física ancla + atención multi-cabeza + FFN
            return float(steps) * float(batch_size) * (float(num_bodies) ** 2 * 40.0 + float(num_bodies) * 65536.0)
        else:
            evals_per_step = 1.0
        return float(steps) * evals_per_step * float(batch_size) * (float(num_bodies) ** 2) * 24.0

    def execute_task(
        self,
        integrator: IIntegrator,
        initial_state: PhaseSpaceState,
        dt: float,
        num_steps: int,
        record_interval: int | None = None,
        stream_idx: int = 0,
    ) -> SimulationResult:
        """Ejecuta una tarea completa a través de las 3 etapas del Pipeline-Farm."""
        rec_int = record_interval if record_interval is not None else self.config.record_interval
        stream = self.streams[stream_idx % len(self.streams)]

        payload: Dict[str, Any] = {
            "integrator": integrator,
            "initial_state": initial_state,
            "dt": dt,
            "num_steps": num_steps,
            "record_interval": rec_int,
        }

        payload = self.emitter.execute(payload, stream=stream)
        payload = self.farm.execute(payload, stream=stream)
        return self.collector.execute(payload, stream=stream)

    def generate_training_dataset(
        self,
        reference_integrator: IIntegrator,
        initial_batch_state: PhaseSpaceState,
        dt: float,
        total_fine_steps: int,
        seq_history: int,
        macro_step_jump: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Genera tensores de entrenamiento en paralelo sobre GPU usando el integrador de alta precisión:
          Retorna: (hist_pos, hist_vel, target_pos, target_vel, masses)
        """
        state = initial_batch_state.clone()
        macro_positions = [state.positions.clone()]
        macro_velocities = [state.velocities.clone()]

        num_macro_transitions = total_fine_steps // macro_step_jump
        for _ in range(num_macro_transitions):
            for _ in range(macro_step_jump):
                state = reference_integrator.step(state, dt)
            macro_positions.append(state.positions.clone())
            macro_velocities.append(state.velocities.clone())

        # Apilar serie macro-temporal: [B, T_macro, N, 3]
        all_q = torch.stack(macro_positions, dim=1)
        all_v = torch.stack(macro_velocities, dim=1)

        # Construir ventanas deslizantes de longitud seq_history -> predicción en t + 1 macro-paso
        hist_q_list, hist_v_list, targ_q_list, targ_v_list, mass_list = [], [], [], [], []
        total_macro = all_q.shape[1]

        for t in range(total_macro - seq_history):
            hist_q_list.append(all_q[:, t : t + seq_history])
            hist_v_list.append(all_v[:, t : t + seq_history])
            targ_q_list.append(all_q[:, t + seq_history])
            targ_v_list.append(all_v[:, t + seq_history])
            mass_list.append(initial_batch_state.masses)

        return (
            torch.cat(hist_q_list, dim=0),
            torch.cat(hist_v_list, dim=0),
            torch.cat(targ_q_list, dim=0),
            torch.cat(targ_v_list, dim=0),
            torch.cat(mass_list, dim=0),
        )
