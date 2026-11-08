"""
Suite de Pruebas Unitarias y Verificación Física (Principios SOLID, Conservación y GPU).
Ejecutar con: python3 -m unittest discover -s 0_PROYECTO/tests -p "test_*.py"
"""

import unittest
import torch

from src.core.config import SimulationConfig, TransformerConfig, OptimizerConfig, PipelineConfig
from src.physics.gravity_kernel import DirectGravityKernel
from src.physics.invariants import HamiltonianAuditor
from src.physics.initial_conditions import InitialConditionFactory
from src.integrators.euler import EulerCromerIntegrator
from src.integrators.runge_kutta import RK4Integrator
from src.integrators.symplectic import Yoshida4Integrator
from src.models.transformer import SymplecticTransformerOperator, SurrogateTransformerIntegrator
from src.optimization.losses import SelfAdaptivePhysicsLoss
from src.optimization.soap_optimizer import SurrogateTrainer
from src.pipeline.pipeline_farm import PipelineFarmEngine


class TestPhysicsAndArchitectureSOLID(unittest.TestCase):
    """Pruebas de regresión física, contratos SOLID y aceleración CUDA."""

    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float64
        self.kernel = DirectGravityKernel(g_const=1.0, softening=1e-4)
        self.auditor = HamiltonianAuditor(force_kernel=self.kernel)

    def test_newton_third_law_momentum_conservation(self) -> None:
        """Verifica que la suma neta de fuerzas internas es exactamente cero (3.ª Ley de Newton)."""
        state = InitialConditionFactory.create_figure8_three_body(device=self.device, dtype=self.dtype)
        acc = self.kernel.compute_acceleration(state.positions, state.masses)
        total_force = torch.sum(state.masses * acc, dim=1)
        self.assertLess(float(torch.max(torch.abs(total_force)).item()), 1e-13)

    def test_yoshida4_vs_euler_energy_conservation(self) -> None:
        """Verifica que el integrador simpléctico de Yoshida de 4.º orden conserva el Hamiltoniano."""
        state = InitialConditionFactory.create_figure8_three_body(device=self.device, dtype=self.dtype)
        euler_c = EulerCromerIntegrator(self.kernel)
        yoshida = Yoshida4Integrator(self.kernel)

        e0 = self.auditor.evaluate_invariants(state)["total_energy"]

        s_ec = state.clone()
        s_yo = state.clone()
        for _ in range(200):
            s_ec = euler_c.step(s_ec, dt=0.005)
            s_yo = yoshida.step(s_yo, dt=0.005)

        drift_ec = HamiltonianAuditor.compute_relative_drift(
            self.auditor.evaluate_invariants(s_ec)["total_energy"], e0
        )
        drift_yo = HamiltonianAuditor.compute_relative_drift(
            self.auditor.evaluate_invariants(s_yo)["total_energy"], e0
        )
        self.assertLess(float(drift_yo.item()), float(drift_ec.item()) * 1e-4)
        self.assertLess(float(drift_yo.item()), 1e-7)

    def test_symplectic_transformer_manifold_and_soap(self) -> None:
        """Verifica que el SymplecticTransformerOperator y el optimizador SOAP convergen y preservan E_0."""
        t_cfg = TransformerConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, seq_history=4, macro_step_jump=5)
        model = SymplecticTransformerOperator(config=t_cfg, force_kernel=self.kernel, macro_dt=0.01).to(
            device=self.device, dtype=self.dtype
        )

        batch_state = InitialConditionFactory.create_perturbed_figure8_batch(
            batch_size=8, device=self.device, dtype=self.dtype
        )
        pipe_engine = PipelineFarmEngine(PipelineConfig(), self.auditor)
        rk4 = RK4Integrator(self.kernel)

        hq, hv, tq, tv, m = pipe_engine.generate_training_dataset(
            reference_integrator=rk4,
            initial_batch_state=batch_state,
            dt=0.002,
            total_fine_steps=40,
            seq_history=4,
            macro_step_jump=5,
        )

        loss_fn = SelfAdaptivePhysicsLoss(self.auditor, use_self_adaptive=True).to(
            device=self.device, dtype=self.dtype
        )
        opt_cfg = OptimizerConfig(lr=3e-3, epochs=8, batch_size=16)
        trainer = SurrogateTrainer(model, loss_fn, opt_cfg)
        logs = trainer.train_model(hq, hv, tq, tv, m)

        self.assertLess(logs[-1]["pos_mse"], logs[0]["pos_mse"])

        # Probar sustitución de Liskov (LSP) con SurrogateTransformerIntegrator y proyección en variedad
        surrogate_int = SurrogateTransformerIntegrator(
            model=model, warmup_integrator=rk4, auditor=self.auditor, macro_step_jump=5
        )
        fig8 = InitialConditionFactory.create_figure8_three_body(device=self.device, dtype=self.dtype)
        res = pipe_engine.execute_task(surrogate_int, fig8, dt=0.002, num_steps=10, record_interval=2)
        # Con proyección de variedad iso-energética activada, la deriva energética es ~ precisión de máquina
        self.assertLess(float(torch.max(res.relative_energy_drift).item()), 1e-11)


if __name__ == "__main__":
    unittest.main()
