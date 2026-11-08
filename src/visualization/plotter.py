"""
Generador de Gráficas Científicas para Artículo IEEE / Scopus.
Principio SOLID: Single Responsibility Principle (SRP) - separa la representación visual del cálculo numérico.
"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..pipeline.pipeline_farm import SimulationResult


class ScientificPlotter:
    """Genera figuras comparativas de alta resolución (300 DPI) para el artículo final."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 300,
        })

    def plot_orbits_and_phase_space(self, results: List[SimulationResult]) -> Path:
        """
        Figura 1: Trayectorias orbitales (q_x vs q_y) y Retrato en el Espacio de Fases (q_x vs v_x).
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#f39c12"]

        ax_orb, ax_phase = axes[0], axes[1]
        for idx, res in enumerate(results):
            color = colors[idx % len(colors)]
            pos = res.positions_history.numpy()  # [T, N, 3]
            vel = res.velocities_history.numpy()
            linestyle = "--" if "Transformer" in res.integrator_name else "-"
            alpha = 0.9 if ("Yoshida" in res.integrator_name or "Transformer" in res.integrator_name) else 0.7

            # Graficar cuerpo 1 en el plano orbital XY
            ax_orb.plot(
                pos[:, 0, 0],
                pos[:, 0, 1],
                label=res.integrator_name,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
                alpha=alpha,
            )
            # Retrato de fase (q_x vs p_x) del cuerpo 1
            ax_phase.plot(
                pos[:, 0, 0],
                vel[:, 0, 0],
                label=res.integrator_name,
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                alpha=alpha,
            )

        ax_orb.set_title("A) Órbita Coreográfica de 3 Cuerpos (Plano $q_x - q_y$)")
        ax_orb.set_xlabel("Posición $q_x$")
        ax_orb.set_ylabel("Posición $q_y$")
        ax_orb.grid(True, linestyle=":", alpha=0.6)
        ax_orb.legend(loc="upper right")

        ax_phase.set_title("B) Retrato en el Espacio de Fases ($q_x$ vs $p_x$)")
        ax_phase.set_xlabel("Coordenada Generalizada $q_x$")
        ax_phase.set_ylabel("Momento Conjugado $p_x$")
        ax_phase.grid(True, linestyle=":", alpha=0.6)
        ax_phase.legend(loc="upper right")

        fig.tight_layout()
        out_path = self.output_dir / "fig1_orbitas_y_espacio_fases.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_conservation_invariants(self, results: List[SimulationResult]) -> Path:
        """
        Figura 2: Deriva Relativa del Hamiltoniano |Delta E / E_0| en escala logarítmica
        y evolución de la Energía Total H(t).
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        colors = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad", "#f39c12"]

        ax_drift, ax_energy = axes[0], axes[1]
        for idx, res in enumerate(results):
            color = colors[idx % len(colors)]
            t_arr = res.times.numpy()
            drift = np.maximum(res.relative_energy_drift.numpy(), 1e-16)
            energy = res.energy_history.numpy()

            ax_drift.semilogy(
                t_arr,
                drift,
                label=res.integrator_name,
                color=color,
                linewidth=1.6,
            )
            ax_energy.plot(
                t_arr,
                energy,
                label=res.integrator_name,
                color=color,
                linewidth=1.6,
            )

        ax_drift.set_title(r"A) Deriva Energética Relativa $|\Delta H(t) / H_0|$ (Escala Log)")
        ax_drift.set_xlabel("Tiempo Físico $t$")
        ax_drift.set_ylabel(r"Error Relativo $|\Delta E / E_0|$")
        ax_drift.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_drift.legend(loc="best")

        ax_energy.set_title(r"B) Evolución del Hamiltoniano Total $H(\mathbf{q}, \mathbf{p}) = T + V$")
        ax_energy.set_xlabel("Tiempo Físico $t$")
        ax_energy.set_ylabel("Energía Total $H(t)$")
        ax_energy.grid(True, linestyle=":", alpha=0.6)
        ax_energy.legend(loc="best")

        fig.tight_layout()
        out_path = self.output_dir / "fig2_conservacion_hamiltoniano.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_soap_and_gpu_scaling(
        self,
        training_logs: List[Dict[str, float]],
        scaling_records: List[Dict[str, float]],
    ) -> Path:
        """
        Figura 3: Convergencia del Optimizador de 2.º Orden SOAP + Escalabilidad GPU vs CPU (Speedup y GFLOPS).
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        ax_train, ax_scale = axes[0], axes[1]

        epochs = [row["epoch"] for row in training_logs]
        pos_mse = [max(row["pos_mse"], 1e-12) for row in training_logs]
        vel_mse = [max(row["vel_mse"], 1e-12) for row in training_logs]
        ham_mse = [max(row["hamiltonian_mse"], 1e-12) for row in training_logs]

        ax_train.semilogy(epochs, pos_mse, label="MSE Posición $\\mathcal{L}_q$", color="#2980b9", linewidth=1.6)
        ax_train.semilogy(epochs, vel_mse, label="MSE Velocidad $\\mathcal{L}_v$", color="#27ae60", linewidth=1.6)
        ax_train.semilogy(epochs, ham_mse, label="Residuo Hamiltoniano $\\mathcal{L}_H$", color="#8e44ad", linewidth=1.6)
        ax_train.set_title("A) Convergencia del Transformer con Optimizador SOAP")
        ax_train.set_xlabel("Época de Entrenamiento")
        ax_train.set_ylabel("Pérdida Física (Escala Log)")
        ax_train.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_train.legend(loc="upper right")

        n_vals = [int(r["num_bodies"]) for r in scaling_records]
        speedups = [r["speedup_gpu_vs_cpu"] for r in scaling_records]
        gflops_gpu = [r["gpu_gflops"] for r in scaling_records]

        color_sp = "#d35400"
        color_gf = "#16a085"
        ax_scale.plot(n_vals, speedups, marker="o", color=color_sp, linewidth=2.0, label="Speedup GPU vs CPU ($\\times$)")
        ax_scale.set_xscale("log", base=2)
        ax_scale.set_title("B) Escalabilidad en GPU (RTX 4050) con Arquitectura Pipeline-Farm")
        ax_scale.set_xlabel("Número de Cuerpos $N$ (Complejidad $\\mathcal{O}(N^2)$)")
        ax_scale.set_ylabel("Factor de Aceleración Speedup ($\\times$)", color=color_sp)
        ax_scale.tick_params(axis="y", labelcolor=color_sp)
        ax_scale.grid(True, which="both", linestyle=":", alpha=0.6)

        ax_gf = ax_scale.twinx()
        ax_gf.plot(n_vals, gflops_gpu, marker="s", linestyle="--", color=color_gf, linewidth=1.8, label="Rendimiento GPU (GFLOPS)")
        ax_gf.set_ylabel("Rendimiento Computacional (GFLOPS)", color=color_gf)
        ax_gf.tick_params(axis="y", labelcolor=color_gf)

        fig.tight_layout()
        out_path = self.output_dir / "fig3_optimizador_soap_y_escalabilidad_gpu.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path
