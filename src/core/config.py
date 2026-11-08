"""
Configuraciones inmutables y tipadas para el sistema de simulación híbrida.
Principio SOLID: Single Responsibility Principle (SRP) - separa parámetros de la lógica algorítmica.
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SimulationConfig:
    """Configuración física y numérica del problema de N-cuerpos."""
    g_const: float = 1.0
    softening: float = 1e-4
    dt: float = 0.002
    num_steps: int = 5000
    dtype: torch.dtype = torch.float64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TransformerConfig:
    """Configuración arquitectónica del Symplectic Structure-Preserving Transformer."""
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.0
    seq_history: int = 4
    macro_step_jump: int = 10
    use_gravity_bias: bool = True
    use_manifold_projection: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuración del optimizador de segundo orden SOAP y pérdida auto-adaptativa."""
    lr: float = 2e-3
    betas: tuple[float, float] = (0.9, 0.999)
    shampoo_beta: float = 0.95
    weight_decay: float = 1e-5
    precondition_frequency: int = 5
    epochs: int = 120
    batch_size: int = 64
    use_self_adaptive_loss: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración del esqueleto paralelo Pipeline-Farm en GPU (CUDA Streams)."""
    num_cuda_streams: int = 4
    enable_tf32: bool = True
    enable_jit_compile: bool = True
    record_interval: int = 10
