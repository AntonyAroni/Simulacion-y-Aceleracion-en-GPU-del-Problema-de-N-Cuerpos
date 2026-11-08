"""
Módulo Core: Configuraciones, contenedor de estado del espacio de fases e interfaces abstractas SOLID.
"""

from .config import SimulationConfig, TransformerConfig, OptimizerConfig, PipelineConfig
from .state import PhaseSpaceState
from .interfaces import (
    IForceKernel,
    IIntegrator,
    ISurrogateModel,
    IPhysicsAuditor,
    IPipelineStage,
)

__all__ = [
    "SimulationConfig",
    "TransformerConfig",
    "OptimizerConfig",
    "PipelineConfig",
    "PhaseSpaceState",
    "IForceKernel",
    "IIntegrator",
    "ISurrogateModel",
    "IPhysicsAuditor",
    "IPipelineStage",
]
