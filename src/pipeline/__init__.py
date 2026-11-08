"""
Módulo de Paralelismo Estructurado Esquelético: Arquitectura Pipeline-Farm Asíncrona en GPU (CUDA Streams).
Inspirado en:
  A. Aroni Jarata (IEEE SCCC 2025), "Optimizing Vision Transformers for Object Detection through Skeleton-Based Parallelism".
"""

from .pipeline_farm import (
    SimulationResult,
    EmitterStage,
    GPUWorkerFarmStage,
    CollectorStage,
    PipelineFarmEngine,
)

__all__ = [
    "SimulationResult",
    "EmitterStage",
    "GPUWorkerFarmStage",
    "CollectorStage",
    "PipelineFarmEngine",
]
