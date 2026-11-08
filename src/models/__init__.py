"""
Módulo de Arquitecturas Basadas en Transformers Informados por la Física y Estructura Simpléctica.
"""

from .embeddings import PhaseSpaceTokenizer, WaveletFourierActivation
from .attention import PhysicsGuidedAttentionBlock
from .projection import SymplecticManifoldProjection
from .transformer import SymplecticTransformerOperator, SurrogateTransformerIntegrator

__all__ = [
    "PhaseSpaceTokenizer",
    "WaveletFourierActivation",
    "PhysicsGuidedAttentionBlock",
    "SymplecticManifoldProjection",
    "SymplecticTransformerOperator",
    "SurrogateTransformerIntegrator",
]
