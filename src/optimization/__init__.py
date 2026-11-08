"""
Módulo de Optimización Científica de Segundo Orden (SOAP) y Funciones de Pérdida Auto-Adaptativas.
"""

from .losses import SelfAdaptivePhysicsLoss
from .soap_optimizer import SOAPOptimizer, SurrogateTrainer

__all__ = [
    "SelfAdaptivePhysicsLoss",
    "SOAPOptimizer",
    "SurrogateTrainer",
]
