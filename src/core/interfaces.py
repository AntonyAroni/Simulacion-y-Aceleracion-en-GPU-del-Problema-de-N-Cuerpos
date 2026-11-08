"""
Contratos e Interfaces Abstractas (Principios ISP, LSP y DIP de SOLID).
Garantiza que cualquier integrador numérico, kernel de fuerzas o modelo Transformer
pueda intercambiarse de forma transparente dentro del motor Pipeline-Farm.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from .state import PhaseSpaceState


class IForceKernel(ABC):
    """Interfaz para el cálculo de aceleraciones y energía potencial de interacción."""

    @abstractmethod
    def compute_acceleration(self, positions: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """
        Calcula la aceleración gravitacional a_i para cada cuerpo.
        Entrada: positions [B, N, 3], masses [B, N, 1] -> Salida: accelerations [B, N, 3]
        """

    @abstractmethod
    def compute_potential_energy(self, positions: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """
        Calcula la energía potencial total V(q) por elemento del lote.
        Entrada: positions [B, N, 3], masses [B, N, 1] -> Salida: V [B]
        """


class IIntegrator(ABC):
    """Interfaz común para integradores numéricos clásicos y aceleradores neuronales (LSP)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador del método de integración."""

    @property
    @abstractmethod
    def order(self) -> int:
        """Orden formal de convergencia del esquema."""

    @property
    @abstractmethod
    def is_symplectic(self) -> bool:
        """Indica si el integrador preserva exactamente o por construcción la geometría simpléctica."""

    @abstractmethod
    def step(self, state: PhaseSpaceState, dt: float) -> PhaseSpaceState:
        """Avanza el estado dinámico un paso temporal dt."""


class ISurrogateModel(ABC):
    """Interfaz para operadores neuronales y Transformers sustitutos."""

    @abstractmethod
    def predict_macro_step(
        self,
        history_positions: torch.Tensor,
        history_velocities: torch.Tensor,
        masses: torch.Tensor,
        reference_energy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predice el estado (q_{t+M}, v_{t+M}) saltando M pasos de integración.
        """


class IPhysicsAuditor(ABC):
    """Interfaz para la auditoría de leyes de conservación e invariantes geométricos."""

    @abstractmethod
    def evaluate_invariants(self, state: PhaseSpaceState) -> Dict[str, torch.Tensor]:
        """Devuelve diccionario con energía cinética, potencial, total, momento lineal y angular."""


class IPipelineStage(ABC):
    """Interfaz para etapas dentro del esqueleto paralelo Pipeline-Farm."""

    @abstractmethod
    def execute(self, payload: Any, stream: torch.cuda.Stream | None = None) -> Any:
        """Ejecuta la carga de trabajo asignada en el stream CUDA correspondiente."""
