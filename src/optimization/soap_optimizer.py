"""
Optimizador Cuasi-Segundo Orden SOAP (Shampoo with Adam in the Preconditioner's Eigenbasis)
y Entrenador del Operador Simpléctico Transformer.

Referencias:
  - Vyas et al. (ICLR 2025), "SOAP: Improving and Stabilizing Shampoo using Adam", arXiv:2409.11321
  - Wang et al. (NeurIPS 2025), "Gradient Alignment in Physics-Informed Neural Networks"
"""

import math
from typing import Iterable, Dict, List
import torch
from torch.optim import Optimizer
from ..core.config import OptimizerConfig
from .losses import SelfAdaptivePhysicsLoss


class SOAPOptimizer(Optimizer):
    """
    Implementación de SOAP (Shampoo + Adam en la base propia del precondicionador de Kronecker).
    Para matrices de pesos 2D W de dimensión (m, n):
      1. Acumula factores de covarianza direccional L_t = beta_s L_{t-1} + (1 - beta_s) G G^T
         y R_t = beta_s R_{t-1} + (1 - beta_s) G^T G.
      2. Cada `precondition_frequency` pasos actualiza las bases ortogonales Q_L, Q_R mediante
         descomposición espectral simétrica (torch.linalg.eigh).
      3. Proyecta el gradiente a la base propia: G_rot = Q_L^T G Q_R.
      4. Aplica momentos adaptativos de Adam (m_t, v_t) en la base propia y reproyecta al espacio original.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
        precondition_frequency: int = 5,
        max_precond_dim: int = 512,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precond_freq = group["precondition_frequency"]
            max_dim = group["max_precond_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    if p.ndim == 2 and p.shape[0] <= max_dim and p.shape[1] <= max_dim:
                        m_dim, n_dim = p.shape
                        state["use_kronecker"] = True
                        state["L"] = torch.eye(m_dim, device=p.device, dtype=p.dtype) * eps
                        state["R"] = torch.eye(n_dim, device=p.device, dtype=p.dtype) * eps
                        state["Q_L"] = torch.eye(m_dim, device=p.device, dtype=p.dtype)
                        state["Q_R"] = torch.eye(n_dim, device=p.device, dtype=p.dtype)
                    else:
                        state["use_kronecker"] = False

                state["step"] += 1
                step_t = state["step"]

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                if state["use_kronecker"]:
                    # Actualizar factores de Kronecker L y R
                    L_mat = state["L"]
                    R_mat = state["R"]
                    L_mat.mul_(shampoo_beta).add_(grad @ grad.T, alpha=1.0 - shampoo_beta)
                    R_mat.mul_(shampoo_beta).add_(grad.T @ grad, alpha=1.0 - shampoo_beta)

                    if step_t == 1 or step_t % precond_freq == 0:
                        try:
                            _, q_l = torch.linalg.eigh(L_mat + eps * torch.eye(L_mat.shape[0], device=p.device, dtype=p.dtype))
                            _, q_r = torch.linalg.eigh(R_mat + eps * torch.eye(R_mat.shape[0], device=p.device, dtype=p.dtype))
                            state["Q_L"] = q_l
                            state["Q_R"] = q_r
                        except RuntimeError:
                            pass

                    q_l = state["Q_L"]
                    q_r = state["Q_R"]

                    # Proyectar gradiente a la base propia del precondicionador
                    grad_eigen = q_l.T @ grad @ q_r

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad_eigen, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_eigen, grad_eigen, value=1.0 - beta2)

                    bias_corr1 = 1.0 - beta1 ** step_t
                    bias_corr2 = 1.0 - beta2 ** step_t

                    step_eigen = (exp_avg / bias_corr1) / (torch.sqrt(exp_avg_sq / bias_corr2) + eps)
                    # Reproyectar al espacio original de parámetros
                    update = q_l @ step_eigen @ q_r.T
                    p.add_(update, alpha=-lr)
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_corr1 = 1.0 - beta1 ** step_t
                    bias_corr2 = 1.0 - beta2 ** step_t

                    denom = torch.sqrt(exp_avg_sq / bias_corr2) + eps
                    p.addcdiv_(exp_avg / bias_corr1, denom, value=-lr)

        return loss


class SurrogateTrainer:
    """
    Orquestador de entrenamiento para el SymplecticTransformerOperator utilizando
    SOAP + Cosine Annealing + SelfAdaptivePhysicsLoss.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: SelfAdaptivePhysicsLoss,
        config: OptimizerConfig,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

        all_params = list(model.parameters()) + list(loss_fn.parameters())
        self.optimizer = SOAPOptimizer(
            params=all_params,
            lr=config.lr,
            betas=config.betas,
            shampoo_beta=config.shampoo_beta,
            weight_decay=config.weight_decay,
            precondition_frequency=config.precondition_frequency,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.02
        )

    def train_model(
        self,
        hist_pos: torch.Tensor,
        hist_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        masses: torch.Tensor,
    ) -> List[Dict[str, float]]:
        """
        Entrena el operador sobre el dataset de trayectorias físicas en GPU.
        """
        self.model.train()
        num_samples = hist_pos.shape[0]
        batch_size = min(self.config.batch_size, num_samples)
        history_logs: List[Dict[str, float]] = []

        for epoch in range(1, self.config.epochs + 1):
            perm = torch.randperm(num_samples, device=hist_pos.device)
            epoch_total = 0.0
            epoch_pos = 0.0
            epoch_vel = 0.0
            epoch_ham = 0.0
            epoch_ang = 0.0
            n_batches = 0

            for start_idx in range(0, num_samples, batch_size):
                idx = perm[start_idx : start_idx + batch_size]
                b_hq = hist_pos[idx]
                b_hv = hist_vel[idx]
                b_tq = target_pos[idx]
                b_tv = target_vel[idx]
                b_m = masses[idx]

                self.optimizer.zero_grad()
                pred_q, pred_v = self.model(b_hq, b_hv, b_m, enforce_energy_manifold=False)
                losses = self.loss_fn(pred_q, pred_v, b_tq, b_tv, b_m)

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_total += float(losses["total"].item())
                epoch_pos += float(losses["pos"].item())
                epoch_vel += float(losses["vel"].item())
                epoch_ham += float(losses["hamiltonian"].item())
                epoch_ang += float(losses["angular"].item())
                n_batches += 1

            self.scheduler.step()
            history_logs.append(
                {
                    "epoch": float(epoch),
                    "total_loss": epoch_total / max(n_batches, 1),
                    "pos_mse": epoch_pos / max(n_batches, 1),
                    "vel_mse": epoch_vel / max(n_batches, 1),
                    "hamiltonian_mse": epoch_ham / max(n_batches, 1),
                    "angular_mse": epoch_ang / max(n_batches, 1),
                    "lr": float(self.scheduler.get_last_lr()[0]),
                }
            )

        self.model.eval()
        return history_logs
