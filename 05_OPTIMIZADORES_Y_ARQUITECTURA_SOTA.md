# Estado del Arte en Optimizadores, Pérdidas Físicas y Arquitectura Pipeline-Farm (2024–2026)

Este documento detalla los algoritmos de optimización de última generación, funciones de pérdida auto-adaptativas y esquemas de paralelismo estructurado seleccionados de la literatura reciente (2024–2026) para potenciar cada etapa de nuestro algoritmo híbrido de $N$-cuerpos (**Yoshida 4.º Orden vs. RK4 vs. Symplectic Transformer**).

---

## 1. Nuevos Papers de Vanguardia Incorporados (2024–2026)

| Categoría | Paper / Referencia Real | Año / Venue | Aporte Aplicado a Nuestro Código |
|---|---|---|---|
| **Optimizador de 2.º Orden** | **SOAP: Improving and Stabilizing Shampoo using Adam**<br>*(N. Vyas, D. Morwani, R. Zhao, D. Kwun, S. Kakade — Harvard)* | **2024 / ICLR 2025**<br>`arXiv:2409.11321` | Ejecuta Adam en la base propia (*eigenbasis*) del precondicionador de Shampoo (Kronecker-factored). Acelera la convergencia en paisajes de pérdida rígidos (*stiff*). |
| **Alineación de Gradientes en PINNs** | **Gradient Alignment in Physics-Informed Neural Networks: A Second-Order Optimization Perspective**<br>*(Wang et al.)* | **NeurIPS 2025** | Demuestra que SOAP elimina el conflicto direccional entre los gradientes de datos ($\nabla \mathcal{L}_{\text{traj}}$) y los gradientes de conservación de energía ($\nabla \mathcal{L}_H$), logrando mejoras de $2\times$ a $10\times$ en precisión física. |
| **Pérdida Física Auto-Adaptativa** | **Self-Adaptive Loss Balanced Physics-Informed Neural Networks** / **Homoscedastic Uncertainty Weighting**<br>*(McClenny & Braga-Neto; Kendall et al.; Wang et al. NTK)* | **2023–2025** | Ajusta dinámicamente los pesos $\lambda_k = \frac{1}{2\sigma_k^2}$ de cada invariante físico (Energía, Momento Lineal, Momento Angular) durante el entrenamiento sin ajuste manual de hiperparámetros. |
| **Transformers con Estructura Simpléctica** | **Structure-Preserving Transformers for Learning Parametrized Hamiltonian Systems**<br>*(B. Brantner, G. de Romémont, M. Kraus, Z. Li — Max Planck)* | **2024**<br>`arXiv:2312.10046` | Introduce capas de atención compatibles con la geometría del espacio de fases $(\mathbf{q}, \mathbf{p})$ para evitar la deriva secular de energía en horizontes largos. |
| **Flujos Simplécticos Neuronales** | **Symplectic Neural Flows for Modeling and Discovery (SympFlow)**<br>*(Canizares et al.)* | **Dic 2024 / 2025**<br>`arXiv:2412.16787` | Combina mapas de flujo hamiltonianos parametrizados con proyección en variedades de energía constante. |
| **Redes Equivariantes en $N$-Cuerpos** | **SympGNNs: Symplectic Graph Neural Networks for Identifying High-Dimensional Hamiltonian Systems**<br>*(Varghese, Zhang, Karniadakis — Brown Univ.)* | **2024 / 2025**<br>`arXiv:2408.15139` | Incorpora sesgos relacionales equivariantes a permutaciones y distancias inversas $\|\mathbf{q}_i - \mathbf{q}_j\|^{-1}$ dentro del mecanismo de interacción entre partículas. |
| **Paralelismo Esquelético en GPU** | **Optimizing Vision Transformers for Object Detection through Skeleton-Based Parallelism**<br>*(Antony Aroni Jarata — UNSA)* | **IEEE SCCC 2025**<br>IEEE Xplore / Scopus | Arquitectura **Pipeline-Farm** adaptada aquí a flujos asíncronos multi-stream en CUDA para solapar generación de condiciones iniciales, integración/inferencia en paralelo y recolección de invariantes. |

---

## 2. Innovaciones Algorítmicas por Etapa del Sistema

Para que el proyecto destaque tanto en **rigor físico** como en **ingeniería de software e IA**, aplicamos optimizaciones específicas en cuatro niveles del algoritmo:

### Nivel A: Kernel Físico de Interacción $\mathcal{O}(N^2)$ en GPU
En lugar de bucles secuenciales en Python, el cálculo de aceleraciones gravitacionales:

$$\mathbf{a}_i = G \sum_{j \neq i} m_j \frac{\mathbf{q}_j - \mathbf{q}_i}{\left(\|\mathbf{q}_j - \mathbf{q}_i\|^2 + \epsilon^2\right)^{3/2}}$$

se formula mediante **operaciones tensoriales difundidas (*broadcasted matrix kernels*)** en CUDA con soporte para lotes masivos de sistemas en paralelo `[B, N, 3]`, aprovechando la jerarquía de caché L2 y núcleos CUDA de tu **NVIDIA GeForce RTX 4050**.

### Nivel B: Sesgo Inductivo Gravitacional en la Atención (Physics-Biased FlashAttention)
Un Transformer estándar calcula $\text{softmax}(QK^T / \sqrt{d_k})V$ sin conocer la ley del inverso del cuadrado. En nuestro módulo `PhysicsGuidedAttention`, inyectamos una **matriz de sesgo relacional gravitacional** $B_{ij}$ directamente en los logits de atención:

$$\text{Attention}(Q, K, V, \mathbf{q}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \gamma \cdot \Phi(\|\mathbf{q}_i - \mathbf{q}_j\|, m_i, m_j)\right)V$$

donde $\Phi$ codifica el potencial par-a-par $-\frac{G m_i m_j}{\sqrt{r_{ij}^2 + \epsilon^2}}$, dotando al Transformer de equivariancia translacional y conciencia directa de la intensidad gravitatoria.

### Nivel C: Proyección Geométrica en la Variedad de Conservación (Symplectic Manifold Projection)
Para garantizar que el Transformer jamás viole la conservación del momento lineal ni disipe energía en inferencia a largo plazo, añadimos una capa diferenciable de **Proyección Simpléctica de Variedad** al final del bloque predictor:
1. **Corrección exacta de Centro de Masa y Momento Lineal:**
   $$\mathbf{v}_i \leftarrow \mathbf{v}_i - \frac{1}{M_{\text{total}}}\sum_{j=1}^N m_j \mathbf{v}_j + \frac{\mathbf{P}_0}{M_{\text{total}}}$$
2. **Escalamiento Iso-Energético Suave (Manifold Projection):**
   Ajusta el factor cinético $\alpha = \sqrt{\frac{\max(E_0 - V(\mathbf{q}), 0)}{T(\mathbf{v})}}$ mediante una compuerta aprendible/física para anclar la trayectoria a la hipersuperficie $H(\mathbf{q}, \mathbf{p}) = E_0$.

### Nivel D: Optimizador SOAP + Función de Pérdida Auto-Adaptativa (Homoscedastic Uncertainty)
Durante el entrenamiento del Transformer sustituto, minimizamos una pérdida compuesta por 4 términos físicos con **pesos auto-adaptativos aprendibles** $s_k = \log(\sigma_k^2)$:

$$\mathcal{L}_{\text{total}} = \sum_{k \in \{\text{pos}, \text{vel}, H, L\}} \left( \frac{1}{2} e^{-s_k} \mathcal{L}_k + \frac{1}{2} s_k \right)$$

Y actualizamos los parámetros con **SOAP (Shampoo + Adam en Eigenbasis)**, el cual mantiene factores de precondicionamiento de Kronecker $L_t, R_t$ para las matrices de pesos del Transformer, proyectando el gradiente a la base ortonormal $Q_L^T G_t Q_R$, aplicando Adam en dicho espacio y reproyectando al espacio original.

---

## 3. Arquitectura de Software SOLID y Patrón Pipeline-Farm en GPU

Siguiendo tu línea de investigación publicada en **IEEE SCCC 2025**, el motor de ejecución implementa el esqueleto algorítmico **Pipeline-Farm Asíncrono sobre CUDA Streams**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       ARQUITECTURA SKELETON PIPELINE-FARM (CUDA STREAMS)                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   [ETAPA 1: EMITTER]                  [ETAPA 2: GPU WORKER FARM]         [ETAPA 3: COLLECTOR]   │
│  Generador de Estados  ──────────►  ┌─► Stream 0: Integrador RK4       ─┐                       │
│  Astrofísicos (Batch)               ├─► Stream 1: Simpléctico Yoshida  ─┼─► Reducción de        │
│  • Figura-8 (3 cuerpos)             ├─► Stream 2: Symplectic ViT/Trans ─┤   Invariantes (H, L)  │
│  • Sistema Planetario               └─► Stream k: Rollout Paralelo     ─┘   y Métricas GFLOPS   │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Aplicación Estricta de los Principios SOLID en `0_PROYECTO/src/`

1. **S — Single Responsibility Principle (Responsabilidad Única):**
   * `gravity_kernel.py`: Exclusivamente cálculo de fuerzas y potenciales gravitatorios.
   * `invariants.py`: Exclusivamente auditoría de leyes de conservación ($E, \mathbf{P}, \mathbf{L}$, Virial).
   * `runge_kutta.py` y `symplectic.py`: Exclusivamente esquemas de avance temporal numérico.
   * `soap_optimizer.py`: Exclusivamente álgebra de precondicionamiento de segundo orden.
2. **O — Open/Closed Principle (Abierto/Cerrado):**
   * Nuevos integradores (ej. *Ruth 3rd-order*, *Forest-Ruth*, *Leapfrog*) o nuevos modelos neuronales pueden añadirse heredando de `IIntegrator` sin modificar una sola línea del `PipelineFarmEngine`.
3. **L — Liskov Substitution Principle (Sustitución de Liskov):**
   * El envoltorio `SurrogateTransformerIntegrator` implementa exactamente la misma firma `step(state, dt)` y `integrate(state, dt, steps)` que `RK4Integrator` y `Yoshida4Integrator`.
4. **I — Interface Segregation Principle (Segregación de Interfaces):**
   * Interfaces separadas en `src/core/interfaces.py`: `IForceKernel`, `IIntegrator`, `ISurrogateModel`, `IPhysicsMetric`, `IPipelineStage`.
5. **D — Dependency Inversion Principle (Inversión de Dependencias):**
   * Tanto los integradores como el motor Pipeline-Farm reciben la abstracción `IForceKernel` por inyección de dependencias en el constructor.
