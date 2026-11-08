# Matriz Comparativa: Pros, Contras y Análisis Estratégico

Este documento evalúa de forma rigurosa las cuatro rutas de investigación posibles para el proyecto final de **Física Computacional (UNSA 2026-B)**, contrastando tu perfil profesional (CV: Vision Transformers, Triton, C++17, OpenGL, GPU Pipeline-Farm) contra los objetivos académicos del Dr. Danny Giancarlo Apaza Veliz.

---

## 1. Definición Detallada de las Cuatro Rutas Candidatas

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CUATRO RUTAS DE INVESTIGACIÓN                                  │
├──────────────────────────┬──────────────────────────┬─────────────────────────┬──────────────────┤
│ RUTA 1: HPC & SIMPLÉCTICO│ RUTA 2: TRANSFORMER SCI  │ RUTA 3: SCHRÖDINGER PINN│ RUTA 4: AUTÓMATAS│
│ N-Cuerpos GPU: Yoshida   │ Surrogate Transformer /  │ PINNsFormer Cuántico vs.│ Celulares 2D con │
│ vs RK4 + Visor OpenGL    │ FNO para Dinámica Caótica│ Disparo (RK4 + Secante) │ Triton / Shaders │
└──────────────────────────┴──────────────────────────┴─────────────────────────┴──────────────────┘
```

---

### RUTA 1: Simulación y Aceleración en GPU de N-Cuerpos (Integrador Simpléctico Yoshida 4.º orden vs. RK4 Adaptativo)
* **Hipótesis física:** En integraciones de dinámica orbital a largo plazo ($t > 10^5$ pasos), el integrador simpléctico de Yoshida (orden 4) preserva la invariancia del Hamiltoniano ($\Delta H / H_0 \approx 10^{-6}$ acotada sin deriva secular), mientras que RK4 estándar, a pesar de tener orden de convergencia local $\mathcal{O}(\Delta t^4)$, acumula una disipación numérica artificial que degrada la estabilidad de las órbitas planetarias.
* **Aporte computacional:** Implementación de un kernel en **C++17/CUDA** o **Triton** para el cómputo de fuerzas gravitacionales directas $\mathcal{O}(N^2)$ con *shared memory tiling* y paralelismo por bloques, complementado con un **visor interactivo en OpenGL 3.3** con shaders GLSL.

### RUTA 2: Transformer Estructurado / Neural Operator como Acelerador Sustituto de EDOs vs. RK4
* **Hipótesis física:** Un modelo basado en autoatención (*Transformer-based Neural Operator* o *Structure-Preserving Transformer*) entrenado sobre trayectorias numéricas de alta precisión generadas por RK4 puede predecir la evolución macrotemporal en bloques de tiempo largos ($\Delta T \gg \Delta t$), logrando una aceleración de $50\times - 100\times$ en GPU con una cota de deriva energética controlada frente a la propagación secuencial paso a paso de RK4.
* **Aporte computacional:** Diseño de un Transformer con tokenización espacial/temporal de partículas, optimizado mediante **PyTorch JIT (`torch.compile`)** y kernels de atención personalizados en **Triton**, comparando la violación de invariantes físicos (momento angular, energía) frente al método numérico clásico.

### RUTA 3: Physics-Informed Transformer (PINNsFormer) para la Ecuación de Schrödinger 1D vs. Método de Disparo
* **Hipótesis física:** Un Transformer informado por la física (PINNsFormer) optimizado mediante una función de pérdida que castiga el residuo del operador Hamiltoniano $-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi$ y la condición de normalización puede inferir los autovalores de energía $E_n$ y funciones de onda $\psi_n(x)$ para pozos de potencial asimétricos o discontinuos con menor sensibilidad a la condición inicial que el método clásico de disparo (*Shooting Method* con RK4 + Secante).
* **Aporte computacional:** Arquitectura con capas de atención multi-cabeza y activación tipo Wavelet para resolver el problema de autovalores de forma *mesh-free*, benchmarkeada directamente contra el algoritmo clásico implementado en NumPy/C++.

### RUTA 4: Autómatas Celulares Tensoriales Masivos en GPU (Kernels Triton) para Flujo de Tráfico y Percolación
* **Hipótesis física:** La dinámica colectiva de transición de fase en el modelo de tráfico de Nagel-Schreckenberg (aparición de congestión "fantasma") y el umbral crítico de percolación exhiben invariancia de escala demostrable numéricamente mediante simulaciones a escala masiva ($N > 10^7$ celdas activas simultáneas).
* **Aporte computacional:** Aceleración de las reglas de transición probabilística mediante operaciones tensoriales vectorizadas en **Triton** y renderizado directo mediante texturas OpenGL / compute shaders.

---

## 2. Matriz de Contraste Multidimensional

| Criterio de Decisión | Peso | RUTA 1: N-Body GPU Simpléctico | RUTA 2: Surrogate Transformer / FNO | RUTA 3: PINNsFormer Schrödinger 1D | RUTA 4: Autómatas Triton |
|---|:---:|:---:|:---:|:---:|:---:|
| **Alineación con tu CV (ViT / GPU / HPC / C++)** | 25% | **9.5/10** (C++17, CUDA, Triton, OpenGL 3.3) | **10.0/10** (Transformers, ViT, Triton, GPU) | **8.5/10** (Transformers, SciML, PyTorch) | **8.0/10** (Triton, C++, OpenGL) |
| **Cumplimiento estricto del Sílabo UNSA** | 25% | **10.0/10** (Temas 08, 10, 11 directo) | **8.0/10** (Requiere justificar RK4 como baseline) | **9.5/10** (Tema 20 idéntico: Shooting RK4) | **9.0/10** (Temas 14, 15 directo) |
| **Aceptación y Criterio del Docente (Dr. Apaza)** | 20% | **10.0/10** (Cero riesgo de rechazo; 100% física) | **7.5/10** (Si ve ML sin física estricta, observa) | **9.0/10** (Método numérico validado por papers 2026) | **8.5/10** (Fácil de entender y evaluar) |
| **Viabilidad y Control de Tiempo (10.º Ciclo)** | 15% | **9.5/10** (Sin hiperparámetros de entrenamiento) | **7.0/10** (Entrenamiento, tuning de loss física) | **8.0/10** (Entrenamiento rápido en 1D) | **9.0/10** (Rápido de codificar y probar) |
| **Potencial de Publicación (IEEE / Scopus)** | 10% | **8.5/10** (HPC / IEEE Latin America / SCCC) | **9.5/10** (Top AI / SciML Conferences) | **8.5/10** (Scopus / IEEE Q2-Q3) | **6.5/10** (Tema clásico muy explorado) |
| **Impacto en Responsabilidad Social (Demo)** | 5% | **10.0/10** (Simulador orbital 3D interactivo) | **8.5/10** (Web demo interactiva con inferencia) | **8.0/10** (Visualizador de funciones cuánticas) | **9.0/10** (Simulador visual de tráfico 2D) |
| **PUNTAJE PONDERADO TOTAL** | **100%** | **9.55 / 10** | **8.45 / 10** | **8.80 / 10** | **8.35 / 10** |

---

## 3. Análisis FODA por Ruta

### RUTA 1: N-Body Symplectic GPU (Yoshida vs. RK4)
* **Fortalezas:** Máxima solidez teórica ante el docente. Encaja perfectamente en la Semana 8 (Gravitación) y Semana 5 (RK4). Puedes reutilizar tu base gráfica de OpenGL 3.3 para renderizar el sistema solar / cúmulo globular en tiempo real.
* **Oportunidades:** Puedes implementar el kernel de fuerzas tanto en C++/CUDA como en Triton, demostrando tu competencia de compiladores de dominio específico frente a librerías estándar.
* **Debilidades:** No emplea redes neuronales ni Transformers directamente.
* **Amenazas:** Que el docente considere que $N$-cuerpos clásico es un tema estándar si no se enfatiza fuertemente el análisis simpléctico de orden 4 (Yoshida) y el profiling de hardware en GPU.

### RUTA 2: Structure-Preserving Transformer / FNO Surrogate
* **Fortalezas:** Es el estado del arte absoluto en Inteligencia Artificial Científica (SciML). Demuestra tu capacidad de investigación en Transformers y autoatención (publicación SCCC 2025).
* **Oportunidades:** Gran potencial de convertir el informe final del curso en un paper indexado de alto impacto en Scopus/IEEE.
* **Debilidades:** Mayor costo de ingeniería: generación de datasets sintéticos con RK4, balanceo de hiperparámetros y riesgo de inestabilidad durante el entrenamiento autorregresivo.
* **Amenazas:** **Riesgo ante el docente:** Si el Dr. Apaza nota que la red no preserva rigurosamente el invariante o no se explica a fondo la deducción matemática del RK4 del sílabo, exigirá correcciones teóricas.

### RUTA 3: PINNsFormer Cuántico para Schrödinger 1D vs. Método de Disparo
* **Fortalezas:** Se apoya en la literatura más reciente disponible (**Mahmood et al., ArXiv Feb 2026** y **Sarkar, 2025**), donde se compara exactamente el **Método de Disparo (RK4 + Secante)** del Tema 20 contra redes informadas por la física.
* **Oportunidades:** El dominio es 1D, por lo que el entrenamiento del Transformer converge en minutos en cualquier GPU moderna, reduciendo drásticamente el tiempo de experimentación.
* **Debilidades:** Menor lucimiento para un motor gráfico interactivo 3D tipo OpenGL en comparación con la simulación orbital.
* **Amenazas:** Es un tema de la Tercera Unidad (Semana 16), por lo que para los entregables de la Fase 1 (Semana 6) y Fase 2 (Semana 12) se debe adelantar el marco teórico cuántico.

---

## 4. Veredicto y Opciones Recomendadas

1. **Si priorizas SEGURIDAD ACADÉMICA ABSOLUTA + RENDIMIENTO GPU MÁXIMO:**  
   👉 **Elegir RUTA 1 (N-Body Symplectic GPU con Yoshida vs RK4 + Visor OpenGL 3.3).**  
   *Justificación:* Riesgo académico nulo, aprovecha tu dominio en C++17, CUDA/Triton y OpenGL, y cumple con el 100% de la rúbrica del sílabo sin depender del azar de entrenamiento de una red.

2. **Si priorizas MANTENERTE EN TRANSFORMERS / IA Y PUBLICAR EN SCOPUS:**  
   👉 **Elegir RUTA 2 bajo un esquema HÍBRIDO BLINDADO:**  
   *Formulación:* *"Acelerador Basado en Transformers para la Dinámica de N-Cuerpos/Osciladores Caóticos: Evaluación Comparativa de Deriva Simpléctica frente a Runge-Kutta 4 en GPU"*.  
   *Estrategia:* El núcleo del paper parte obligatoriamente del análisis formal de RK4 y Yoshida (satisfaciendo al Dr. Apaza), y propone el Transformer como módulo acelerador de inferencia rápida en GPU.

3. **Si te atrae la Física Cuántica Computacional (Capítulo IX):**  
   👉 **Elegir RUTA 3 (PINNsFormer vs Shooting Method en el Pozo Cuántico).**  
   *Justificación:* Está respaldado directamente por los papers del 2025/2026 y conecta de lleno con los métodos no lineales (Secante/Newton) y RK4 exigidos en el sílabo.
