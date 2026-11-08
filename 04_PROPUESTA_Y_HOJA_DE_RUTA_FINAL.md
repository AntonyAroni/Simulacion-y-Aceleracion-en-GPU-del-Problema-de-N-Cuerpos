# Propuesta Técnica Definitiva y Hoja de Ruta de Ejecución

**Proyecto:** Framework Híbrido de Simulación Dinámica y Aceleración en GPU: Métodos Numéricos Clásicos vs. Operadores Basados en Transformers  
**Curso:** Física Computacional (E) — Universidad Nacional de San Agustín (UNSA)  

---

## 1. La Estrategia Maestra: "El Enfoque Híbrido Blindado"

Para resolver el dilema entre **satisfacer al 100% la rúbrica formal del Dr. Danny Apaza** (física matemática rigurosa, RK4, orden de convergencia, conservación de energía) y **explotar al máximo tu perfil de investigación en Transformers y GPU de bajo nivel**, la propuesta ganadora integra ambos mundos en una arquitectura unificada:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           ARQUITECTURA DEL FRAMEWORK CIENTÍFICO                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│  [MÓDULO 1: LÍNEA BASE NUMÉRICA CLÁSICA] (Ground Truth del Sílabo UNSA)                        │
│  • Integrador Runge-Kutta 4 (RK4) con paso fijo y adaptativo.                                    │
│  • Integrador Simpléctico de 4.º orden de Yoshida (Geometría simpléctica y Hamiltoniano).        │
│  • Análisis formal de error de truncamiento $\mathcal{O}(\Delta t^4)$ y deriva $\Delta E/E_0$.  │
│                                                                                                 │
│  [MÓDULO 2: ACELERACIÓN DE ALTO RENDIMIENTO (HPC / GPU)]                                       │
│  • Cálculo de interacciones directas interparticulares ($N$-cuerpos) mediante Triton / CUDA.     │
│  • Optimización de memoria compartida (*shared memory tiling*) y reducción de divergencia.      │
│                                                                                                 │
│  [MÓDULO 3: SURROGATE MODELING CON TRANSFORMERS / OPERADORES] (Innovación de Investigación)     │
│  • Transformer temporal / Structure-Preserving Transformer entrenado con datos de RK4.          │
│  • Inferencia autorregresiva de bloques macrotemporales ($\Delta T = 50 \cdot \Delta t$).       │
│  • Evaluación de Speedup vs. Violación de simetrías físicas frente a RK4 y Yoshida.             │
│                                                                                                 │
│  [MÓDULO 4: PROYECCIÓN SOCIAL Y VISUALIZADOR INTERACTIVO]                                       │
│  • Visor en tiempo real en OpenGL 3.3 (C++17, Shaders GLSL) o aplicación web interactiva.       │
│  • Demostración pedagógica: comparación visual instantánea entre la órbita RK4, Yoshida y ViT.   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Formulación Matemática del Problema Físico

### A. Sistema Dinámico: Problema Gravitacional de $N$-Cuerpos y Órbitas Caóticas
El sistema está gobernado por el Hamiltoniano clásico:

$$H(\mathbf{q}, \mathbf{p}) = \sum_{i=1}^N \frac{\|\mathbf{p}_i\|^2}{2m_i} - \sum_{i=1}^N \sum_{j < i} \frac{G m_i m_j}{\|\mathbf{q}_i - \mathbf{q}_j\| + \epsilon}$$

donde $\epsilon$ es el factor de suavizado (*softening parameter*) para evitar singularidades por colisión.

Las ecuaciones de movimiento canónicas de Hamilton son:

$$\frac{d\mathbf{q}_i}{dt} = \frac{\mathbf{p}_i}{m_i} = \mathbf{v}_i, \quad \frac{d\mathbf{p}_i}{dt} = -\nabla_{\mathbf{q}_i} V = \mathbf{F}_i$$

### B. El Integrador Simpléctico de 4.º Orden de Yoshida
A diferencia de RK4 (que no preserva el volumen simpléctico en el espacio de fases $d\mathbf{q} \wedge d\mathbf{p}$), el integrador de Yoshida descompone el operador de evolución temporal $e^{\Delta t(A+B)}$ utilizando coeficientes simétricos:

$$w_1 = \frac{1}{2 - 2^{1/3}}, \quad w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}}$$

$$c_1 = c_4 = \frac{w_1}{2}, \quad c_2 = c_3 = \frac{w_0 + w_1}{2}$$

$$d_1 = d_3 = w_1, \quad d_2 = w_0, \quad d_4 = 0$$

Permitiendo garantizar que el error de energía oscila alrededor de una cota sin presentar deriva secular a través de millones de pasos.

### C. El Transformer como Operador Sustituto (Surrogate Transformer)
El Transformer toma como entrada una secuencia histórica de estados en el espacio de fases:

$$\mathbf{S}_{t-k:t} = [(\mathbf{q}_1, \mathbf{p}_1), \dots, (\mathbf{q}_N, \mathbf{p}_N)]_{t-k:t} \in \mathbb{R}^{k \times N \times 6}$$

A través de capas de autoatención multi-cabeza:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

El modelo predice directamente el estado del sistema en $t + M$, saltándose $M$ pasos de integración numérica discreta.

---

## 3. Cronograma de Entregables Alíneado con el Sílabo UNSA

| Fase del Curso | Semana | Entregable Oficial del Sílabo | Entregable Técnico Concreto de tu Proyecto |
|---|:---:|---|---|
| **Fase 1: Avance Inicial** | **Semana 6** | • Revisión bibliográfica.<br>• Definición del problema físico.<br>• Primer informe de avances. | 1. Informe en formato IEEE (Abstract, Introducción y Marco Teórico).<br>2. Implementación completa y validación numérica de RK2 y RK4 en Python/C++.<br>3. Curvas de error de truncamiento y convergencia formal $\mathcal{O}(\Delta t^4)$. |
| **Fase 2: Núcleo Experimental** | **Semana 12** | • Metodología computacional.<br>• Resultados preliminares de simulación.<br>• Segundo informe de avances. | 1. Implementación del Integrador Simpléctico de Yoshida (orden 4).<br>2. Kernel de aceleración en GPU (Triton / CUDA) para cálculo de fuerzas.<br>3. Generación del dataset sintético y entrenamiento de la arquitectura Transformer.<br>4. Primer benchmark de deriva de energía ($\Delta E / E_0$) a lo largo de $10^5$ pasos temporales. |
| **Fase 3: Sustentación y Proyección** | **Semana 17** | • Artículo científico final completo.<br>• Repositorio de código documentado.<br>• Actividad de Responsabilidad Social. | 1. Artículo científico final con formato Scopus/IEEE (6 a 8 páginas).<br>2. Benchmark completo de Speedup GPU vs CPU y métricas de invariantes físicos.<br>3. Demo interactiva de Responsabilidad Social: Visualizador en tiempo real (OpenGL 3.3 o Web) con manual pedagógico para estudiantes. |

---

## 4. Estrategia del Entregable de Responsabilidad Social

El apartado 6.4.b del sílabo exige:
> *"La actividad de responsabilidad social universitaria estará orientada a la difusión de los principios de la física computacional, el pensamiento algorítmico, las simulaciones de sistemas dinámicos... mediante una charla, taller práctico, simulador web interactivo o aplicativo educativo..."*

### Propuesta de Ejecución (Alineada a tu CV)
Aprovechando tu experiencia en **C++17, OpenGL 3.3, React 19 y desarrollo frontend**:
* **Herramienta:** Crear un **Simulador Interactivo de Órbitas y Caos Determinista**.
* **Funcionalidad Didáctica:**
  * Permite al usuario mover planetas/cuerpos interactivos y cambiar el paso temporal $\Delta t$.
  * Muestra en tiempo real dos órbitas simultáneas: la órbita de **Runge-Kutta (con colapso por error numérico)** vs. la órbita **Simpléctica de Yoshida (estable en el tiempo)**.
  * Muestra gráficos interactivos en vivo del error de energía $\Delta E / E_0$.
* **Público Objetivo:** Estudiantes de los primeros ciclos de Ciencias de la Computación o colegios de Arequipa, explicando por qué los métodos numéricos son cruciales en astrofísica y desarrollo de motores de videojuegos.

---

## 5. Resumen de Decisiones Listas para Ejecutar

1. **Tu tema queda formalmente titulado como:**  
   *"Simulación y Aceleración en GPU del Problema de N-Cuerpos: Análisis Comparativo de Integradores Simplécticos vs. Runge-Kutta 4 y Modelos Basados en Transformers"*.
2. **Tu repositorio local de trabajo está centralizado en:**  
   `/home/antony/Documents/Fisica/0_PROYECTO/`.
3. **Tu siguiente paso práctico cuando decidas comenzar a codificar:**  
   Implementar el script base de RK4 clásico y Yoshida de 4.º orden en Python / C++ para generar los primeros gráficos de comparación de conservación de energía requeridos para la Fase 1.
