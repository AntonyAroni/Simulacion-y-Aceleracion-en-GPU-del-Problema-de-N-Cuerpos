# Diagnóstico de Alineación: Perfil Profesional vs. Sílabo UNSA

**Estudiante:** Antony Aroni Jarata  
**Curso:** Física Computacional (E) — Semestre 2026-B, Escuela Profesional de Ciencia de la Computación, UNSA  
**Cátedra:** Dr. Danny Giancarlo Apaza Veliz  

---

## 1. Mapeo de Competencias del Perfil (CV de Antony)

El perfil técnico documentado en tu currículum vitae y publicaciones presenta una especialización de frontera muy clara en **tres pilares**:

```
                       ┌──────────────────────────────────────────────┐
                       │        PERFIL ANTONY ARONI JARATA            │
                       └──────────────────────┬───────────────────────┘
                                              │
            ┌─────────────────────────────────┼─────────────────────────────────┐
            ▼                                 ▼                                 ▼
   [IA & TRANSFORMERS]               [HPC & KERNELS GPU]               [VISIÓN 3D & GRÁFICA]
• Vision Transformers (ViT)       • Paralelismo Pipeline-Farm       • C++17 y OpenGL 3.3 Core
• Híbridos CNN-ViT & RT-DETR      • Kernels personalizados Triton   • Shaders GLSL (luces, física)
• Publicado IEEE SCCC 2025        • PyTorch JIT / TorchDynamo       • Reconstrucción 3D / Nubes .PLY
• Mecanismos de Autoatención      • Cuantización INT8 y CUDA        • Visor interactivo en C++
```

### Fortalezas Clave Diferenciales
1. **Modelos Basados en Atención y Visión:** Experiencia probada en autoatención (*self-attention*), atención cruzada (*cross-attention*), extracción espacial tipo convolucional y tokenización de secuencias.
2. **Computación de Alto Rendimiento (HPC) en GPU:** Dominio de paralelismo esquelético (*pipeline-farm*), optimización de memoria global/compartida, reducción de latencia de inferencia y compilación JIT con Triton.
3. **Desarrollo de Software Gráfico de Bajo Nivel:** Experiencia sólida en C++17 moderno, pipelines de renderizado con shaders GLSL en OpenGL 3.3, gestión matricial (GLM) y visualización tridimensional en tiempo real.

---

## 2. Anatomía del Sílabo de Física Computacional (UNSA 2026-B)

El sílabo impartido por el **Dr. Danny Giancarlo Apaza Veliz** (Doctor en Física) establece competencias formativas muy específicas de física clásica y moderna que determinan el criterio de calificación del jurado:

### Unidades y Temas Clave del Sílabo
* **Primera Unidad (Semanas 1 a 6):**
  * Análisis numérico fundamental: Propagación de errores de truncamiento y redondeo (Tema 03).
  * Solución de ecuaciones no lineales: Bisección, Falsa Posición, Newton-Raphson, Secante aplicados al equilibrio físico (Temas 05 y 06).
  * EDOs en cinemática y dinámica: Euler, Euler-Cromer con fricción (Tema 07).
  * Métodos de Runge-Kutta: RK2 y RK4, análisis de orden y convergencia (Tema 08).
  * *Hito 1 (Semana 6):* Primer examen y evaluación de avances Fase 1 del Proyecto Formativo y Responsabilidad Social.
* **Segunda Unidad (Semanas 7 a 12):**
  * Osciladores armónicos y no lineales resueltos con RK4 (Tema 10).
  * **Gravitación y dinámica orbital del problema de $N$-cuerpos** (Tema 11).
  * Fluidos simplificados y métodos estocásticos termodinámicos (Temas 12 y 13).
  * **Autómatas celulares y física discreta:** Reglas de Wolfram, Conway (Game of Life), tráfico vehicular (Nagel-Schreckenberg), difusión de calor y percolación (Temas 14 y 15).
  * *Hito 2 (Semana 12):* Segundo examen y evaluación Fase 2 del Proyecto Formativo.
* **Tercera Unidad (Semanas 13 a 17):**
  * Óptica computacional: Difracción, interferencia y Huygens por superposición discreta (Tema 17).
  * Relatividad especial: Transformaciones de Lorentz numéricas (Tema 18).
  * **Mecánica cuántica computacional:** Solución numérica de la **ecuación de Schrödinger 1D** mediante el **método de disparo (*shooting method*)** acoplando RK4 con búsqueda de raíces (Newton-Raphson o secante) para pozos de potencial (Tema 20).
  * *Hito 3 (Semana 17):* Tercer examen y sustentación final del Artículo Científico y Proyecto de Responsabilidad Social.

### Exigencias Rígidas del Docente y Criterios de Evaluación
Según el apartado 8.1 y 6.4 del sílabo, el docente evalúa:
1. *Comprensión y dominio de los fundamentos físicos y matemáticos.*
2. *Estabilidad numérica, orden de convergencia y propagación de errores acumulados.*
3. *Conservación de leyes fundamentales (energía, momento lineal, momento angular, unitariedad cuántica).*
4. *Análisis comparativo de complejidad algorítmica y eficiencia computacional ($O(N)$, $O(N^2)$, Speedup).*
5. *Estructura formal de artículo científico (formato IEEE / Scopus).*
6. *Entregable interactivo de Responsabilidad Social (difusión hacia la comunidad universitaria o colegios).*

---

## 3. Identificación de Riesgos Académicos y "Trampas" a Evitar

> [!CAUTION]
> **La Trampa de la "Caja Negra" (Black-Box ML):**  
> Si se presenta una red neuronal convencional (MLP, LSTM o ViT vanilla) entrenada para "predecir la siguiente posición de partículas" minimizando únicamente el error cuadrático medio (MSE), el proyecto corre **alto riesgo de ser rechazado u observado severamente** por la cátedra de Física. Los docentes de física computacional rechazan modelos que ignoran las leyes de conservación, no analizan el error de truncamiento numérico del método discreto y no demuestran ventajas matemáticas fundamentadas frente a RK4.

> [!IMPORTANT]
> **El Enfoque Ganador (Scientific Machine Learning / Operadores / HPC):**  
> Para integrar Inteligencia Artificial / Transformers de manera impecable y obtener calificación sobresaliente:
> 1. **La línea base DEBE ser el método clásico del sílabo:** RK4, integrador simpléctico o método de disparo clásico (Shooting RK4 + Secante).
> 2. **La red DEBE operar como un Operador Neural o Physics-Informed Transformer (PINNsFormer, Operator Learning o Surrogate Multistep):** Aprendiendo el flujo del sistema dinámico o resolviendo la EDO incorporando la física en la función de pérdida o en la arquitectura.
> 3. **Las métricas DEBEN ser físicas y computacionales:** No solo MSE o $R^2$, sino deriva del Hamiltoniano $\Delta E / E_0$, invariantes simplécticos, tiempo por paso de tiempo vs. GFLOPS, y escalamiento en GPU.

---

## 4. Matriz de Compatibilidad: Tus Habilidades vs. Contenidos del Curso

| Área de tu Perfil | Contenido del Sílabo Directamente Aplicable | Forma de Integración en el Proyecto |
|---|---|---|
| **Vision Transformers & Atención** | Temas 10, 11 (Dinámica temporal no lineal / $N$-cuerpos) | Modelo Transformer autorregresivo / Operador de Fourier (FNO) o PINNsFormer para propagación en horizontes temporales largos. |
| **Kernels Triton & GPU CUDA** | Tema 11 ($N$-cuerpos) y Temas 14-15 (Autómatas celulares 2D) | Kernel Triton para cálculo directo de interacciones $O(N^2)$ (shared memory tiling) o transiciones de autómatas celulares tensoriales masivos ($10000 \times 10000$). |
| **C++17 & Shaders GLSL OpenGL 3.3** | Tema 11 (Órbitas), Tema 15 (Tráfico/Percolación) | Visualizador 3D interactivo en tiempo real con trails de órbitas, esferas texturizadas e iluminación puntual (cumple 100% Responsabilidad Social). |
| **Algoritmos Numéricos (RK4, Secante)** | Temas 08, 10, 20 (RK4, EDOs, Shooting method Schrödinger) | Ground Truth para validación física estricta, generación de trayectorias analíticas y benchmarking de convergencia formal. |
| **Redacción Científica (IEEE / Scopus)** | Requisito 6.4 (Artículo científico) | Reutilización directa del estándar IEEE de tu ponencia en SCCC 2025 para el informe final del curso. |
