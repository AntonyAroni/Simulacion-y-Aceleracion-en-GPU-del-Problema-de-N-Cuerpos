# Proyecto Final de Física Computacional (UNSA 2026-B)
## Simulación y Aceleración en GPU del Problema de $N$-Cuerpos: Integradores Simplécticos (Yoshida 4.º Orden) vs. Runge-Kutta 4 y Operadores Basados en Transformers

**Estudiante:** Antony Aroni Jarata  
**Programa:** Ciencia de la Computación, 10.º Semestre — Universidad Nacional de San Agustín (UNSA)  
**Docente del Curso:** Dr. Danny Giancarlo Apaza Veliz  
**Hardware de Experimentación:** NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, CUDA / PyTorch 2.15)  

---

### Estructura Modular del Proyecto (Principios SOLID + Pipeline-Farm)

```
0_PROYECTO/
├── README.md                                  # Índice general y guía de ejecución
├── 01_DIAGNOSTICO_PERFIL_Y_SILABO.md          # Cruce exhaustivo CV vs. Sílabo UNSA
├── 02_ESTADO_DEL_ARTE_Y_PAPERS.md             # Taxonomía de papers base (2019–2026)
├── 03_MATRIZ_COMPARATIVA_PROS_CONTRAS.md      # Evaluación multidimensional de rutas
├── 04_PROPUESTA_Y_HOJA_DE_RUTA_FINAL.md       # Formulación matemática y cronograma
├── 05_OPTIMIZADORES_Y_ARQUITECTURA_SOTA.md    # Papers SOTA: SOAP (2025), Pérdida Adaptativa y Variedad Simpléctica
├── run_benchmark_pipeline.py                  # Orquestador maestro de entrenamiento y benchmarks en GPU
├── src/                                       # Paquete de software modular bajo principios SOLID
│   ├── core/
│   │   ├── config.py                          # Dataclasses inmutables de configuración
│   │   ├── state.py                           # PhaseSpaceState (soporte tensorial [B, N, 3] en GPU)
│   │   └── interfaces.py                      # Contratos abstractos (IForceKernel, IIntegrator, ISurrogateModel)
│   ├── physics/
│   │   ├── gravity_kernel.py                  # Kernel Gravitacional O(N^2) con ruta dual: Exacta FP64 y GEMM Tensor-Core
│   │   ├── invariants.py                      # Auditoría de Hamiltoniano (H=T+V), Momento Lineal (P), Angular (L) y Virial
│   │   └── initial_conditions.py              # Órbita Figura-8 (Chenciner-Montgomery) y Cúmulo Globular de Plummer
│   ├── integrators/
│   │   ├── euler.py                           # Euler Explícito y Euler-Cromer (Tema 07 Sílabo)
│   │   ├── runge_kutta.py                     # RK2 y RK4 Clásico (Temas 08 y 10 Sílabo)
│   │   └── symplectic.py                      # Velocity Verlet (Orden 2) y Yoshida Simpléctico 4.º Orden (Tema 11)
│   ├── models/
│   │   ├── embeddings.py                      # Tokenizador Fase-Espacio + Activación Espectral Wavelet-Fourier (PINNsFormer)
│   │   ├── attention.py                       # Autoatención Guiada por Potencial Gravitacional (SympGNN / Transolver)
│   │   ├── projection.py                      # Proyección Diferenciable en la Variedad Iso-Energética y de Momento Angular
│   │   └── transformer.py                     # SymplecticTransformerOperator y adaptador Liskov SurrogateTransformerIntegrator
│   ├── optimization/
│   │   ├── losses.py                          # SelfAdaptivePhysicsLoss (Incertidumbre Homocedástica Aprendible)
│   │   └── soap_optimizer.py                  # Optimizador de 2.º Orden SOAP (Shampoo + Adam en Eigenbasis, ICLR/NeurIPS 2025)
│   ├── pipeline/
│   │   └── pipeline_farm.py                   # Arquitectura Esquelética Pipeline-Farm sobre CUDA Streams (Emitter-Farm-Collector)
│   └── visualization/
│       └── plotter.py                         # Generador de gráficas de estándar IEEE/Scopus a 300 DPI
├── tests/
│   └── test_solid_suite.py                    # Suite de pruebas unitarias de conservación física, contratos SOLID y GPU
└── results/
    ├── REPORTE_RESULTADOS_EXPERIMENTALES.md   # Tablas cuantitativas generadas por la RTX 4050
    └── figures/                               # Figuras científicas (300 DPI)
        ├── fig1_orbitas_y_espacio_fases.png
        ├── fig2_conservacion_hamiltoniano.png
        └── fig3_optimizador_soap_y_escalabilidad_gpu.png
```

---

### Comandos Rápidos de Ejecución y Verificación

1. **Ejecutar la suite de pruebas unitarias (Verificación SOLID y Leyes de Conservación):**
   ```bash
   PYTHONPATH=0_PROYECTO python3 -m unittest discover -s 0_PROYECTO/tests -p "test_*.py" -v
   ```

2. **Ejecutar el pipeline completo de entrenamiento SOAP, benchmarks de invariantes y gráficas:**
   ```bash
   python3 0_PROYECTO/run_benchmark_pipeline.py
   ```
