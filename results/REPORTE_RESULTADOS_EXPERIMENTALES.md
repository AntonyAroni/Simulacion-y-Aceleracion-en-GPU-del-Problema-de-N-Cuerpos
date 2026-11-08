# Reporte de Resultados Experimentales — Proyecto Final de Física Computacional

**Dispositivo de Aceleración Detectado:** `NVIDIA GeForce RTX 4050 Laptop GPU`  
**Arquitectura de Software:** Modular SOLID + Esqueleto Paralelo Pipeline-Farm sobre CUDA Streams  
**Sistema Físico Evaluado:** Problema Gravitacional Coreográfico de 3 Cuerpos (Figura-8 de Chenciner-Montgomery) y Cúmulos Globulares de Plummer ($N$-Cuerpos)  

---

## 1. Comparativa de Conservación de Invariantes Físicos y Eficiencia ($t \in [0, 18.0]$, $\Delta t = 0.003$)

| Método Numérico / Operador | Simpléctico | Orden | Deriva Media $|\Delta E / E_0|$ | Deriva Máxima $|\Delta E / E_0|$ | Deriva Momento Angular $|\Delta L|$ | Tiempo Ejecución (ms) | Pasos Ejecutados |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Euler-Cromer** | Sí | 1 | `1.729e-03` | `3.305e-03` | `1.566e-15` | `95.08` | `1200` |
| **Velocity Verlet (Orden 2)** | Sí | 2 | `5.740e-05` | `1.328e-04` | `3.109e-15` | `180.17` | `1200` |
| **Runge-Kutta 4 (RK4)** | No | 4 | `2.639e-08` | `5.662e-08` | `2.827e-09` | `447.05` | `1200` |
| **Yoshida Simpléctico 4.º Orden** | Sí | 4 | `4.206e-08` | `9.836e-08` | `4.774e-15` | `303.86` | `1200` |
| **Symplectic Transformer (Salto 10x)** | Sí | 4 | `8.333e-16` | `1.208e-15` | `9.330e-06` | `1027.19` | `600` |

---

## 2. Convergencia del Symplectic Transformer con Optimizador de 2.º Orden SOAP

| Época | Pérdida Total Adaptativa | MSE Posición $\mathcal{L}_q$ | MSE Velocidad $\mathcal{L}_v$ | Residuo Hamiltoniano $\mathcal{L}_H$ | Tasa de Aprendizaje (LR) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `0.3650` | `1.212e-07` | `7.492e-07` | `2.389e-06` | `2.50e-03` |
| 21 | `-4.8663` | `2.531e-10` | `6.125e-09` | `1.245e-08` | `2.11e-03` |
| 41 | `-8.6770` | `6.077e-11` | `3.015e-09` | `1.545e-08` | `1.23e-03` |
| 61 | `-10.4779` | `1.533e-11` | `7.087e-10` | `2.366e-09` | `3.75e-04` |
| 80 | `-10.8516` | `1.414e-11` | `5.608e-10` | `1.339e-09` | `5.00e-05` |

---

## 3. Escalabilidad en GPU (Cúmulo de Plummer $O(N^2)$ — Lote Paralelo $B = 4$)

| Número de Cuerpos ($N$) | Interacciones por Paso ($B \cdot N^2$) | Tiempo CPU (ms) | Tiempo GPU RTX 4050 (ms) | Speedup GPU vs CPU ($\times$) | Throughput GPU (GFLOPS) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **16** | `1,024` | `3.93` | `7.81` | **`0.50x`** | **`0.28`** |
| **64** | `16,384` | `13.95` | `9.38` | **`1.49x`** | **`3.77`** |
| **128** | `65,536` | `32.77` | `10.47` | **`3.13x`** | **`13.52`** |
| **256** | `262,144` | `42.17` | `9.05` | **`4.66x`** | **`62.54`** |
| **512** | `1,048,576` | `281.49` | `13.39` | **`21.02x`** | **`169.11`** |
| **1024** | `4,194,304` | `2460.83` | `144.07` | **`17.08x`** | **`62.89`** |
| **2048** | `16,777,216` | `12186.56` | `768.00` | **`15.87x`** | **`47.19`** |

---

## 4. Figuras Científicas Generadas (300 DPI)

1. `results/figures/fig1_orbitas_y_espacio_fases.png`: Trayectorias en el plano $(q_x, q_y)$ y retrato en el espacio de fases $(q_x, p_x)$.
2. `results/figures/fig2_conservacion_hamiltoniano.png`: Evolución temporal del Hamiltoniano $H(t)$ y deriva relativa $|\Delta E / E_0|$ en escala logarítmica.
3. `results/figures/fig3_optimizador_soap_y_escalabilidad_gpu.png`: Curvas de convergencia con el optimizador SOAP y Speedup/GFLOPS en GPU.
