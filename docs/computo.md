# Plan de cómputo

Dónde corre cada parte de la reproducción (Fase 2) y con qué datos se decidió. Decisiones en `supuestos.md` (#3, #7, #8).

## Recursos disponibles (18 nov 2026)

| Recurso | Qué ofrece | Límite |
|---|---|---|
| Laptop | Ryzen 7 5700U, 8 núcleos / 16 hilos, 15 GB RAM, sin GPU NVIDIA, Python 3.10.11 | Solo CPU |
| Google Colab Pro | 396 CU disponibles (+100 CU/mes); GPUs T4, L4, A100 (G4 y H100 no disponibles) | **2 sesiones con GPU a la vez**; sin ejecución en segundo plano; sesiones de ≤24 h |
| Kaggle | GPU T4 x2 gratis, ejecución en segundo plano ("Save & Run All") | ~30 h de GPU por semana, 12 h por ejecución. **Pendiente:** la cuenta nueva no se pudo verificar (el teléfono ya está en otra cuenta de Kaggle); hay que usar la cuenta anterior. Opcional |
| AWS (crédito $130) | Cuota Standard de 32 vCPU (instancias CPU) | Cuota G and VT = 0: **sin GPU** ("not enough usage history") |
| GCP | — | `GPUS_ALL_REGIONS` = 0, no ajustable por falta de historial |

Presupuesto máximo: 400 CU de Colab + hasta 200 CU extra (2 × R$ 58), y hasta ~$30 del crédito de AWS para la extracción de tsfresh. Tope de bolsillo: $30–40.

## Mediciones en Colab (18 nov 2026)

Multiplicación de matrices 8192×8192 con `torch 2.14.1+cu130`, Python 3.13.16.

| GPU | VRAM | vCPU | RAM | FP32 (TFLOPS) | FP16 (TFLOPS) | CU/h medido |
|---|---|---|---|---|---|---|
| CPU estándar | — | 2 | 13.6 GB | — | — | — |
| T4 (RAM alta) | 15.6 GB | 8 | 55 GB | 3.5 | 21.4 | por medir |
| **L4** | 23.7 GB | 12 | 57 GB | 11.4 | ~57 | **1.94** (verificado con 1 y 2 sesiones) |
| A100 80 GB | 85 GB | 12 | 179 GB | 19.1 | 265.6 | por medir |

Referencia: la RTX 3060 del paper (§4.4) ronda ~25 TFLOPS FP16 (cifra teórica aproximada, no medida). La L4 es ~2.3× más rápida en matmul; falta confirmar con TabPFN real.

## Horas de cómputo de los autores

Suma de la columna `time` (fit + predict de TabPFN, 30 runs) en `references/TS2TabPFN/tsc/outputs/{c22,multi,tsfresh}.csv` y del tiempo de extracción en `time_tsfresh.csv`:

| Extractor | TabPFN (RTX 3060) | Extracción |
|---|---|---|
| catch22 | ~10 h | minutos |
| tsfresh | ~44 h | ~43 h con 16 hilos; **EigenWorms solo = 33.4 h**, MotorImagery 4.8 h, los otros 132 ≈ 4.5 h |
| MultiROCKET | ~199 h | ~0.5 h |

## Reparto

| Tarea | Dónde |
|---|---|
| Código, pruebas, análisis y reporte | Laptop |
| Extracción de tsfresh (livianos primero, EigenWorms al final) | AWS `c6i.8xlarge` (32 vCPU), on-demand, apagado automático al terminar |
| Extracción de catch22 y MultiROCKET | Colab, en la misma sesión que TabPFN (las features de MultiROCKET pesan GB) |
| TabPFN × 30 resamples, 3 extractores, 134 datasets | Colab: 2 sesiones L4 |
| Apoyo (p. ej., tsfresh + TabPFN) | Kaggle, en segundo plano |

## Estimación

- GPU: ~254 h equivalentes a la RTX 3060 → ~110–130 h de L4 si TabPFN acelera ~2× → **~215–250 CU**. Entra en las 396 CU disponibles.
- Tiempo: con 2 sesiones L4, **~2.5–3 días** de corridas tras la Fase 1 (~2–2.5 días si se suma Kaggle).
- Regla de corte: si a las ~12 h la proyección supera 600 CU o 48 h, se prioriza catch22 y tsfresh completos + MultiROCKET en los 7 de ECG, y el resto de MultiROCKET sigue con la recarga mensual.

## Requisitos para el código (Fase 1)

- **Ejecución reanudable:** la unidad de trabajo es (dataset, extractor, resample); cada resultado se guarda en Drive al terminar y el script salta lo ya hecho. Prueba que simula un corte.
- **Calibración:** 2–3 datasets con TabPFN real en T4, L4 y A100; medir tiempo y CU antes de lanzar todo.
- **Entorno:** Colab usa Python 3.13.16; el paper usó 3.10 (`env.yaml`). Decidir en Fase 1 si se fija un entorno virtual (`uv`) con las versiones del paper.
- Para evitar cortes: laptop sin suspensión y la pestaña de Colab excluida del ahorro de memoria de Chrome.
