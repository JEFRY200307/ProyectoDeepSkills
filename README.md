# ProyectoDeepSkills: TS2TabPFN aplicado a ECG

Reproducción del paper **TS2TabPFN** (Merlin y Silva, 2026) y extensión a señales de ECG, comparando la extracción genérica de features (catch22, tsfresh, MultiROCKET) contra extractores específicos de ECG, con **TabPFN 2.5** como clasificador.

Este proyecto sirve a dos fines:

1. **Investigación con el ICMC-USP** (Prof. Diego Furtado Silva): primero reproducir un experimento del artículo y, si funciona, pasar a señales fisiológicas.
2. **Proyecto final del curso AI for Engineers (DeepSkill)**: reproducción de un paper usando agentes de código.

## Estado

| Fase | Descripción | Estado |
|---|---|---|
| 0 | Base del proyecto: estructura, papers, contexto, datasets documentados | ✅ Hecha |
| 1 | Lectura del paper y del código de referencia, entorno en Colab | En curso |
| 2 | Reproducir el experimento de clasificación completo: 3 extractores × 134 datasets × 30 resamples | Pendiente |
| 3 | Pipeline de ECG: PTB-XL + extractores específicos | Pendiente |
| 4 | Comparación genérico vs. específico + reporte | Pendiente |

Detalle en [docs/plan-fases.md](docs/plan-fases.md).

## Estructura

```
ProyectoDeepSkills/
├── AGENTS.md            Contexto y reglas para agentes de código
├── CLAUDE.md            Apunta a AGENTS.md
├── paper/               PDFs de los artículos
├── docs/                Contexto, resumen de papers, datasets, plan y supuestos
│   └── correos/         Capturas de los correos originales
├── data/
│   ├── raw/             Datos descargados (no se versionan)
│   └── processed/       Features extraídas (no se versionan)
├── src/                 Código del proyecto
├── tests/               Pruebas automatizadas
├── notebooks/           Exploración
├── results/             Tablas y gráficos finales
└── references/          Repos de terceros clonados (no se versionan, ver docs/referencias.md)
```

## Documentos

- [Contexto del proyecto](docs/contexto.md): correos, personas y objetivo
- [Resumen de papers](docs/papers.md): TS2TabPFN y ECGomics
- [Datasets de ECG](docs/datasets.md): lista recomendada por Rafael da Costa Silva
- [Referencias de código](docs/referencias.md): repos clonados y commits
- [Plan por fases](docs/plan-fases.md)
- [Plan de cómputo](docs/computo.md): Colab Pro, AWS, Kaggle y mediciones
- [Guía de ejecución](docs/ejecucion.md): código, entorno local, Colab y AWS
- [Registro de supuestos](docs/supuestos.md)

## Papers

- Merlin, G. C.; Silva, D. F. *TS2TabPFN: Time Series Classification and Extrinsic Regression through Feature Extraction and a Tabular Foundation Model*. arXiv:2608.04174, 2026. Código: <https://github.com/gabrielcmerlin/TS2TabPFN>
- Zhang, D. et al. *ECGomics: An Open Platform for AI-ECG Digital Biomarker Discovery*. Health Data Science 6 (2026): 0427. <https://doi.org/10.34133/hds.0427>
