# AGENTS.md

Contexto para cualquier agente de código (OpenCode, Claude Code, Codex) que trabaje en este repositorio.

## Objetivo

1. Reproducir TS2TabPFN (arXiv:2608.04174): extracción de features de series temporales + TabPFN 2.5 como clasificador.
2. Extenderlo a ECG: comparar extractores genéricos (catch22, tsfresh, MultiROCKET) contra extractores específicos de ECG (NeuroKit2 / `ECG_Feature_Extraction`) sobre PTB-XL.

Antes de cualquier tarea, lee `docs/contexto.md`, `docs/papers.md` y `docs/plan-fases.md`.

## Reglas

- **No inventes hiperparámetros ni detalles del paper.** Si el paper o el código de referencia no los dan, pregunta y registra la decisión en `docs/supuestos.md`.
- **Cita la fuente** de cada decisión: sección del paper, archivo y línea de `references/TS2TabPFN/`.
- **`references/` es solo lectura.** No modifiques los repos clonados; copia o adapta a `src/` indicando el origen.
- **Una fase a la vez.** Presenta un plan y espera aprobación antes de escribir código de una fase nueva.
- **Cada función nueva lleva su prueba** en `tests/` (pytest).
- **Semillas fijas** (`seed = 0` como en el `config.yaml` de referencia) y versiones de librerías anotadas.
- **Separación train/test estricta.** Nunca ajustes nada (normalización, selección de features) con datos de test. Ojo: el paper concatena train y test para extraer con tsfresh; documenta si se replica o no.
- **PTB-XL:** usa los folds oficiales (`strat_fold`: 1-8 train, 9 validación, 10 test) y separa por paciente.
- **Datos fuera de Git.** Nada de `data/raw`, `data/processed`, pesos de modelos ni tokens (Hugging Face) en commits.
- **Resultados en `results/`** como CSV + gráfico, con el commit y la configuración que los generó.
- Desconfía de resultados demasiado buenos: revisa leakage antes de reportarlos.

## Entorno

- Python 3.10 con las versiones exactas del `env.yaml` de referencia, instaladas con `uv` (`requirements.txt`). Decisión #9 en `docs/supuestos.md`.
- Librerías clave: `aeon`, `tsfresh`, `tabpfn` (2.5, pesos vía Hugging Face), `neurokit2`, `wfdb`, `pywavelets`.
- Hardware de referencia del paper: RTX 3060 12 GB. El equipo local no tiene GPU: el código y las pruebas se escriben en local; TabPFN corre en Google Colab Pro (máximo 2 sesiones GPU a la vez), la extracción de tsfresh en AWS (CPU), y los datos y resultados van a Google Drive. Ver `docs/computo.md`.
- Colab usa Python 3.13; el código debe ser reanudable (un resultado por dataset × extractor × resample) porque las sesiones se cortan.
- Los datasets se usan completos (sin submuestrear instancias) y con 30 resamples, como el paper. Alcance de la reproducción en `docs/plan-fases.md` (Fase 2).

## Idioma

Documentación en español. Código, nombres de variables y commits en inglés.
