# Plan por fases

Ciclo en cada fase: **planear → ejecutar → verificar**. Un commit al cerrar cada fase.

**Dónde se corre cada cosa:** el código y las pruebas se escriben en local. TabPFN corre en Google Colab Pro (2 sesiones L4), la extracción de tsfresh en AWS (CPU) y Kaggle sirve de apoyo. Los datos, features y resultados se guardan en Google Drive. Detalle y mediciones en [computo.md](computo.md).

## Fase 0: Base del proyecto ✅

- Estructura de carpetas, `AGENTS.md`, `.gitignore`.
- Papers en `paper/`, contexto de correos, datasets y referencias documentados.

## Fase 1: Lectura y entorno

- ~~Leer el PDF de TS2TabPFN y contrastarlo con `docs/papers.md`.~~ Hecho el 3 nov 2026.
- ~~Identificar los datasets exactos y el costo de cómputo del paper.~~ Hecho el 18 nov 2026 (`docs/datasets.md`).
- ~~Recorrer `references/TS2TabPFN/code/` y documentar el flujo.~~ Hecho: `docs/ejecucion.md` y decisiones #9–#15.
- ~~`requirements.txt` con las versiones exactas del `env.yaml` (Python 3.10, `uv`).~~ Hecho. `neurokit2` y `wfdb` se agregan en la Fase 3.
- ~~Código en `src/ts2tabpfn/` (datos, features, modelo, ejecución reanudable) con pruebas.~~ Hecho: 25 pruebas pasan; la de TabPFN real espera el `HF_TOKEN`.
- ~~Notebook de Colab y script de tsfresh para AWS.~~ Hecho: `notebooks/colab_runner.ipynb`, `scripts/extract_tsfresh.py`.
- ~~Descargar y verificar los datos de los autores.~~ Hecho: 134/134, sin NaN (decisión #14).
- **Pendiente — verificación en Colab:** `HF_TOKEN` + términos de TabPFN 2.5, una predicción con GPU y calibración con 2–3 datasets en T4, L4 y A100 (tiempo y CU reales).

## Fase 2: Reproducir el experimento de clasificación completo (pedido de Diego)

El paper tiene dos experimentos: clasificación (TSC, 134 datasets) y regresión (TSER, 55). Se replica **TSC completo**: **3 extractores (catch22, tsfresh, MultiROCKET) × 134 datasets × 30 resamples** (decisión #7).

Orden de ejecución, pensado para tener un primer resultado pronto:
1. catch22 en los 134 datasets (~10 h en la RTX 3060 de los autores).
2. Los 3 extractores en los 7 datasets de ECG de UCR: `ECG200`, `ECG5000`, `ECGFiveDays`, `TwoLeadECG`, `CinCECGTorso`, `NonInvasiveFetalECGThorax1` y `NonInvasiveFetalECGThorax2`.
3. tsfresh en el resto, a medida que AWS entrega las features.
4. MultiROCKET en el resto (~199 h en la RTX 3060; es lo que marca el ritmo).

- Cada dataset se usa completo y con 30 resamples (decisiones #4 y #5).
- Estimación: ~2–2.5 días con 2 sesiones L4 + Kaggle y ~215–250 CU (`computo.md`).
- **Verificación:** comparar la accuracy, dataset por dataset, con `references/TS2TabPFN/tsc/results/Tab_{c22,multi,tsfresh}.csv`, y un test de Wilcoxon por extractor para verificar que no hay diferencia significativa con los autores. Las diferencias se explican en `docs/supuestos.md`.
- Entregable: tablas y gráficos en `results/`, y correo a Diego con el avance (primero catch22 + ECG, después el resto).

## Fase 3: Pipeline de ECG con PTB-XL (objetivo principal de la investigación)

- PTB-XL completo a 100 Hz con los folds oficiales (1-8 train, 9 validación, 10 test). Tarea inicial: superclases diagnósticas (NORM, MI, STTC, CD, HYP).
- Extractores genéricos (los de Fase 2) y específicos (NeuroKit2 / `ECG_Feature_Extraction`; ECGomics si es abierto).
- tsfresh se extrae por lotes guardados en Drive, por su costo.
- **Verificación:** pruebas de forma de datos, ausencia de pacientes compartidos entre folds y métricas sobre el fold 10.

## Fase 4: Comparación y reporte

- Genérico vs. específico vs. combinado: accuracy/AUC, tiempo de extracción, interpretabilidad.
- Repetir en HUCA y CODE-15 si el tiempo alcanza.
- `report.html` en la raíz y README actualizado.
- Entrega: enlace a Diego (USP) y respuesta al correo de DeepSkill para el certificado.

## Fuera de alcance (por ahora)

- Regresión (TSER, 55 datasets): ~72 h de GPU en la RTX 3060 de los autores. Si sobra tiempo, la primera candidata son los 3 datasets BIDMC de TSER (señales fisiológicas).
