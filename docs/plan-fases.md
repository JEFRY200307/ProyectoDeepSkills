# Plan por fases

Ciclo en cada fase: **planear → ejecutar → verificar**. Un commit al cerrar cada fase.

## Fase 0: Base del proyecto ✅

- Estructura de carpetas, `AGENTS.md`, `.gitignore`.
- Papers en `paper/`, contexto de correos, datasets y referencias documentados.

## Fase 1: Lectura y entorno

- Leer el PDF de TS2TabPFN y contrastarlo con `docs/papers.md`.
- Recorrer `references/TS2TabPFN/code/` y documentar el flujo (carga → extracción → TabPFN → métricas).
- Crear el entorno (Python 3.11, aeon, tsfresh, tabpfn, neurokit2, wfdb) y acceder a los pesos de TabPFN 2.5.
- **Verificación:** `import tabpfn` y una predicción sobre un dataset de juguete.

## Fase 2: Reproducir un experimento de TS2TabPFN (pedido de Diego)

- Subconjunto de UCR con datasets de ECG: `ECG200`, `ECGFiveDays`, `TwoLeadECG`, `ECG5000`.
- Los 3 extractores: catch22, tsfresh, MultiROCKET.
- **Verificación:** accuracy comparada con `references/TS2TabPFN/tsc/results/`. Diferencias explicadas en `docs/supuestos.md`.
- Entregable: tabla en `results/` y correo a Diego con el avance.

## Fase 3: Pipeline de ECG con PTB-XL

- Descarga a 100 Hz, folds oficiales, tarea inicial: superclases diagnósticas (NORM, MI, STTC, CD, HYP).
- Extractores genéricos (los de Fase 2) y específicos (NeuroKit2 / `ECG_Feature_Extraction`; ECGomics si es abierto).
- **Verificación:** pruebas de forma de datos, ausencia de pacientes compartidos entre folds, métricas sobre el fold 10.

## Fase 4: Comparación y reporte

- Genérico vs. específico vs. combinado: accuracy/AUC, tiempo de extracción, interpretabilidad.
- Repetir en HUCA y CODE-15 si el tiempo alcanza.
- `report.html` en la raíz y README actualizado.
- Entrega: enlace a Diego (USP) y respuesta al correo de DeepSkill para el certificado.
