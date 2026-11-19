# Datasets

## Paso 1: reproducción de TS2TabPFN

Los autores empaquetaron los datos en Google Drive (ver `references/TS2TabPFN/README.md`):

- TSC (UCR/UEA): `gdown 1p8D-haDDwPKloKMBQCQHgENYHVsjixR4`
- TSER (TSML Extended): `gdown 1Bg_KHSv77eMbqDyElPohJ1dIam1CkS_V`

Descargado el 18 nov 2026: `tsc_data.zip`, 1.29 GB (4.1 GB descomprimido, 158 carpetas con `<dataset>_TRAIN.ts` y `<dataset>_TEST.ts`), SHA-256 `b927bcbc3be117a1dd2c643bf3c6eabbbf2c4b43ceca3641fb27209b1f61bab7`. Los 134 datasets del paper cargan sin errores, con longitud igual y sin NaN (`results/data_check.csv`, decisión #14).

Alternativa sin el zip completo: descargar datasets individuales con `aeon.datasets.load_classification("ECG200")`.

Fuentes originales: <https://www.timeseriesclassification.com> y <https://zenodo.org/records/11236865>.

### Datasets exactos del paper (verificado el 18 nov 2026)

Fuente: tablas finales `references/TS2TabPFN/tsc/results/Tab_{c22,multi,tsfresh}.csv` (una fila por dataset, 30 columnas = resamples 0-29).

- **TSC: 134 = 112 univariados (UCR) + 22 multivariados (UEA).** Es el conjunto común a los tres extractores (= `Tab_tsfresh.csv`). `Tab_c22.csv` trae 5 extra (Crop, DuckDuckGeese, FaceDetection, PenDigits, PEMS-SF) y `Tab_multi.csv` 1 extra (PEMS-SF); todos salvo PEMS-SF están en `invalid_datasets` de `config.yaml`. PEMS-SF no está en `Tab_tsfresh.csv`, así que queda fuera. *(Corregido el 18 nov 2026: antes decía 111 + 23 por contar PEMS-SF.)*
- Multivariados: ArticularyWordRecognition, AtrialFibrillation, BasicMotions, Cricket, ERing, EigenWorms, Epilepsy, EthanolConcentration, FingerMovements, HandMovementDirection, Handwriting, Heartbeat, LSST, Libras, MotorImagery, NATOPS, PhonemeSpectra, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, StandWalkJump, UWaveGestureLibrary.
- Univariados: los 112 restantes de `Tab_tsfresh.csv` (incluye ECG200, ECG5000, ECGFiveDays, TwoLeadECG, CinCECGTorso, NonInvasiveFetalECGThorax1/2 y Fungi).
- Lista completa con los tiempos de los autores: `configs/datasets_tsc.csv` (generada por `scripts/build_dataset_config.py`).
- **TSER: 55** (`tser/results/Tab_tsfresh.csv`; `Tab_c22` trae 58 y `Tab_multi` 56).

### Costo de cómputo reportado por los autores (RTX 3060)

Suma de la columna `time` (fit + predict de TabPFN, 30 runs) en `tsc/outputs/*.csv` y `tser/outputs/*.csv`, más tiempos de extracción en `time_*.csv`:

| Tarea | catch22 | MultiROCKET | tsfresh |
|---|---|---|---|
| TSC (134) | ~10 h | ~199 h | ~44 h + ~43 h de extracción (16 núcleos) |
| TSER (55) | ~6 h | ~57 h | ~9 h |

Total aproximado: **~370 h de GPU**. No es viable completo en Colab gratuito.

## Paso 2: ECG (recomendados por Rafael, en orden de relevancia)

| # | Dataset | Por qué | Enlace |
|---|---|---|---|
| 1 | **PTB-XL** | Clásico de la literatura | <https://physionet.org/content/ptb-xl/1.0.1/> (Rafael enlazó 1.0.1; la última es 1.0.3) |
| 2 | **Brugada-HUCA** | Anota el síndrome de Brugada, enfermedad rara poco explorada | <https://physionet.org/content/brugada-huca/1.0.0/> |
| 3 | **CODE-15%** | Brasileño; rótulos de cardiopatías y de Chagas (autodeclarados) | <https://zenodo.org/records/4916206> · rótulos Chagas: <https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip> |
| 3b | **CODE-Test** | Conjunto de test de CODE-15 (sin rótulos de Chagas) | <https://zenodo.org/records/3765780> |
| 4 | **MIT-BIH Arrhythmia** | Otro clásico | <https://www.physionet.org/content/mitdb/1.0.0/> |
| 5 | **ECG Fragment** | Subconjunto de MIT-BIH clasificado por riesgo; difícil | <https://physionet.org/content/ecg-fragment-high-risk-label/1.0.0/> |
| 6 | **MIMIC-IV-ECG** | ~800 000 registros de ~160 000 pacientes; dejar para el final | <https://physionet.org/content/mimic-iv-ecg/1.0/> (⚠️ en el correo el enlace apuntaba por error a HUCA) |

## Decisión inicial

Empezar con **PTB-XL** porque:

- Tiene folds oficiales (`strat_fold`) que evitan leakage entre pacientes.
- Ofrece versión a 100 Hz (más liviana) además de 500 Hz.
- Hay notebooks de referencia en `references/ECG_Feature_Extraction/` que ya lo usan.

Después: HUCA (pequeño y con una clase rara) y CODE-15 (Chagas).

## Reglas

- Los datos van en `data/raw/<dataset>/` y **no se suben a Git**.
- Anotar versión, fecha de descarga y checksum en `data/raw/<dataset>/SOURCE.md` (este archivo sí se puede versionar si se ajusta el `.gitignore`).
- Algunos datasets de PhysioNet piden cuenta o acuerdo de uso. Revisar la licencia antes de descargar.
