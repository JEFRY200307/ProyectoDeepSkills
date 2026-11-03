# Datasets

## Paso 1: reproducción de TS2TabPFN

Los autores empaquetaron los datos en Google Drive (ver `references/TS2TabPFN/README.md`):

- TSC (UCR/UEA): `gdown 1p8D-haDDwPKloKMBQCQHgENYHVsjixR4`
- TSER (TSML Extended): `gdown 1Bg_KHSv77eMbqDyElPohJ1dIam1CkS_V`

Alternativa sin el zip completo: descargar datasets individuales con `aeon.datasets.load_classification("ECG200")`.

Fuentes originales: <https://www.timeseriesclassification.com> y <https://zenodo.org/records/11236865>.

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
