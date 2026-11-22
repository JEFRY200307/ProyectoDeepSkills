# Guía de ejecución

Cómo correr la reproducción de clasificación (Fase 2). Plan de cómputo y costos en [computo.md](computo.md).

## Estructura del código

| Archivo | Qué hace | Origen |
|---|---|---|
| `src/ts2tabpfn/data.py` | Carga `.ts`, resamples estratificados (copia literal), descripción de cada dataset | `code/utils/utils_data.py` |
| `src/ts2tabpfn/features.py` | catch22, tsfresh, MultiROCKET y caché de features en `.npz` | `code/utils/utils_train.py:22-82` |
| `src/ts2tabpfn/model.py` | `TabPFNClassifier` + `ManyClassClassifier(alphabet_size=10)` y métricas | `code/utils/utils_train.py:136-218` |
| `src/ts2tabpfn/runner.py` | Bucle reanudable: un JSON por (extractor, dataset, resample); reparto en sesiones | `code/main.py` |
| `configs/datasets_tsc.csv` | Los 134 datasets con los tiempos de los autores | `scripts/build_dataset_config.py` |
| `scripts/run_tsc.py` | Lanza una sesión (extractor, shard, datasets, resamples) | — |
| `scripts/extract_tsfresh.py` | Extrae tsfresh para la caché (AWS) | — |
| `scripts/check_data.py` | Verifica los 134 datasets (longitud, NaN, clases) | — |
| `scripts/collect_results.py` | Junta los JSON en `results/Tab_<extractor>.csv` (formato de los autores) | — |
| `notebooks/colab_runner.ipynb` | Todo lo anterior en Colab | — |

Los resultados en Drive quedan en `results/<extractor>/<dataset>/resample_XX.json`. Un archivo existe solo si el resample terminó (escritura atómica), así que volver a lanzar el mismo comando salta lo ya hecho.

## Local (laptop)

```powershell
python -m pip install --user uv==0.9.5
python -m uv venv --python 3.10 .venv
python -m uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

La prueba con TabPFN real se salta si no está definida la variable `HF_TOKEN`.

## Colab

1. Sube los cambios a GitHub (el notebook clona la rama `phase-1`).
2. Abre `notebooks/colab_runner.ipynb` en Colab (*Arquivo → Abrir notebook → GitHub*).
3. Elige **GPUs: L4**, guarda `HF_TOKEN` en *Secrets* y acepta los términos de `Prior-Labs/tabpfn_2_5` en Hugging Face.
4. Ajusta la celda de parámetros y ejecuta todo (*Executar tudo*).

Orden previsto: catch22 → los 3 extractores en los 7 de ECG → tsfresh → MultiROCKET (ver `plan-fases.md`). Con 2 sesiones: `NUM_SHARDS = 2` y `SHARD = 0` / `1`.

## Kaggle (tercera sesión de apoyo)

Notebook: `notebooks/kaggle_runner.ipynb`. Kaggle no escribe en Drive: corre una lista explícita de datasets (`DATASETS`) y deja los JSON en `/kaggle/working`, que luego se suben a Drive a mano. Límites: 12 h por corrida (si se pasa, no guarda nada) y ~30 h de GPU por semana. Por eso la celda de ejecución se detiene sola a las `TIME_BUDGET_H = 10.5` h y conserva lo terminado.

Uso para MultiROCKET (20 nov 2026): Kaggle toma la cola del shard 0, que avanza de barato a caro, empezando por lo más caro, para que ninguna sesión repita trabajo.

| Corrida | `DATASETS` | Costo de los autores |
|---|---|---|
| 1 | `["PhonemeSpectra"]` | 15.1 h (lo que no termine lo sigue el shard 0) |
| 2 | `["NonInvasiveFetalECGThorax2", "Adiac"]` | 7.4 h + 5.1 h |

1. *File → Import Notebook* → sube `notebooks/kaggle_runner.ipynb`.
2. *Settings → Accelerator* → **GPU T4 x2**; internet encendido; *Add-ons → Secrets* → `HF_TOKEN` activado.
3. Ajusta `DATASETS` y **Save Version → Save & Run All (Commit)**. Corre en segundo plano; puedes cerrar el navegador.
4. Al terminar, en la versión guardada → *Output* → descarga `results_kaggle_<fecha>.zip`.
5. Súbelo a `MyDrive/ts2tabpfn/` y, en una celda de Colab: `!unzip -n /content/drive/MyDrive/ts2tabpfn/results_kaggle_<fecha>.zip -d /content/drive/MyDrive/ts2tabpfn/` (`-n` no sobrescribe nada). Hazlo **antes** de que el shard 0 llegue a ese dataset; si llega antes, recalcula lo que falte (mismo resultado, tiempo perdido).

Medido en la T4 de Kaggle: PhonemeSpectra ~45 min por resample (~1.5× los autores; la L4 va a ~0.4–0.6×).

**tsfresh en Kaggle (22 nov 2026):** toma LSST (4.9 h de los autores), el dataset más caro del shard 1 de tsfresh, antes de que ese shard empiece. Descarga `MyDrive/ts2tabpfn/features/tsfresh/LSST.npz`, súbelo a Kaggle como dataset (*Add Input → Upload*), agrégalo como Input y usa `EXTRACTOR = "tsfresh"`, `DATASETS = ["LSST"]`. La celda de features copia los `.npz` de `/kaggle/input` y verifica forma y NaN.

## AWS (extracción de tsfresh)

Guía completa paso a paso en [aws_tsfresh.md](aws_tsfresh.md): instancia `c7a.8xlarge` que se configura sola con [`scripts/aws/tsfresh_user_data.sh`](../scripts/aws/tsfresh_user_data.sh), sube las features a S3 y se borra al terminar.
