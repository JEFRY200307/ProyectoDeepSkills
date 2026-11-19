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

## AWS (extracción de tsfresh)

1. Lanza una `c6i.8xlarge` on-demand en Ohio con Ubuntu 22.04 y disco de 50 GB, con SSH solo desde tu IP.
2. Instala el entorno y baja los datos:
   ```bash
   sudo apt-get update && sudo apt-get install -y unzip git python3-pip
   pip install uv==0.9.5 && export PATH=$HOME/.local/bin:$PATH
   git clone -b phase-1 https://github.com/JEFRY200307/ProyectoDeepSkills.git && cd ProyectoDeepSkills
   uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -r requirements.txt
   .venv/bin/python -m gdown 1p8D-haDDwPKloKMBQCQHgENYHVsjixR4 -O tsc_data.zip && unzip -q tsc_data.zip -d data/raw/tsc
   ```
3. Calibra con 3 datasets livianos y mide el tiempo:
   ```bash
   .venv/bin/python scripts/extract_tsfresh.py --data-dir <carpeta con ECG200/> --features-dir features --n-jobs 32 --datasets ECG200 Coffee GunPoint
   ```
4. Lanza el resto en segundo plano, con apagado automático al terminar:
   ```bash
   nohup .venv/bin/python scripts/extract_tsfresh.py --data-dir <carpeta> --features-dir features --n-jobs 32 --shutdown > tsfresh.log 2>&1 &
   ```
5. Copia `features/tsfresh/*.npz` a `MyDrive/ts2tabpfn/features/tsfresh/` (por ejemplo con `rclone` o descargándolos) y **termina la instancia** (*Terminate*).

Alerta de presupuesto en AWS Budgets antes de lanzar (decisión #8).
