# Referencias de código

Repos de terceros clonados en `references/` (excluidos de Git). Para recrearlos:

```bash
mkdir -p references && cd references
git clone https://github.com/gabrielcmerlin/TS2TabPFN.git
git -C TS2TabPFN checkout 70543e4912727aa73d02401aca30e16b4d0d85b9
git clone https://github.com/Soroushsrd/ECG_Feature_Extraction.git
git -C ECG_Feature_Extraction checkout 25d0943824c67b6d6892f5de8b560d6293386bb2
```

| Repo | Commit (3 nov 2026) | Contenido relevante |
|---|---|---|
| [TS2TabPFN](https://github.com/gabrielcmerlin/TS2TabPFN) | `70543e4` | `code/extract_{c22,multi,tsfresh}.py`, `code/main.py`, `config.yaml`, `env.yaml` (entorno conda `ecml`), resultados en `tsc/` y `tser/` |
| [ECG_Feature_Extraction](https://github.com/Soroushsrd/ECG_Feature_Extraction) | `25d0943` | `ECG_Feature_Extraction.py` (NeuroKit2 + PyWavelets), notebooks sobre PTB-XL |

## Notas del código de referencia

- `config.yaml` corre por defecto `c22` sobre `ArrowHead`, `task: tsc`, `seed: 0`, `device: cuda`, `runs: 1`.
- Lista `invalid_datasets` con los datasets excluidos por memoria.
- Los pesos de TabPFN 2.5 requieren login en Hugging Face (`huggingface-cli login`, token de lectura).
- Para comparar resultados usan `labicompare`: `pip install git+https://github.com/jose-gilberto/labicompare/`.
- `ECG_Feature_Extraction` fija versiones antiguas (`numpy==1.24.3`, `neurokit2==0.2.7`); posible conflicto con aeon/TabPFN.

## Otros

- ECGomics: <https://github.com/PKUDigitalHealth/ECGomics> (no clonado aún)
- NI LabVIEW Biomedical Toolkit, ECG Feature Extractor (propietario, solo referencia)
