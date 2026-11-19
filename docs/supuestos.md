# Registro de supuestos y decisiones

Toda decisión que no venga explícita del paper o del código de referencia se anota aquí.

| # | Fecha | Fase | Decisión | Motivo | Fuente / alternativa |
|---|---|---|---|---|---|
| 1 | 2026-11-03 | 0 | El NI LabVIEW Biomedical Toolkit se usa solo como referencia | Es propietario y no se integra en Python | Correo de Diego |
| 2 | 2026-11-03 | 0 | PTB-XL como primer dataset de ECG | Primero en la lista de Rafael; folds oficiales | `docs/datasets.md` |
| 3 | 2026-11-18 | 1 | Experimentos en Google Colab (GPU); desarrollo y pruebas en local | El equipo local no tiene GPU NVIDIA; el paper usó RTX 3060 12 GB (§4.4) | Alternativa: Kaggle Notebooks |
| 4 | 2026-11-18 | 2-3 | Cada dataset se usa completo (split train/test oficial íntegro, sin submuestrear instancias) | Pedido del usuario: la reproducción debe ser sobre datasets completos | Si un caso no entra en VRAM, se documenta aquí antes de reducir nada |
| 5 | 2026-11-18 | 2 | 30 resamples por dataset y extractor (resample 0 = split oficial; 1-29 = remuestreo estratificado con `random_state = seed + run`) | Igual que el paper (§4.2); el `config.yaml` de referencia trae `runs: 1`, se cambia a 30 | `references/TS2TabPFN/code/main.py` (`SEED+run`), `code/utils/utils_data.py:155-162` |
| 6 | 2026-11-18 | 2 | Alcance de la reproducción: solo clasificación (TSC). (a) C22-TabPFN sobre los 134 datasets completos; (b) catch22, tsfresh y MultiROCKET sobre los 7 datasets de ECG de UCR. Todo con 30 resamples | Diego pidió "algum experimento"; replicar todo el paper cuesta ~370 h de GPU; TSC es lo que sirve para PTB-XL | `docs/datasets.md` (costos); TSER queda fuera de alcance por ahora. **Reemplazada por #7** |
| 7 | 2026-11-18 | 2 | Alcance ampliado: experimento de clasificación completo, 3 extractores × 134 datasets × 30 resamples. TSER sigue fuera de alcance | Con Colab Pro (396 CU, L4 a 1.94 CU/h medido) el costo estimado es ~215–250 CU, dentro del presupuesto | `docs/computo.md` |
| 8 | 2026-11-18 | 1-2 | Cómputo: TabPFN en Colab Pro (2 sesiones L4), extracción de tsfresh en AWS `c6i.8xlarge`, Kaggle de apoyo. Presupuesto: 400 CU + hasta 200 CU extra; ~$30 del crédito de AWS | AWS y GCP no dan cuota de GPU a cuentas sin historial; Colab permite 2 sesiones GPU a la vez (probado) | `docs/computo.md`; alternativa: GPU del ICMC-USP |
