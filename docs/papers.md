# Resumen de papers

## TS2TabPFN (artículo principal)

**Merlin, G. C.; Silva, D. F.** *TS2TabPFN: Time Series Classification and Extrinsic Regression through Feature Extraction and a Tabular Foundation Model.* arXiv:2608.04174v1 (4 ago 2026). DOI 10.1007/978-3-032-37676-3_32.
Código: <https://github.com/gabrielcmerlin/TS2TabPFN> (clonado en `references/TS2TabPFN`).
PDF: `paper/ts2tabpfn-merlin-silva-2026.pdf` (18 páginas).

> Cifras verificadas contra el PDF el 3 nov 2026. Entre paréntesis, la sección del paper.

### Idea

Serie temporal → extractor de features ψ → tabla → **TabPFN 2.5** (pesos congelados, inferencia in-context en un forward pass). Sin ajuste de hiperparámetros por dataset.

### Extractores (§3.2)

| Variante | Herramienta | Configuración |
|---|---|---|
| C22-TabPFN | catch22 (aeon) | 22 features; en multivariado se concatenan por canal |
| tsfresh-TabPFN | tsfresh | `EfficientFCParameters`; concatena train+test antes de extraer; `impute` para NaN |
| MultiROCKET-TabPFN | MultiROCKET (aeon) | Configuración por defecto |

TabPFN: límites recomendados de ~50 000 muestras y ~2 000 features; más de 500 features activa un ensemble con submuestreo automático de features.

### Datos y protocolo (§4.1, §4.2)

- **TSC:** UCR + UEA, 158 → 134 datasets (los demás excedían la VRAM). Normalización z-score de UCR.
- **TSER:** TSML Extended Archive, 63 → 55 datasets. Escalas originales.
- Valores faltantes: interpolación lineal.
- 30 resamples con las semillas y splits oficiales.
- 134/55 se eligieron por la memoria de GPU: TabPFN carga train y test juntos en VRAM (atención O(n²)).
- Métricas: accuracy (TSC), RMSE (TSER). Wilcoxon con corrección de Holm (α = 0.05), win/tie/loss y diagramas de diferencia crítica.

### Baselines (§4.3)

- **No se re-ejecutaron**: los resultados se toman de la literatura. HC2 y DrCIF solo se corrieron localmente en 5 datasets para medir tiempo.
- TSC: DrCIF, ROCKET, Arsenal, HC1, HC2. TSER: FreshPRINCE, InceptionTime, MiniROCKET, MultiROCKET, RotationForest, DrCIF.

### Resultados clave (§5)

- **TSC:** MultiROCKET-TabPFN fue el mejor extractor. Accuracy media 0.8627 vs. 0.8674 de HC2 (diferencia no significativa, p = 0.669). Frente a ROCKET: 99/3/32.
- **TSER:** tsfresh-TabPFN fue el mejor; frente a DrCIF y FreshPRINCE ganó 51/0/4.
- Entre 2.29× (BeetleFly) y 98.39× (CinCECGTorso) más rápido que HC2 de punta a punta (Tabla 1).

### Hardware (§4.4)

i7-10700F, 32 GB RAM, RTX 3060 12 GB. Software: aeon, TabPFN oficial, CodeCarbon para energía.

### Limitaciones (§6)

- tsfresh y MultiROCKET generan muchas features: extraer por lotes si falta RAM.
- TabPFN necesita todo el contexto en GPU; más de 500 features implica más forward passes.

### Observaciones para la reproducción

- El `config.yaml` del repo trae `runs: 1`, pero el paper promedia **30 resamples**. Una reproducción rápida con 1 run no es directamente comparable con la tabla del paper; hay que decidirlo y anotarlo en `supuestos.md`.
- Como los baselines vienen de la literatura, para comparar basta con reproducir TS2TabPFN y contrastar con los resultados publicados en `references/TS2TabPFN/tsc/results/`.

### Datasets de ECG dentro de UCR (útiles para el Paso 1)

`ECG200`, `ECG5000`, `ECGFiveDays`, `TwoLeadECG`, `CinCECGTorso`, `NonInvasiveFetalECGThorax1`, `NonInvasiveFetalECGThorax2`. En TSER: `BIDMC32HR`, `BIDMC32RR`, `BIDMC32SpO2` (señales fisiológicas).

## ECGomics (lectura recomendada por Rafael)

**Zhang, D. et al.** *ECGomics: An Open Platform for AI-ECG Digital Biomarker Discovery.* Health Data Science 6 (2026): 0427. CC BY 4.0. PDF en `paper/`.

- Propone una taxonomía de 4 dimensiones para el ECG: **Structural, Intensity, Functional, Comparative**.
- Combina features expertas (morfológicas, vía pipelines ENCASE y FeatureDB) con embeddings profundos (modelo fundacional ECGFounder).
- Plataforma web, app móvil y API: <https://github.com/PKUDigitalHealth/ECGomics>.
- Validado en detección de fibrilación auricular, recurrencia tras crioablación, estenosis coronaria en ECG aparentemente normales y monitoreo materno.

**Relevancia:** sirve como catálogo de qué features específicas de ECG vale la pena extraer, y como un tercer candidato de extractor si su código es abierto.
