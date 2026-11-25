# Resultados de la reproducción (Fase 2, clasificación)

Comparación con las tablas de los autores (`references/TS2TabPFN/tsc/results/Tab_*.csv`): media de los 30 resamples por dataset, victorias/empates/derrotas (empate si |diferencia| ≤ 0.0001) y test de Wilcoxon de rangos con signo sobre los datasets (§4.2). Generado con `scripts/compare_with_authors.py`; archivos en `results/`.

## catch22 + TabPFN (C22-TabPFN) ✅ completo

Corrido en Colab L4 el 19 nov 2026 (código `phase-1`, ver `compare_summary.csv`).

| | Nuestra | Autores |
|---|---|---|
| Datasets × resamples | 134 × 30 | 134 × 30 |
| Accuracy media | **0.7992** | **0.7993** |

- Diferencia media: **−0.0001**. Victorias/empates/derrotas: 32 / 60 / 42. **Wilcoxon p = 0.221**: no hay diferencia significativa con los autores.
- **2 743 de 4 020 resamples (68 %) dan exactamente la misma accuracy** que los autores.
- En los 108 datasets con ≤ 10 clases la diferencia máxima entre medias es 0.0023. Las diferencias mayores están en los 26 datasets con más de 10 clases (máx. Fungi, −0.0168; PigCVP, +0.0109): ahí `ManyClassClassifier` reparte las clases al azar y ni el código de los autores ni el nuestro fijan su semilla (`code/utils/utils_train.py:148`), así que no se espera coincidencia exacta.
- Gráfico: `results/compare_c22.png`. Detalle por dataset: `results/compare_c22.csv`.

**Conclusión:** C22-TabPFN queda reproducido.

## MultiROCKET + TabPFN (MultiROCKET-TabPFN) ✅ completo

Corrido del 19 al 25 nov 2026: 2 sesiones Colab L4 (shards 0 y 1) + Kaggle T4 para PhonemeSpectra (resamples 0–13) y NonInvasiveFetalECGThorax2 (los 30). Código `phase-1`; comparación generada en el commit indicado en `compare_summary.csv`.

| | Nuestra | Autores |
|---|---|---|
| Datasets × resamples | 134 × 30 | 134 × 30 |
| Accuracy media | **0.8632** | **0.8632** |

- Diferencia media: **+0.0001**. Victorias/empates/derrotas: 60 / 24 / 50. **Wilcoxon p = 0.543**: no hay diferencia significativa con los autores.
- Mediana de |diferencia| por dataset: 0.0012; percentil 95: 0.0077. Solo 5 datasets pasan de 0.01, y en 132 de 134 la diferencia cabe en 2 errores estándar de la media de los autores.
- 1 014 de 4 020 resamples (25 %) dan exactamente la misma accuracy. Se esperan menos coincidencias que con catch22 porque los kernels de MultiROCKET dependen de la semilla, que los autores no fijan (decisión #11).
- Las diferencias mayores son ruido esperable: BeetleFly (20 series de test: una serie = 0.05), Herring (64), SelfRegulationSCP2 (2 clases, accuracy ~0.5) y PigCVP / PigAirwayPressure (52 clases, `ManyClassClassifier` sin semilla, igual que en catch22).
- Los 7 datasets de ECG de UCR quedan a ≤ 0.0053 de los autores (ECG200: 0.902 vs. 0.907). Los resultados corridos en Kaggle (T4) también coinciden: NonInvasiveFetalECGThorax2 +0.0001, PhonemeSpectra +0.0042.
- El paper (§5) reporta 0.8627 de media; la tabla publicada de los autores (`Tab_multi.csv`) da 0.8632 sobre los mismos 134 datasets. Comparamos contra la tabla.
- Gráfico: `results/compare_multirocket.png`. Detalle por dataset: `results/compare_multirocket.csv`.

**Conclusión:** MultiROCKET-TabPFN queda reproducido.

## tsfresh + TabPFN — en curso

Extracción completa (134/134, sin NaN). TabPFN: 61 de 134 datasets al 25 nov 2026 (shards 0 y 1 en Colab L4; LSST completo en Kaggle).
