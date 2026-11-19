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

## MultiROCKET + TabPFN — en curso

## tsfresh + TabPFN — pendiente (extracción en AWS)
