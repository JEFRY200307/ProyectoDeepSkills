# Contexto del proyecto

## Personas

| Persona | Rol | Contacto |
|---|---|---|
| Diego Furtado Silva | Profesor asociado, ICMC-USP. Coautor de TS2TabPFN. Orienta el trabajo | diegofsilva@icmc.usp.br |
| Rafael da Costa Silva | USP. Recomendó datasets de ECG y el artículo ECGomics | rafael.csilva@usp.br |
| José Gilberto | En copia. Probable autor de `labicompare` (librería usada por TS2TabPFN para comparar resultados) | |
| Gabriel da Costa Merlin | Primer autor de TS2TabPFN | |

## Correos

### Diego, 7 de septiembre de 2026

> Tente reproduzir algum experimento do nosso artigo que te mandei. Dando certo, seguiremos para algo específico para sinais fisiológicos. Copio o Rafael e o Gilberto, que podem apontar para vocês alguns conjuntos de dados de ECG mais fáceis para fazerem as primeiras análises.

Indicaciones adicionales de Diego:

- Artículo: <https://arxiv.org/pdf/2608.04174>
- Flujo: extracción de features → TabPFN.
- La idea es extraer algo directo de los ECG, más enfocado. Comparar lo que hay en el artículo (extracción con tsfresh y ROCKET) contra una herramienta específica, y ver cuál conviene usar:
  - <https://github.com/Soroushsrd/ECG_Feature_Extraction>
  - NI LabVIEW Biomedical Toolkit, ECG Feature Extractor: <https://www.ni.com/docs/en-US/bundle/labview-biomedical-toolkit-api-ref/page/lvbiomed/bio_ecg_extractor.html>

### Rafael, 8 de septiembre de 2026

- Entendió el objetivo: extraer features de ECG y pasarlas a TabPFN.
- Sugiere empezar leyendo **ECGomics** (adjunto, en `paper/`), una herramienta de extracción de features y biomarcadores específicos para ECG. Ofreció reunirse para discutirlo.
- Recomendó datasets en orden de relevancia: ver [datasets.md](datasets.md).

Capturas originales en [correos/](correos/).

## Objetivo concreto

1. **Paso 1 (pedido por Diego):** reproducir algún experimento de TS2TabPFN.
2. **Paso 2:** aplicar el pipeline a ECG y comparar extracción genérica frente a específica de dominio.

## Notas sobre las herramientas

- `ECG_Feature_Extraction` es Python (NeuroKit2 + PyWavelets). Es usable directamente.
- El NI Biomedical Toolkit es **LabVIEW** (software propietario de NI). No se integra fácil en un pipeline Python; se usará solo como referencia de qué features extrae, salvo que se consiga licencia.
- ECGomics publica código en <https://github.com/PKUDigitalHealth/ECGomics> (por revisar si el extractor FeatureDB es abierto o solo vía plataforma/API).

## Uso doble

Este repositorio también es el proyecto final del curso **AI for Engineers** (DeepSkill), cuyo único requisito es compartir el enlace al proyecto respondiendo el correo de cierre.
