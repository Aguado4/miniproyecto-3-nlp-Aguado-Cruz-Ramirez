# EXPERIMENTS - Bitácora de resultados

> **Regla:** solo números efectivamente ejecutados. **Estado:** corrida de referencia completa.

| Campo | Valor |
|---|---|
| Fecha | 2026-09-23 |
| Plataforma / GPU | Local (Windows 11) · NVIDIA GeForce RTX 4060, 8 GB |
| Python / torch / transformers | 3.12.2 / 2.11.0+cu128 / 5.17.0 (según la salida guardada de §1 y §4.4) |
| Semilla | 42 |
| Checkpoint principal | `dccuchile/bert-base-spanish-wwm-cased` |
| `MAX_LEN_BERT` | 192 (P95 WordPiece = 189) · padding dinámico · 4,2 % de test truncado |
| Ejecución | Restart & Run All, 64/64 celdas, 0 errores, **4.013 s (67 min)** |
| Referencias externas | MP1 `EXPERIMENTS.md` §2 · MP2 `EXPERIMENTS.md` §2 (corrida 2026-09-23) |

> **Sobre la reproducibilidad exacta.** Con `fp16` los núcleos de cuDNN no son deterministas, así
> que repetir el notebook completo mueve los resultados en el 3.º–4.º decimal (en §10, hasta
> 0,011 en mBERT). Las cifras de abajo son las de la corrida guardada en el `.ipynb` versionado.
> Diferencias menores de ~0,01 entre filas no deben interpretarse como reales.

## 1. Baselines heredados (bloqueante: deben reproducir MP1)

| Baseline | Accuracy | macro-F1 | MAE | QWK |
|---|---:|---:|---:|---:|
| Clase mayoritaria (5★) | 0.6565 | 0.1585 | 0.5495 | 0.0000 |
| Azar estratificado | 0.4798 | 0.1931 | 0.8342 | 0.0045 |

Reproducen exactamente los de MP1 → submuestra y Split A idénticos, comparación válida.

## 2. Tokenizador (§4.5-4.6)

| Métrica | Valor |
|---|---:|
| Fertilidad (subpalabras/palabra) | 1.29 |
| Tasa de `[UNK]` | 0.069 % |
| Mediana / media / P95 en tokens | 65 / 86 / 189 |
| Costo de atención vs. 512 fijo | ~14 % (tope fijo) · ~2,6 % (padding dinámico) |

Más fragmentadas: anglicismos (*snorkel*, *tripadvisor*), toponimia (*teotihuacán*,
*tlaquepaque*, *pátzcuaro*), comida local (*chilaquiles*), léxico afectivo con tilde
(*riquísimo*, *inigualable*).

## 3. Las tres técnicas (§5-§7) y comparación entre entregas (§8)

| # | Modelo | Entrega | macro-F1 | Accuracy | MAE | QWK | Entrenables | Tiempo (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C | **BETO fine-tuning completo** | MP3 | **0.5953** | 0.6847 | 0.3378 | 0.7840 | 109,854,725 | 1752.3 |
| | MP1 · BiLSTM + atención | MP1 | 0.5274 | 0.6758 | 0.3620 | 0.7546 | 9,441,862 | 88.4 |
| | MP1 · TF-IDF + LogReg | MP1 | 0.5235 | 0.6785 | 0.3760 | 0.7261 | 60,000 | 27.0 |
| B | BETO congelado + MLP (promedio) | MP3 | 0.5191 | 0.6823 | 0.3580 | 0.7551 | 526,341 | 871.8 |
| | MP1 · BiLSTM + spaCy (afinados) | MP1 | 0.4863 | 0.6358 | 0.4310 | 0.6942 | 9,441,605 | 38.9 |
| A | BETO congelado + lineal (promedio) | MP3 | 0.4719 | 0.6468 | 0.4275 | 0.6983 | 3,845 | 870.8 |
| A | BETO congelado + lineal (`[CLS]`) | MP3 | 0.4578 | 0.6178 | 0.4808 | 0.6460 | 3,845 | 873.1 |
| A' | BETO congelado + LogReg (promedio) | MP3 | 0.4398 | 0.6135 | 0.4842 | 0.6432 | 3,845 | 897.3 |
| | MP2 · Transformer desde cero | MP2 | 0.4160 | 0.5858 | 0.6050 | 0.5068 | 4,105,605 | 196.6 |
| | MP2 · MLP (embeddings promediados) | MP2 | 0.3545 | 0.5268 | 0.8860 | 0.3501 | 3,857,157 | 21.1 |
| | MP1 · LSTM desde cero | MP1 | 0.2951 | 0.5755 | 0.6510 | 0.3552 | 3,972,741 | 50.1 |

**F1 por clase, C vs. TF-IDF.** C: 1★ 0.6762 · 2★ 0.4672 · 3★ 0.5311 · 4★ 0.4984 · 5★ 0.8036.
Ganancia sobre TF-IDF: **1★ +0.080 · 2★ +0.166 · 3★ +0.098 · 4★ +0.023 · 5★ −0.009**.

**Costo de inferencia (C).** 8.8 ms/reseña con lote 1 · 1.91 ms/reseña con lote 64.
Extracción de embeddings congelados (40.000 reseñas, una vez): 867 s en la corrida versionada
(77 s en una corrida anterior en la misma máquina; probablemente por carga de la GPU, que estaba
compartida con otras tareas; §17, limitación 3). El tiempo de A, A' y B incluye esa extracción; las cabezas en sí entrenan en segundos.
La cabeza MLP (B) llegó a 16 épocas antes del early stopping.

## 4. Estudio de capas y LLRD (§9, `CFG_EST` = 8.000 reseñas / 2 épocas)

| Variante | % entrenable | macro-F1 | QWK | Tiempo (s) |
|---|---:|---:|---:|---:|
| k=0 capas | 0.54 | 0.4463 | 0.6482 | 52.4 |
| k=2 capas | 13.45 | 0.4871 | 0.7412 | 63.5 |
| **k=4 capas** | 26.35 | **0.5670** | 0.7787 | 76.0 |
| k=12 capas | 77.97 | 0.5648 | 0.7710 | 126.2 |
| k=12 + LLRD (0.9) | 100.00 | 0.5708 | 0.7739 | 143.1 |

Satura en k=4. LLRD aporta +0.0060 sobre k=12.

## 5. Checkpoints (§10, mismo `CFG_EST`)

| Checkpoint | Fertilidad | macro-F1 | MAE | QWK | Parámetros | Tiempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| BETO | 1.2941 | **0.5648** | 0.3658 | 0.7710 | 85,648,901 | 126.2 |
| mBERT | 1.5191 | 0.4854 | 0.4265 | 0.7169 | 177,857,285 | 172.1 |
| DistilBETO | 1.2527 | 0.5003 | 0.4200 | 0.7304 | 67,325,957 | 67.6 |

## 6. Curva de eficiencia de datos (§11)

| n reseñas | TF-IDF + LogReg | BETO fine-tuning |
|---:|---:|---:|
| 500 | 0.3061 | 0.3776 |
| 2.000 | 0.4299 | 0.4986 |
| 8.000 | 0.4994 | 0.5648 |
| 32.000 | 0.5235 | 0.5953 |

BETO con 8.000 supera a TF-IDF con 32.000 → **4× en datos etiquetados**.
BETO con 2.000 (0.4986) supera al Transformer desde cero de MP2 con 32.000 (0.4160) → el cruce
está en torno a **1.000 reseñas, 1/32 de los datos** (interpolando entre 500 y 2.000).

## 7. LoRA (§12)

| Variante | % entrenable | Entrenables | macro-F1 | MAE | QWK | Tiempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| LoRA r=8 (`query`,`value`) | 0.272 | 298,757 | 0.5286 | 0.3770 | 0.7647 | 111.7 |

## 8. Tarea de control `Type` (§13)

| Tarea | macro-F1 BETO | Referencia MP1 |
|---|---:|---:|
| `Type` (3 clases, balanceada) | 0.9579 | BiLSTM 0.949 |
| Polaridad (5 clases, desbalanceada) | 0.5648 | — |
| **Brecha** | **0.393** | |

## 9. Análisis de errores (§16)

|  | TF-IDF acierta | TF-IDF falla |
|---|---:|---:|
| **BETO acierta** | 2.289 | 450 |
| **BETO falla** | 425 | 836 |

| Subconjunto | n | Error BETO | Error TF-IDF |
|---|---:|---:|---:|
| Reseñas mixtas | 42 | 33.3 % | 38.1 % |
| Etiqueta incoherente | 94 | 28.7 % | 20.2 % |
| Todas | 4.000 | 31.5 % | 32.2 % |

## 10. Interpretabilidad (§14)

UMAP: sin ajuste los `[CLS]` forman dos nubes que no separan por polaridad (1★-2★ tienden a un
borde, mezcladas con 4★-5★); tras el ajuste, una sola banda ordenada 5★→1★ con solapamiento en
2★-3★-4★.

Integrated Gradients, top-3 subpalabras hacia 1★ en 18 reseñas negativas con negación:
negación 4 % · léxico 1★ del EDA 6 % · otra 91 %. Los tokens más frecuentes (`pé`, `imo`, `fas`,
`peor`, `caro`) son fragmentos WordPiece de palabras afectivas (*pésimo*, *feas*), así que la
cifra subestima el uso real de léxico de sentimiento. **No concluyente a esta escala.**

## 11. Veredicto de hipótesis

| Hipótesis | Veredicto |
|---|---|
| **H1** · el FT completo supera a TF-IDF, sobre todo en 2★/3★ | **Confirmada** (0.5953 vs. 0.5235; +0.166 en 2★, +0.098 en 3★) |
| **H2** · BETO congelado no supera a TF-IDF | **Confirmada** (mejor congelado: MLP 0.5191 < 0.5235) |
| **H3** · BETO con ~2.000 reseñas iguala al Transformer desde cero de MP2 con 32.000 | **Confirmada con margen** — BETO@2.000 = 0.4986 vs. MP2@32.000 = 0.4160 (+0.083 con el 16 % de los datos); cruce interpolado en ~1.000 reseñas |
