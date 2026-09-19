# EXPERIMENTS - Bitácora de resultados

> **Regla:** solo números efectivamente ejecutados. Si no se corrió, la celda queda vacía.

> **Estado:** sin ejecutar.

| Campo | Valor |
|---|---|
| Fecha | — |
| Plataforma / GPU | — |
| Python / torch / transformers | — |
| Semilla | 42 |
| Submuestra | 40,000 (Split A: 32,000 / 4,000 / 4,000) |
| Checkpoint | `dccuchile/bert-base-spanish-wwm-cased` |
| `MAX_LEN` (WordPiece, P95) / truncamiento | — |
| Fertilidad (subpalabras por palabra) | — |

## 1. Baselines (deben reproducir MP1: acc 0.6565, macro-F1 0.1585)

| Baseline | Accuracy | macro-F1 | MAE | QWK |
|---|---:|---:|---:|---:|
| Clase mayoritaria | | | | |
| Azar estratificado | | | | |

## 2. Tres técnicas + entregas anteriores — Split A, test

| Modelo | Entrega | macro-F1 | Acc | MAE | QWK | Params entrenables | Tiempo (s) | ms/reseña |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LogReg | MP1 | 0.5235 | 0.6785 | 0.376 | 0.7261 | 60,000 | 27.0 | |
| BiLSTM + atención | MP1 | 0.5274 | 0.6758 | 0.362 | 0.7546 | 9,441,862 | 88.4 | |
| MLP simple | MP2 | | | | | | | |
| Transformer desde cero | MP2 | | | | | | | |
| A · BETO congelado + lineal (`[CLS]`) | MP3 | | | | | | | |
| A' · BETO congelado + lineal (mean) | MP3 | | | | | | | |
| B · BETO congelado + MLP | MP3 | | | | | | | |
| C · BETO *fine-tuning* completo | MP3 | | | | | | | |

### F1 por clase

| Modelo | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg (MP1) | 0.596 | 0.301 | 0.433 | 0.475 | 0.813 |
| A | | | | | |
| B | | | | | |
| C | | | | | |

H1: — · H2: —

## 3. Capas descongeladas (§9, `CFG_EST`)

| k capas | LLRD | macro-F1 | Params entrenables | Tiempo (s) |
|---|---|---:|---:|---:|
| 0 | no | | | |
| 2 | no | | | |
| 4 | no | | | |
| 12 | no | | | |
| 12 | sí | | | |

## 4. Checkpoints (§10, `CFG_EST`)

| Checkpoint | Fertilidad | macro-F1 | Params | Tiempo (s) |
|---|---:|---:|---:|---:|
| BETO | | | | |
| mBERT | | | | |
| DistilBETO | | | | |

## 5. Eficiencia de datos (§11)

| n train | TF-IDF macro-F1 | BETO FT macro-F1 |
|---:|---:|---:|
| 500 | | |
| 2,000 | | |
| 8,000 | | |
| 32,000 | | |

Referencia: Transformer desde cero MP2 (32k) = — . H3: —

## 6. LoRA (§12) · 7. Tarea `Type` (§13)

| Variante | % params entrenables | macro-F1 | Tiempo (s) |
|---|---:|---:|---:|
| LoRA r=8 | | | |

| Modelo | `Type` macro-F1 |
|---|---:|
| BiLSTM (MP1) | 0.949 |
| BETO FT | |
