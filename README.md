# Miniproyecto 3 - NLP

**¿Cuánto vale el preentrenamiento? BETO sobre reseñas turísticas en español**

Maestría · Universidad Icesi · Curso de Procesamiento de Lenguaje Natural

Autores: Juan José Aguado · Juan David Cruz · Juan Diego Ramírez

---

## El problema

Dada una reseña turística en español sobre un destino mexicano, predecir cuántas estrellas
(1 a 5) le puso su autor, usando solo el texto. Es la tarea de los Miniproyectos
[1](../miniproyecto%201/) y [2](../miniproyecto%202/); cambia la técnica y con ella la pregunta:

> **¿Cuánto vale el preentrenamiento sobre esta tarea, y qué parte del modelo hay que
> ajustar para cobrarlo?**

En dos entregas, entrenando desde cero, nada superó con claridad a TF-IDF + Regresión
Logística (macro-F1 0,524). El Miniproyecto 1 dejó BETO fuera por indicación del profesor
(«se verá en una entrega posterior»): es esta.

- **H1.** BETO con *fine-tuning* supera a TF-IDF, y la ganancia se concentra en 2★ y 3★.
- **H2.** Congelado, BETO no supera a TF-IDF; lo que importa es descongelar, no la cabeza.
- **H3.** Con ~2.000 reseñas, BETO iguala al Transformer desde cero de MP2 entrenado con 32.000.

## La propuesta

| Sección | Qué hace |
|---|---|
| §4 | Protocolo heredado + análisis del tokenizador WordPiece; `MAX_LEN` por P95, no 512 |
| §5 | **Técnica A** — BETO congelado + cabeza lineal (`[CLS]` vs. *mean pooling*) |
| §6 | **Técnica B** — BETO congelado + cabeza MLP propia |
| §7 | **Técnica C** — *fine-tuning* completo con pérdida ponderada y early stopping por macro-F1 |
| §8 | Comparación de las tres + modelos de MP1 y MP2, costo-beneficio |
| §9 | ¿Cuántas capas descongelar? + LR discriminativa por capa |
| §10 | BETO vs. mBERT vs. DistilBETO |
| §11 | Curva de eficiencia de datos contra TF-IDF |
| §12 | LoRA (PEFT) |
| §13 | Tarea de control `Type` |
| §14 | UMAP del espacio de embeddings antes/después + *Integrated Gradients* |
| §15 | Demo con las mismas pruebas de estrés que MP2 |
| §16–§17 | Errores y conclusiones |

## Qué corregimos del notebook guía

Parte de `Sesion3/1-text-classification-with-hf.ipynb`, que usa otro corpus
(`mteb/spanish_news`). Detalle en [`docs/DECISIONS.md`](docs/DECISIONS.md) §D-304: solo
accuracy, split sin semilla, padding fijo a 512, `LogSoftmax` antes de una
`CrossEntropyLoss`, BERT congelado recalculado en cada época, pérdida sin pesos.

## El corpus

[`vg055/Rest-Mex2025`](https://huggingface.co/datasets/vg055/Rest-Mex2025) — 208.051 reseñas
(CC-BY-4.0). Reutilizado de MP1 con el EDA copiado sin modificar
([`docs/DATASET.md`](docs/DATASET.md)).

## Resultados

Corrida de referencia completa (`Restart & Run All`, 64/64 celdas, sin errores, RTX 4060 local).
Todas las filas usan la misma submuestra de 40.000 reseñas, el mismo Split A y la misma función
`evaluar`, así que las tres entregas son directamente comparables. Detalle en
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

| Modelo | Entrega | macro-F1 | Accuracy | MAE | QWK |
|---|---|---:|---:|---:|---:|
| **C · BETO *fine-tuning* completo** | MP3 | **0.595** | 0.685 | **0.338** | **0.784** |
| BiLSTM + atención | MP1 | 0.527 | 0.676 | 0.362 | 0.755 |
| TF-IDF + Regresión Logística | MP1 | 0.524 | 0.679 | 0.376 | 0.726 |
| B · BETO congelado + MLP | MP3 | 0.519 | 0.682 | 0.358 | 0.755 |
| A · BETO congelado + lineal (promedio) | MP3 | 0.472 | 0.647 | 0.428 | 0.698 |
| Transformer desde cero | MP2 | 0.416 | 0.586 | 0.605 | 0.507 |
| Baseline (clase mayoritaria) | — | 0.159 | 0.657 | 0.549 | 0.000 |

- **H1 — confirmada.** El *fine-tuning* supera a TF-IDF (+0,072) y la ganancia se concentra en
  las clases difíciles: 2★ +0,166, 3★ +0,098, 1★ +0,080, frente a 4★ +0,023 y 5★ −0,009.
- **H2 — confirmada, por poco.** Congelado, ni la mejor cabeza (MLP, 0,519) supera a TF-IDF
  (0,524). El valor del preentrenamiento se cobra ajustando los pesos; con 4 de las 12 capas
  descongeladas ya se obtiene todo (§9).
- **H3 — confirmada con margen.** BETO con 2.000 reseñas (0,499) supera al Transformer desde cero
  de MP2 entrenado con 32.000 (0,416). Con 8.000 reseñas, BETO (0,565) ya supera al TF-IDF
  entrenado con las 32.000.

Además: mBERT y DistilBETO rinden menos que BETO (§10), LoRA entrena el 0,27 % de los pesos y
llega a 0,529 (§12), y la tarea de control `Type` alcanza 0,958 con el mismo modelo (§13), lo
que sitúa buena parte de la dificultad en la propia polaridad y en el ruido de sus etiquetas.

## Cómo ejecutarlo

**Colab (recomendado):** abrir `notebooks/miniproyecto3_restmex_bert.ipynb`, GPU T4, ejecutar todo.

**Local:**

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download es_core_news_lg
jupyter lab notebooks/miniproyecto3_restmex_bert.ipynb
```

Sin GPU el notebook no falla: reduce submuestra, `MAX_LEN` y épocas, y lo avisa.

## Estructura

```
.
├── CLAUDE.md · README.md · consigna.txt · rubrica.txt · requirements.txt
├── docs/        SPEC · PLAN · DATASET · DECISIONS · EXPERIMENTS
├── notebooks/   miniproyecto3_restmex_bert.ipynb
└── results/     figures/ · metrics/
```
