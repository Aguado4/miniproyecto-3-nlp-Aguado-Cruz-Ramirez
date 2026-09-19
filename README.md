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

> Pendiente de la corrida de referencia ([`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)).

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
