# CLAUDE.md — Instrucciones de trabajo para este repositorio

Este archivo es la fuente de verdad operativa para cualquier sesión de agente que trabaje
en este proyecto. Léelo completo antes de tocar cualquier archivo.

---

## 1. Qué es este repositorio

Entregable del **Miniproyecto 3** del curso de NLP (Maestría, Universidad Icesi).
El producto final es **un único Jupyter Notebook** ejecutable de principio a fin que
clasifica la polaridad (1–5★) de reseñas turísticas en español con un **modelo BERT
preentrenado en español (BETO)**, comparando tres formas de usarlo —extractor congelado +
cabeza lineal, extractor congelado + cabeza MLP propia, y *fine-tuning* completo— y
situándolas frente a los modelos de los Miniproyectos 1 y 2.

Se basa en el notebook guía `icesi-nlp/Sesion3/1-text-classification-with-hf.ipynb`
(BETO sobre `mteb/spanish_news`). **No se reutiliza su dataset**: se trabaja sobre el corpus
propio `vg055/Rest-Mex2025`.

Los dos archivos que definen el éxito del trabajo están en la raíz y **no se modifican**:

- `consigna.txt` — qué pide el profesor (clasificación de texto con BERT).
- `rubrica.txt` — cómo se califica (7 puntos en 4 criterios).

### Relación con las entregas anteriores

- **Secciones 1–4.3 heredadas del Miniproyecto 1 SIN MODIFICAR** (celdas 3–70: entorno,
  corpus, EDA, tokenizador, submuestra, Split A/B, `evaluar`, baselines), con una celda puente
  antes y otra después. **No editar esas celdas** (tag `heredado-mp1`; una celda verifica la
  identidad). Lo propio de BERT empieza en §4.4 y usa `CFG_BERT` / `MAX_LEN_BERT`; nunca
  reasignar `CFG`, `MAX_LEN`, `evaluar` ni los splits heredados (§D-302).
- **MP1 §D-009 excluyó BETO** porque el profesor indicó que el *fine-tuning* se vería en una
  entrega posterior. **Es esta.** El notebook lo dice explícitamente.
- Las líneas de comparación están en `../miniproyecto 1/docs/EXPERIMENTS.md` y
  `../miniproyecto 2/docs/EXPERIMENTS.md`: misma submuestra de 40k, mismo Split A, mismas
  métricas.

## 2. Documentos de especificación (leer en este orden)

| Archivo | Contiene |
|---|---|
| `docs/SPEC.md` | Especificación completa del notebook. **Es el contrato.** |
| `docs/PLAN.md` | Plan de ejecución por fases, con estado. |
| `docs/DATASET.md` | Ficha del corpus (heredada) + tokenización WordPiece de BETO. |
| `docs/DECISIONS.md` | Registro de decisiones (ADR), numeradas desde D-301. |
| `docs/EXPERIMENTS.md` | Bitácora de resultados. Solo números ejecutados. |

## 3. Reglas duras (violarlas cuesta puntos de rúbrica)

### 3.1 Reproducibilidad — 2 de 7 puntos

- «Restart & Run All» limpio, sin intervención manual.
- **`SEED = 42`** en `random`, `numpy`, `torch`, splits **y** `TrainingArguments(seed=...,
  data_seed=...)`. Fijar la semilla antes de cada corrida de los estudios (§9–§12).
- Nada de rutas locales: corpus y checkpoints se descargan de HuggingFace Hub.
- **Detectar Colab/local y GPU/CPU y degradarse sin romperse.** Sin GPU, *fine-tuning*
  completo de BETO es inviable: se usa submuestra pequeña, `MAX_LEN` corto y 1 época, y se
  imprime una advertencia. Nunca fallar.
- **Presupuesto: ≤ 45 min end-to-end en T4.** El *fine-tuning* manda: `fp16=True` en GPU,
  embeddings congelados **precalculados una sola vez** (§5), y los estudios de §9–§12 con
  `CFG_EST` (submuestra de train reducida, menos épocas).
- **No guardar checkpoints en disco** (`save_strategy="no"` o `save_total_limit=1` en un
  directorio temporal). Colab se queda sin disco con varios BERT-base.
- Las salidas de las celdas se guardan en el `.ipynb` versionado.

### 3.2 Narrativa — 1 de 7 puntos

- **Todo en español.** Código, comentarios, markdown, gráficas.
- Ninguna celda de código sin una markdown antes que explique el porqué.
- Toda gráfica y tabla seguida de su lectura. Cada sección cierra con hallazgos; el notebook
  con conclusiones y limitaciones honestas.

### 3.3 Técnicas — 2 de 7 puntos («al menos tres técnicas» comparadas)

Las tres técnicas centrales, las mismas del guía pero **bien medidas**:

1. BETO congelado + cabeza lineal.
2. BETO congelado + cabeza MLP propia.
3. BETO *fine-tuning* completo.

Mismo split, misma función `evaluar(...)`, misma tabla. Se añaden como filas de contexto los
modelos de MP1 y MP2.

### 3.4 Innovación — 2 de 7 puntos

Diferenciales frente al guía: fine-tuning parcial por capas y LR discriminativa (§9),
comparación de checkpoints BETO vs. mBERT vs. DistilBETO (§10), curva de eficiencia de datos
contra TF-IDF y el Transformer desde cero (§11), LoRA (§12), mapa del espacio de embeddings
antes/después del *fine-tuning* e *Integrated Gradients* (§14), demo con pruebas de estrés
idénticas a las de MP2 (§15).

## 4. Convenciones técnicas

- **Notebook autocontenido**, sin `src/`. Funciones auxiliares en la sección donde se usan.
- Métrica principal: **macro-F1**. Nunca accuracy sola (el guía solo reporta accuracy;
  aquí la mayoritaria da 65,6 %).
- Una única `evaluar(...)` para todos los modelos. El `compute_metrics` del `Trainer`
  **llama a `evaluar`** (con `registrar=False`) para que early stopping y tabla usen lo mismo.
- **Pérdida ponderada por clase** en el `Trainer` (subclase con `compute_loss`), como en MP1/MP2.
  El guía usa CE sin pesos.
- Etiquetas internas 0–4; `evaluar` recibe estrellas 1–5 (sumar 1). No mezclar.
- Checkpoint principal: `dccuchile/bert-base-spanish-wwm-cased` (el del guía).
- Gráficas: `matplotlib` + `seaborn`, ejes en español.

## 5. Qué NO hacer

- No usar `mteb/spanish_news` ni ningún dato del guía.
- No copiar los defectos del guía (§D-304): split sin semilla, `max_length=512` fijo con
  *padding* a 512, solo accuracy, `LogSoftmax` al final de la cabeza propia (el modelo de HF
  aplica `CrossEntropyLoss` sobre logits: `LogSoftmax` + CE es una doble normalización),
  recalcular BERT congelado en cada época.
- No entrenar el *fine-tuning* sobre val/test ni elegir hiperparámetros mirando test.
- No añadir dependencias fuera de `requirements.txt`. Toda instalación va en la sección 1.
- No dejar `TODO`, celdas vacías ni comentarios de andamiaje (`<!-- LEER -->`) en el entregable.
- No inventar resultados en `EXPERIMENTS.md`.
- No hacer `git push` ni crear releases sin pedirlo al usuario.

## 6. Comandos útiles

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download es_core_news_lg   # solo para el EDA §3.7

jupyter nbconvert --to notebook --execute notebooks/miniproyecto3_restmex_bert.ipynb \
  --output ejecutado.ipynb --ExecutePreprocessor.timeout=5400
```
