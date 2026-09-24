# SPEC - Especificación del entregable

> **Estado:** borrador · **Versión:** 0.1 · **Última actualización:** 2026-09-19
>
> Contrato del entregable. Si la implementación se desvía, se actualiza primero este archivo.

---

## 1. Problema

### 1.1 Enunciado

Dada una reseña turística en español sobre un destino mexicano, predecir la **polaridad**
(1 a 5 estrellas) que le asignó su autor usando solo el texto. Es la tarea de MP1 y MP2.

La pregunta de esta entrega:

> **¿Cuánto vale el preentrenamiento sobre esta tarea, y qué parte del modelo hay que
> ajustar para cobrarlo?**

### 1.2 Por qué esta pregunta

Las dos entregas anteriores dejaron el mismo resultado incómodo: **TF-IDF + LogReg
(macro-F1 0,524) es difícil de superar** entrenando desde cero con 32.000 reseñas. Solo la
BiLSTM con atención lo empató (0,527). MP2 cerró con la pregunta de *qué haría falta para que
un Transformer ganara*; la respuesta canónica es el preentrenamiento. Aquí se mide.

El guía de la Sesión 3 muestra tres formas de usar BERT (congelado + lineal, congelado +
MLP, *fine-tuning*) pero las compara solo por accuracy, sobre un corpus balanceado, sin
semilla ni costo. Sobre un corpus con 66 % de 5★ esa comparación no dice nada. Hipótesis:

> **H1.** BETO con *fine-tuning* supera a TF-IDF, y la ganancia se concentra en **2★ y 3★**
> (las fronteras finas donde todos los modelos anteriores rondan F1 0,2–0,4), no en 1★ ni 5★.
>
> **H2.** Congelado, BETO **no** supera a TF-IDF: el `[CLS]` de un modelo entrenado con MLM no
> está organizado por sentimiento. La cabeza MLP apenas ayuda; lo que importa es
> **descongelar** (§9 mide cuántas capas bastan).
>
> **H3 (eficiencia de datos).** BETO *fine-tuned* con ~2.000 reseñas iguala o supera al
> Transformer desde cero de MP2 entrenado con 32.000.

### 1.3 Por qué la propuesta es adecuada

1. **Idioma y dominio.** BETO se preentrenó en ~3.000 M de palabras en español; el guía usa
   noticias, aquí son reseñas coloquiales con mexicanismos y negación — un salto de dominio
   que pone a prueba la transferencia.
2. **Los datos minoritarios son pocos** (~840 reseñas de 1★ y 2★ en la submuestra). Es el
   régimen en el que el preentrenamiento debería marcar más diferencia.
3. **Hay una escalera ya medida** (MP1 y MP2) sobre el mismo split: el resultado se lee como
   un escalón más, no como un número aislado.

## 2. Datos

`vg055/Rest-Mex2025`. Ficha en [`DATASET.md`](DATASET.md). EDA sobre el corpus completo
(208k, heredado); modelado sobre la **submuestra estratificada de 40.000** y el **Split A
80/10/10**, `SEED = 42`, idénticos a MP1 y MP2.

## 3. Protocolo

### 3.1 Métricas

Las seis de MP1/MP2 con la misma función `evaluar(...)`: **macro-F1 (principal)**,
accuracy (solo junto a la mayoritaria), MAE, QWK, F1 por clase, tiempo de entrenamiento.
Se añaden **parámetros totales y entrenables**, y **latencia de inferencia** (ms/reseña),
porque la mitad del argumento es el costo.

### 3.2 Baselines obligatorios

Mayoritaria (acc 0,6565, macro-F1 0,1585) y azar estratificado. **Deben reproducir MP1.**
Si no, parar: el split no es el mismo.

### 3.3 Configuraciones

- `CFG` — heredada de MP1, intacta (no se usa para BERT). `CFG_BERT` — corridas principales (§5–§7).
- `CFG_EST` — estudios (§9–§12): train reducido (8.000 reseñas estratificadas), 2 épocas,
  mismo val/test. **Sus filas comparan entre sí, no contra la tabla de §8.**

## 4. Estructura del notebook

Archivo único: `notebooks/miniproyecto3_restmex_bert.ipynb`

### Sección 0 — Portada y problema (propias)

### Secciones 1–4.3 — Bloque heredado del Miniproyecto 1

**Las celdas 3–65 del notebook de MP1 se copian SIN MODIFICAR**: §1 entorno y semilla, §2
corpus, §3 EDA completo, §4.1 tokenizador por palabras, §4.2 submuestra + Split A + Split B,
§4.3 `evaluar(...)` + baselines. Es lo que garantiza que las tres entregas se comparen sobre la
misma submuestra, el mismo split y la misma función de evaluación. Piezas heredadas que aquí
no se usan (vocabulario por palabras, Split B, `CFG`, `MAX_LEN` en palabras) se conservan para
no romper la identidad y **no se sobrescriben**: lo propio usa `CFG_BERT`, `MAX_LEN_BERT`.

Una celda puente antes y otra después del bloque. **Aceptación:** verificación programática de
identidad (63 celdas) y assert de que los baselines reproducen MP1.

### Secciones 4.4–4.7 — Protocolo propio de BERT

| # | Contenido | Aceptación |
|---|---|---|
| 4.4 | Dependencias (`transformers`, `peft`, `captum`…), `CHECKPOINT`, `CFG_BERT`, `CFG_EST` | Config impresa |
| 4.5 | Tokenizador WordPiece de BETO: fertilidad, `[UNK]`, ejemplos con mexicanismos | Contraste con EDA §3.7 |
| 4.6 | `MAX_LEN_BERT` = P95 en tokens WordPiece (no 512 del guía); padding dinámico | Truncamiento y ahorro frente a 512 |
| 4.7 | `DatasetDict` tokenizado, etiquetas 0–4, pesos de clase | — |

La `evaluar(...)` heredada no se modifica: los costos propios (params entrenables, latencia) se
añaden a la fila que devuelve, y el early stopping usa un envoltorio silencioso sobre ella.

### Sección 5 — Técnica A: BETO congelado + cabeza lineal

Como el guía, pero **se precalculan los embeddings una vez** (BETO en `no_grad`) y se entrena
solo la cabeza sobre el tensor cacheado: mismo resultado, órdenes de magnitud más rápido.
Se comparan dos resúmenes de la secuencia: `[CLS]` vs. *mean pooling* enmascarado (conexión
con MP2 §D-205). Como referencia barata, **LogReg sobre los mismos embeddings**.

### Sección 6 — Técnica B: BETO congelado + cabeza MLP propia

La cabeza del guía, **corregida**: sin `LogSoftmax` final (§D-304), con pérdida ponderada.
Mismos embeddings cacheados de §5. Pregunta: ¿la no linealidad extrae sentimiento que la
lineal no ve?

### Sección 7 — Técnica C: *fine-tuning* completo

`AutoModelForSequenceClassification` + `Trainer` con: `WeightedTrainer` (pérdida ponderada),
`lr=2e-5` (aquí sí es la tasa correcta — contraste explícito con MP2 §D-204), warmup 10 %,
*weight decay* 0,01, `fp16`, padding dinámico, early stopping por **macro-F1 de validación**,
3 épocas máx. Curvas de pérdida train/val.

### Sección 8 — Comparación de las tres técnicas (+ MP1 y MP2)

Tabla única: macro-F1, acc, MAE, QWK, F1 por clase, params totales / entrenables, tiempo,
latencia. Matrices de confusión lado a lado. Gráfica costo-beneficio (tiempo o params
entrenables vs. macro-F1) con **todos** los modelos de las tres entregas.
**Aceptación:** veredicto explícito sobre H1 y H2 con números.

### Sección 9 — ¿Cuántas capas hay que descongelar? (aporte propio)

Con `CFG_EST`: descongelar las últimas *k* ∈ {0, 2, 4, 12} capas del encoder; más una
corrida con **LR discriminativa por capa** (*layer-wise LR decay*, Howard & Ruder 2018).
Gráfica macro-F1 y tiempo vs. *k*.

### Sección 10 — ¿Qué modelo preentrenado? (aporte propio)

Con `CFG_EST`, *fine-tuning* completo de:

| Checkpoint | Pregunta |
|---|---|
| `dccuchile/bert-base-spanish-wwm-cased` (BETO) | Referencia |
| `google-bert/bert-base-multilingual-cased` (mBERT) | ¿Cuánto vale un modelo monolingüe? |
| `dccuchile/distilbert-base-spanish-uncased` (DistilBETO) | ¿Cuánto se pierde con la mitad de capas? |

Se reporta fertilidad del tokenizador de cada uno sobre el corpus (mBERT fragmenta más el
español: ¿se nota en F1?).

### Sección 11 — Curva de eficiencia de datos (aporte propio, H3)

BETO *fine-tuned* y TF-IDF + LogReg con {500, 2.000, 8.000, 32.000} reseñas de train
(estratificadas, mismo val/test). Se marca como línea horizontal el macro-F1 del Transformer
desde cero de MP2 (32k). **Aceptación:** respuesta explícita a H3.

### Sección 12 — LoRA: *fine-tuning* eficiente en parámetros (más allá de clase)

`peft` con LoRA (r=8, sobre `query`/`value`) con `CFG_EST`. Se compara contra el
*fine-tuning* completo y el congelado: % de parámetros entrenables vs. macro-F1.

### Sección 13 — Tarea de control: `Type`

La técnica C prediciendo `Type` (3 clases balanceadas) con `CFG_EST`. Se compara contra la
BiLSTM de MP1 (0,949) y el Transformer de MP2. Mide cuánto de la dificultad es de la tarea.

### Sección 14 — ¿Qué aprendió el *fine-tuning*? Interpretabilidad

1. **Espacio de embeddings**: UMAP del `[CLS]` de test coloreado por estrellas, **antes**
   (BETO congelado) y **después** del *fine-tuning*. Muestra visualmente H2.
2. ***Integrated Gradients*** (`captum`) sobre reseñas con negación: atribución por token.
3. Contraste con el léxico distintivo del EDA §3.5.
Se advierte que atribución ≠ explicación causal.

### Sección 15 — Demo con pruebas de estrés

`predecir_resena(texto)` → estrellas, distribución, atribución. **Los mismos tres pares de
MP2 §15** (negación, orden invertido, queja al final), ejecutados con texto fijo, y una fila
comparativa BETO vs. Transformer de MP2 cuando esté disponible. Widget opcional.

### Sección 16 — Análisis de errores

Errores por distancia en estrellas; reseñas que falla BETO y acierta TF-IDF (y viceversa);
categorías (ironía, mixta, muy corta, etiqueta incoherente) con ejemplos.

### Sección 17 — Conclusiones y limitaciones

Respuesta a §1.1, veredicto sobre H1–H3, recomendación práctica (qué técnica usaría en
producción y por qué), limitaciones y trabajo futuro.

## 5. Criterios de aceptación

- [ ] «Restart & Run All» sin errores en ≤ 45 min con T4.
- [ ] Bloque heredado (§1–§4.3, celdas 3–65 de MP1) idéntico (verificación programática).
- [ ] Baselines reproducen MP1.
- [ ] Las tres técnicas del guía implementadas y en una tabla única con MP1/MP2.
- [ ] Ninguno de los defectos del guía (§D-304) replicado.
- [ ] §9–§13 fijan semilla antes de cada corrida.
- [ ] Toda gráfica/tabla con su lectura; conclusiones responden H1–H3.

## 6. Mapa rúbrica → notebook

| Criterio | Pts | Dónde |
|---|---:|---|
| Notebook completo | 1 | Narrativa en toda celda, cierre por sección, §17 |
| Reproducibilidad | 2 | §1 (semilla, config adaptativa, `CFG_EST`), sin checkpoints en disco, presupuesto |
| ≥ 3 técnicas comparadas | 2 | §5, §6, §7 → §8 (+ §9, §10, §12 como técnicas adicionales) |
| Innovación | 2 | §9 capas/LLRD · §10 checkpoints · §11 eficiencia de datos · §12 LoRA · §14 UMAP + IG · §15 demo · comparación entre 3 entregas |

Consigna: *«prueba diferentes técnicas, arquitecturas, parámetros»* → §5–§7, §9, §10.
*«técnicas más allá de las vistas en clase»* → §9 LLRD, §12 LoRA, §14 IG. *«hacer demos»* →
§15. *«añadir plots»* → §8, §9, §11, §14. *«buen EDA»* → §2–§3 + §4.3.

## 7. Fuera de alcance

- Generación de texto (ver §D-301).
- Modelos grandes (> 400 M parámetros) o LLMs vía API.
- Búsqueda exhaustiva de hiperparámetros.
- Split B geográfico y pérdidas ordinales (ya en MP1).
