# DECISIONS - Registro de decisiones de diseño

Formato ADR abreviado: contexto, decisión, alternativas, consecuencias.
Estados: `aceptada` · `propuesta` · `revisada` · `revertida`

> Numeración desde **D-301**. Las de entregas anteriores se citan como `MP1 §D-00N` y
> `MP2 §D-2NN`.

---

## D-301 · La tarea es clasificación con BERT

**Estado:** aceptada · 2026-09-19

**Contexto.** La primera versión de `consigna.txt` se titulaba «generación de texto con GPT»
por error. Ya se corrigió a «Mini-Proyecto de clasificación de texto con BERT», que coincide
con el notebook de la Sesión 3.

**Decisión.** Clasificación de polaridad con BETO.

---

## D-302 · Reutilizar corpus, submuestra, Split A y EDA de MP1

**Estado:** aceptada (pendiente de reconfirmar con el profesor) · 2026-09-19

**Contexto.** El profesor autorizó en MP2 la reutilización con la condición de incluir el
mismo EDA. MP1 §D-009 excluyó BETO «para una entrega posterior».

**Decisión.** Mismo corpus y **mismas Secciones 1–4.3 de MP1 sin modificar** (celdas 3–65 de MP1:
entorno, corpus, EDA, tokenizador, submuestra, Split A/B, `evaluar`, baselines), con celdas
puente antes y después. A diferencia de MP2 (que solo copió §2–§3), aquí también se hereda el
protocolo: así la comparación entre entregas no depende de reimplementar el split ni la
métrica. Lo específico de BERT va en §4.4–4.7 con nombres propios (`CFG_BERT`,
`MAX_LEN_BERT`) para no sobrescribir nada heredado.

**Por qué BERT sí aplica a este corpus.** Es texto en español (BETO está preentrenado en
español), la tarea es clasificación de secuencia (exactamente `AutoModelForSequenceClassification`),
y las reseñas son cortas (mediana 48 palabras): `MAX_LEN` ≈ 200 tokens WordPiece, muy por
debajo del límite de 512. Además es el único cambio de técnica que la comparación entre
entregas necesitaba.

**Consecuencias.** Tabla comparativa de tres entregas sobre el mismo split. Se arrastra
`spacy` solo por el EDA §3.7.

---

## D-303 · `MAX_LEN` = P95 en tokens WordPiece, con padding dinámico

**Estado:** aceptada · 2026-09-19

**Contexto.** El guía fija `max_length=512` con `padding='max_length'` porque sus noticias
tienen mediana ~500 palabras. Aquí la mediana es 48 palabras.

**Decisión.** P95 de longitud en tokens WordPiece sobre train (redondeado a múltiplo de 8),
truncamiento a esa longitud y `DataCollatorWithPadding` (cada lote se rellena solo hasta su
secuencia más larga).

**Consecuencias.** Con atención cuadrática, pasar de 512 a ~200 reduce el costo por capa de
atención ~6×; con padding dinámico, más. Se reporta el truncamiento.

---

## D-304 · No replicar los defectos del notebook guía

**Estado:** aceptada · 2026-09-19

| Defecto en el guía | Por qué importa aquí | Corrección |
|---|---|---|
| Solo accuracy | Mayoritaria = 65,6 % | `evaluar` con macro-F1, MAE, QWK |
| `train_test_split` sin semilla | Irreproducible | Split A fijo de MP1 |
| `max_length=512` + padding fijo | ~6× de cómputo desperdiciado | D-303 |
| Cabeza propia termina en `LogSoftmax` | HF aplica `CrossEntropyLoss` a los logits: doble normalización | Cabeza devuelve logits |
| Con BERT congelado, el `Trainer` recalcula el encoder en cada época | Tiempo tirado | Embeddings precalculados una vez |
| CE sin pesos | Colapso hacia 5★ | `WeightedTrainer` |
| Evalúa el último modelo, no el mejor | Sobreajuste | `load_best_model_at_end` por macro-F1 de val |

---

## D-305 · Tres técnicas centrales = las tres del guía, bien medidas

**Estado:** aceptada · 2026-09-19

**Decisión.** Congelado + lineal, congelado + MLP, *fine-tuning* completo. Son una escalera
natural de «cuánto del modelo se entrena» y responden H2. Las variantes de §9–§12 son
innovación, no reemplazo.

---

## D-306 · Estudios con `CFG_EST` reducido

**Estado:** aceptada · 2026-09-19

**Contexto.** Un *fine-tuning* completo de BETO sobre 32k en T4 cuesta ~6–9 min (estimado). §9–§13
suman ~10 corridas.

**Decisión.** Train estratificado de 8.000, 2 épocas, mismo val/test. Se declara que esas
filas comparan entre sí. §11 usa sus propios tamaños por diseño.

---

## D-307 · Latencia y parámetros entrenables como métricas de costo

**Estado:** aceptada · 2026-09-19

**Contexto.** Comparar BETO (110 M) con TF-IDF (60k features) solo en F1 esconde la mitad
de la decisión.

**Decisión.** Reportar params totales, params entrenables, tiempo de entrenamiento y
latencia de inferencia (ms/reseña, lote 1 y lote 64) en el mismo hardware.

---

## D-308 · El andamiaje heredado se limpia en MP1, no en MP3

**Estado:** aceptada · 2026-09-23

**Contexto.** `CLAUDE.md` §5 prohíbe dejar comentarios de andamiaje (`<!-- LEER -->`) en el
entregable, pero §1 obliga a copiar el bloque §1–§4.3 de MP1 **sin modificar**. El bloque heredado
arrastraba 7 marcadores de MP1, así que las dos reglas no se podían cumplir a la vez.

**Hallazgo.** Los 7 marcadores estaban **obsoletos**: en los 7 casos la prosa que pedían ya
estaba redactada en la celda inmediatamente anterior (la 17 la cubre la 16, la 36 la 35, la 40
la 39, la 47 la 46, la 59 la 58, y la 52 es el encabezado de §3.8 cuya tabla está completa en la
53). No faltaba contenido: sobraba andamiaje.

**Decisión.** Borrarlos **en el notebook de MP1** y volver a copiar el bloque a MP3, en vez de
editarlos solo aquí. Así el bloque sigue siendo byte-idéntico a su origen y los dos entregables
quedan limpios. Se aprovechó para quitar otros 2 marcadores de MP1 fuera del bloque heredado
(§5 y §6 de aquel notebook).

**Alternativas.** (a) Dejarlos y documentar la excepción: incumple `CLAUDE.md` §5 en dos
entregas. (b) Editarlos solo en MP3: rompe la identidad byte a byte, que es lo único que
garantiza que las tres entregas sean comparables.

**Consecuencias.** El bloque heredado pasa de 68 a 63 celdas (se eliminaron 5 celdas que solo
contenían el comentario; 2 conservan su texto sin él). Solo se tocó markdown, así que las
salidas ejecutadas de MP1 siguen intactas y no hizo falta reejecutar ninguno de los dos
notebooks. La celda puente de MP3 sigue siendo la única diferencia intencional frente a MP1
(§D-302).

---

## Plantilla para nuevas entradas

## D-3NN · Título breve

**Estado:** propuesta · AAAA-MM-DD

**Contexto.** …  **Decisión.** …  **Alternativas.** …  **Consecuencias.** …
