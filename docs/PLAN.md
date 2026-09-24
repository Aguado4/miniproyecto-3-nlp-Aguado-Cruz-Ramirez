# PLAN - Ejecución por fases

> Estado global: **corrida de referencia completa.** Restart & Run All el 2026-09-23 en una
> RTX 4060 local: 64/64 celdas, 0 errores, 41 min. Lecturas y §17 redactadas con números reales,
> `EXPERIMENTS.md` lleno. Pendiente: **H3 depende de la corrida de MP2** (`PREVIOS_MP2` vacío).

Leyenda: `[ ]` pendiente · `[~]` código listo, falta ejecutar y leer · `[x]` hecho

---

## Fase 0 - Scaffolding y especificación

- [x] Revisar `consigna.txt`, `rubrica.txt` y `Sesion3/1-text-classification-with-hf.ipynb`
- [x] `CLAUDE.md`, `SPEC.md`, `DATASET.md`, `DECISIONS.md`, `PLAN.md`, `EXPERIMENTS.md`
- [x] `README.md`, `requirements.txt`, `.gitignore`, estructura de carpetas
- [x] Copiar §1–§4.3 de MP1 sin modificar (celdas 3–65 de MP1) + verificación programática
- [ ] **Confirmar con el profesor:** reutilización del corpus (D-302)
- [x] `git init`, primer commit, repositorio remoto (privado)

## Fase 1 - Entorno, corpus y EDA (SPEC §0–3)

- [x] Ejecutar secciones 1–3; cifras coinciden con `DATASET.md`
- [x] Verificación de identidad del bloque heredado pasa

## Fase 2 - Protocolo (SPEC §4)

- [x] Bloque heredado §4.1–4.3 corre y el assert de baselines pasa (**bloqueante**)
- [x] 4.4 Config BERT · 4.5 Tokenizador WordPiece · 4.6 `MAX_LEN_BERT` · 4.7 `DatasetDict` (código)

## Fase 3 - Tres técnicas (SPEC §5–7)

- [x] 5 Embeddings cacheados (`[CLS]` y mean) + cabeza lineal + LogReg
- [x] 6 Cabeza MLP
- [x] 7 `WeightedTrainer` + *fine-tuning* completo, curvas

## Fase 4 - Comparación (SPEC §8)

- [x] Tabla única con MP1/MP2, confusión, costo-beneficio; veredicto H1/H2

## Fase 5 - Estudios (SPEC §9–§12)

- [x] 9 Capas descongeladas + LLRD
- [x] 10 BETO vs. mBERT vs. DistilBETO
- [x] 11 Curva de eficiencia de datos · H3 **abierta**: falta la fila del Transformer de MP2
- [x] 12 LoRA

## Fase 6 - Control, interpretabilidad, demo (SPEC §13–§15)

- [x] 13 `Type` · 14 UMAP antes/después + IG · 15 demo y pruebas de estrés

## Fase 7 - Errores, cierre y entrega (SPEC §16–§17)

- [x] 16 Análisis de errores · 17 Conclusiones
- [x] Quitar todo `<!-- LEER -->` / andamiaje
- [x] Restart & Run All (RTX 4060 local), salidas guardadas, `EXPERIMENTS.md` lleno

## Presupuesto de tiempo (T4, estimado — verificar en la primera corrida)

Supuestos: fp16, padding dinámico (longitud media ~80 tokens), `CFG_EST` = 8k × 2 épocas.

| Bloque | min |
|---|---:|
| Entorno + EDA (incluye descarga de spaCy ~570 MB) | 5–7 |
| Protocolo + tokenización | 1–2 |
| §5–§6 extracción de embeddings (una vez) + cabezas | 2–3 |
| §7 *fine-tuning* completo (32k × ≤3 épocas) | 6–9 |
| §9 capas k∈{0,2,4,12} + LLRD (5 corridas) | 5–7 |
| §10 mBERT + DistilBETO (BETO = reutilizar §9 k=12) | 2–3 |
| §11 curva (solo 500 y 2k nuevos; 8k y 32k se reutilizan) | 1–2 |
| §12 LoRA + §13 `Type` | 2–3 |
| §14–§16 | 2–3 |
| **Total** | **~26–39** |
