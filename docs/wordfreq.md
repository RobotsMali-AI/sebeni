# Wordfreq

Morphological counter in the spirit of
[bamfreq](https://mlsftwrs.web.app/app/bamfreq): tokenize and analyze with CLI
mparser / `DabaX.loader` against the **current Distiller checkpoint**
(`baseline_vN` in `{working_dir}/data/baselines/{lang}/`).

Not FastText. FastText / Matryoshka embeddings are a later vectorization track.

## Counts (`{working_dir}/exp/wordfreq/`)

- Surface forms (`token.surface`)
- Lemma / analysis `form` from the best Daba analysis
- Optional morpheme forms from the nested `Morpheme` tree
- Stage histogram (feeds Phi / integrity reporting)

```
sebeni wordfreq -c config.yaml
sebeni wordfreq -c config.yaml --lang bam --lang mku
```

Writes `{working_dir}/exp/wordfreq/wordfreq.json` plus a subdirectory per
language (`bam/`, `mku/`, …) with surfaces / lemmas / morphemes / stages.
Each language is counted against **that** language's Distiller checkpoint.

If no checkpoint exists, Distiller bootstrap runs first (same auto-distill rule
as `train`). Source texts come from `data.source` (or packaged data), grouped
by `lang`. `sebeni exp` evaluates packaged `test.json` instead; wordfreq still
uses `data.source` unless you point it at the same split.
