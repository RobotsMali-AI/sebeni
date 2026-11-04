# Wordfreq

Morphological counter in the spirit of
[bamfreq](https://mlsftwrs.web.app/app/bamfreq): tokenize and analyze with CLI
mparser / `DabaX.loader` against the **current DabaX checkpoint**
(`baseline_vN` in `{working_dir}/data/baselines/{lang}/`).

Not FastText. FastText / Matryoshka embeddings are a later vectorization track.

## Counts (`{working_dir}/exp/wordfreq/`)

- Surface forms (`token.surface`)
- Lemma / analysis `form` from the first Daba analysis
- Morpheme forms from that analysis's nested `Morpheme` tree
- Stage histogram (feeds Phi / integrity reporting)
- Stage −1 miss map (feeds algorithmic Distiller)

## Raw-input pipeline

```yaml
data:
  default_lang: bam
  languages: [bam, mku]
  encoding: utf-8-sig      # used to read the raw files
wordfreq:
  raw_inputs: ./data/raw   # directory, glob, file, or list
```

Wordfreq does not require `{text, lang}` JSONL. It reads raw text (a
directory is scanned recursively for `*.txt` / `*.text`; relative paths resolve
against the config file's directory), splits each file into paragraphs on blank
lines (or into lines when it has no blank lines), and runs DabaX. Each file's
language is its filename stem (`bam.txt`, `mlq.txt` → `mku`); a stem that is not
a known language code falls back to `data.default_lang`. With one language in
`data.languages` / `--lang`, every file is counted as that language; with
several, files whose stem group is not listed are skipped. `data.source` is a
compatibility fallback when `wordfreq.raw_inputs` is unset; otherwise packaged
`beni/data/raw` is used.

```
sebeni wordfreq -c config.yaml
sebeni wordfreq -c config.yaml --lang bam --lang mku
```

Writes `{working_dir}/exp/wordfreq/wordfreq.json` (an index of languages with
`path`, `n_tokens`, `n_sentences`, `checkpoint_id`, `n_misses`) plus a
subdirectory per language (`bam/`, `mku/`, …) holding that language's
`wordfreq.json` and `surfaces.tsv`, `lemmas.tsv`, `morphemes.tsv`, `stages.tsv`,
`misses.tsv` (`form<TAB>count`, most frequent first). Each language is counted
against its latest paired workdir `baseline_vN`, then the workdir
`baseline.gram` / `baseline.dict`, falling back to the packaged baseline
(`checkpoint_id: packaged`). The command does not instantiate an LLM Distiller
and needs no API key.

```python
from beni.core.wordfreq import count_raw_inputs, count_texts

reports = count_raw_inputs(["beni/data/raw/bam.txt", "beni/data/raw/mku.txt"])
report = count_texts(["Aw ka kɛnɛ wa?"], lang="bam")   # WordfreqReport
report.write("./wordfreq/bam")
```
