# Getting started

Install Sebeni, write `{text, lang}` rows (or use packaged data), then
`init` → `train` → `eval` → `generate`. Zero data prep:
[Experiments](experiments.md) (`sebeni exp`). Copy-paste recipes:
[Use cases](use-cases.md).

Public docs: [https://seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

## Install

Python ≥ 3.10. Primary install is GitHub (not an editable checkout):

```bash
pip install "sebeni[train,distil] @ git+https://github.com/mlsftwrs/sebeni.git"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
sebeni --help
```

| Extra | What |
| --- | --- |
| (default) | CLI, YAML, DabaX runtime dependencies, metrics, safety |
| `[train]` | torch, transformers, trl, peft, datasets, accelerate, trackio |
| `[wandb]` | wandb (for `trainer.report_to: wandb`) |
| `[distil]` | google-genai, openai, groq, together |
| `[docs]` | MkDocs Material + mkdocstrings |
| `[dev]` | pytest, ruff |
| `[gui]` | wxPython — **not** required |

Parser credit: [maslinych/daba](https://github.com/maslinych/daba) (GPLv2+).
Install it from GitHub with `--no-deps`; DabaX does **not** require wxPython.
Sebeni already pins the CLI runtime (`setuptools>=65,<81` / `pkg_resources`,
`funcparserlib`, `intervaltree`, `pytrie`, `attrdict3`, `regex`).

From a clone:

```bash
pip install -e ".[train,distil,dev]"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
```

The default algorithmic Distiller needs no key. Optional LLM backends read
`.env`, Google ADC, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`,
`TOGETHER_API_KEY`, `HF_TOKEN`. Distiller keys are required when
`distillation.enabled: true` (the default). Skip Distiller with
`--no` on HITL flags plus `distillation.enabled: false`, or `sebeni train`
after a prior `sebeni distill`.

## Minimal dataset

Every row is one sentence and its language. G and D are **files**, not columns.

```json
{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}
{"text": "i ni ce", "lang": "mku"}
```

CSV (`text,lang`) and `.txt` (one sentence per line; lang from filename or
`default_lang`) also work. Hugging Face datasets: set `data.source` to the
id and `hf_lang_key` if the column is not `lang`. Preference scheme (DPO/APO):
`{text, lang, chosen, rejected}` or `{text, lang, completions, scores}`.

## Happy path

```mermaid
flowchart LR
  Install --> Init["sebeni init"]
  Init --> Train["sebeni train"]
  Train --> Eval["sebeni eval"]
  Eval --> Next["generate / wordfreq / push"]
```

```bash
sebeni init --lang bam --lang mku -w ./runs/manding-001
# set data.source in config.yaml to your jsonl
sebeni train -c ./runs/manding-001/config.yaml --lr 1e-5 --max-steps 20
sebeni eval  -c ./runs/manding-001/config.yaml
sebeni generate -c ./runs/manding-001/config.yaml --prompt "Aw ka kɛnɛ wa?" --lang bam
sebeni wordfreq -c ./runs/manding-001/config.yaml
```

`init` creates `config.yaml` plus `data/`, `models/`, `runs/`, `exp/`,
`runtime/`. Repeat `--lang` for multilingual runs. Trainer / LoRA knobs:
[Hyperparameters](hyperparams.md) or `sebeni train --help`.

## Relocatable working directory

Later wins if set:

1. `~/.sebeni`
2. `SEBENI_HOME` / `SEBENI_WORKING_DIR`
3. YAML `working_dir:`
4. CLI `-w`

Relative YAML paths resolve against the config file directory; relative `-w`
against cwd. Tests should pass `-w` a temp dir.

## Scratch language

No packaged `beni/data/baselines/{lang}/` → Distiller writes stubs under
`{working_dir}/data/baselines/{lang}/` and uses bootstrap prompts. First
promote: parseable files. Later: Φ′ > Φ. Maninka group is **mku**.

## Artifacts

| Path | What |
| --- | --- |
| `{working_dir}/data/baselines/{lang}/` | `baseline.gram` / `.dict`, `baseline_vN` |
| `{working_dir}/models/` | Policy, tokenizer, Hub `README.md`, `safety_snapshot.json` |
| `{working_dir}/exp/` | `eval.json`, wordfreq |
| `{working_dir}/runtime/` | Headless mparser |
