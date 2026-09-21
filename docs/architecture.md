# Architecture

Sebeni is a recursive loop between a rule-based morphological analyzer (Daba)
and a small language model (SLM). Nothing in this page is FastText,
Matryoshka embeddings, or a chatbot — those tracks are out of scope.

Public URL: [https://seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

## Three product phases

1. **Distill resources** — lexical references (`\va` / `\ve` variants) become
   a SIL Toolbox **grammar G** and **dictionary D**. Frontier LLMs synthesize
   Daba pattern files; HITL (`--hitl`) validates Select-Mark
   `pattern select_gloss | mark_gloss` and compounding such as `:n: [ :v: :n: ]`.
2. **Refine G/D while training** — Daba is a **dynamic checkpoint**. If Φ on a
   training batch is below τ, Distiller proposes \(G_{cand}, D_{cand}\)
   and Sebeni promotes only when Φ′ > Φ.
3. **Update θ** — GRPO (default) samples a group of completions and
   scores them with Daba. DPO / APO are drop-in **policy-update plugins**.

```mermaid
flowchart TD
  T["Dataset T: rows of text, lang"] --> Split["Split by Language.group_code"]
  Split --> GD["Per language: G_ℓ, D_ℓ"]
  Split --> Theta["One shared policy θ"]
  GD --> Phi["Φ ← DabaX"]
  Phi --> Dist["Distiller if Φ < τ"]
  Dist --> Theta
  Theta --> R["Rewards → plugin update"]
```

## One θ, many (G, D)

```mermaid
flowchart LR
  subgraph policy["Shared"]
    theta["θ"]
  end
  subgraph langs["Per language"]
    bam["G_bam, D_bam"]
    mku["G_mku, D_mku"]
    dtm["G_dtm, D_dtm"]
  end
  theta --> bam
  theta --> mku
  theta --> dtm
```

## SafetyGovernor gates

```mermaid
flowchart TD
  Promote["Promote G, D"] --> G1["Φ′ > Φ / parseable first-create"]
  Policy["Policy step"] --> G2["Valid JSON tokens; R_lang; U / KL"]
  Hub["Hub push"] --> G3["Model card + safety_snapshot.json"]
```

## Objects

| Symbol | Role |
| --- | --- |
| \(T\) | Alignment dataset: `{text, lang}` rows |
| \(B_ℓ\) | Texts in the batch whose group code is ℓ |
| \(G_ℓ, D_ℓ\) | Daba `.gram` / `.dict` for that group |
| \(\Phi_ℓ\) | Mean token stage score of \(B_ℓ\) given \(G_ℓ, D_ℓ\) |
| \(\tau\) | Default **0.5** |
| \(\theta\) | **One** shared policy, even when \(T\) is multilingual |
| Distiller | Proposes G, D (not `distillation_hook`, which is KL scaling) |

## Package layout

```
beni/
  cli/main.py              sebeni init|train|distill|eval|exp|wordfreq|generate|push
  core/srl/                SAMPG + GRPO/DPO/APO plugins
  core/safety/             SafetyGovernor gates
  core/compute/            Φ, MER, MCS, U, RewardManager
  core/morphotactic/       Distiller, DabaX (CLI daba.mparser)
  core/language.py         ISO → group_code (mku, mey, kao, spp)
  core/hub/                model cards
  core/wordfreq/           surface / lemma / morpheme / stage counts
  data/baselines/{lang}/   packaged G, D (Maninka files may live under mlq/)
  data/raw/                packaged train texts for sebeni exp
  data/test.json           packaged eval texts for sebeni exp
  utils/                   workdir, prompts, language metadata
```

Relocatable workdir (later wins): `~/.sebeni` → `SEBENI_HOME` /
`SEBENI_WORKING_DIR` → YAML `working_dir` → CLI `-w`.

| Path under workdir | Contents |
| --- | --- |
| `data/baselines/{lang}/` | `baseline.gram` / `.dict`, then `baseline_vN` |
| `models/` | Policy, tokenizer, Hub `README.md`, `safety_snapshot.json` |
| `exp/` | `eval.json`, `wordfreq/` |
| `runtime/` | Headless mparser scratch |

## What Sebeni does *not* do

- Vendor GPL `daba` sources (install maslinych/daba from GitHub with
  `--no-deps` for its CLI modules).
- Require wxPython (`[gui]` is optional for upstream gparser).
- Treat G or D as dataset columns.
- Mix languages **inside** one completion JSON object (`R_lang` is 0).
