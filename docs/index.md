<figure markdown="span">
  ![SEBEN!](assets/seben-wordmark.png){ width="360" }
</figure>

# Sebeni

Sebeni is a **morphotactic post-training toolkit** for extremely low-resource
languages (ELRL). It runs **Self-Aware Morphotactic Pattern Generation (SAMPG)**:
a dataset of `(text, language)` rows is scored with a Daba **grammar G** and
**dictionary D**; if morphological integrity Φ is below τ the Distiller proposes
new G, D; then the shared policy θ is updated with GRPO (or DPO / APO as
policy-update plugins).

```bash
pip install "sebeni[train,distil] @ git+https://github.com/mlsftwrs/sebeni.git"
sebeni --help
sebeni exp -c configs/exp.yaml -w ./runs/exp-001
```

Public docs: [https://seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).
Project home: [seben.robotsmali.org](https://seben.robotsmali.org).
Code: [github.com/mlsftwrs/sebeni](https://github.com/mlsftwrs/sebeni).
Hub: [huggingface.co/mlsftwrs](https://huggingface.co/mlsftwrs).

## Objects in the loop

| Symbol | Role |
| --- | --- |
| \(T\) | Dataset of `{text, lang}` rows — not G or D |
| \(B\) | A mini-batch \(B \subset T\) |
| \(G, D\) | Daba grammar and dictionary files (`.gram` / `.dict`) |
| \(\Phi\) | Morphological integrity of the **batch texts** given \(G, D\) |
| \(\tau\) | Threshold, default **0.5** |
| \(\theta\) | The SLM policy (**one** model, even when \(T\) is multilingual) |

Completions are JSON objects with a `tokens` list. Maninka group code is
**mku** (not `mlq`). GRPO / DPO / APO are **policy-update plugins**; Φ and
Distiller stay the same.

## Read next

1. [Getting started](getting-started.md) — install, dataset shape, happy path
2. [Experiments](experiments.md) — `sebeni exp` on packaged raw / test data
3. [Use cases](use-cases.md) — ten recipes with your own jsonl
4. [SAMPG](sampg.md) — how the training loop works
