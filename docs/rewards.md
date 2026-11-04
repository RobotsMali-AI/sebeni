# Rewards and metrics

Shared \(\pi_{rf}\) for every algorithm:

\[
R_{\mathrm{total}} = \omega_f R_{\mathrm{format}} + \omega_m R_{\mathrm{morph}} + \omega_r R_{\mathrm{rule}} + \omega_\ell R_{\mathrm{lang}}
\]

Sebeni defaults (YAML `reward:`): format **0.2**, morph **0.4**, rule **0.4**,
**lang 0.2**. Each term \(R \in [0, 1]\); the weight is applied **inside** the
reward function, so `reward_format` returns \(\omega_f R_{\mathrm{format}}\)
and TRL sums the functions with unit weight. GRPO stays **outcome-level** on
the completion; token stages are process signal. Daba itself is the
verifiable reward — no separate reward model.

## Metrics

All metrics live on `MorphologyScorer(beta=0.1, eps=1e-8, empr=0.5)`.

**Phi** — mean `stage_to_phi` over tokens. Stage \(< 0\) (Daba −1) → 0;
analyzed stages 0–5 → 1.0; borrowing EMPR / stage 6 → `empr` (0.5); stage
\(> 6\) → \(\epsilon\); non-numeric parser labels → 1.0. An empty sentence
scores 0. `phi_corpus` returns per-text scores plus their `avg`. Trigger
Distiller if batch Φ < τ (0.5).

**MER** — morpheme Levenshtein \((S_m+D_m+I_m)/N_m\) vs the reference parse.
\(N_m\) is the reference morpheme count. 0 when both sides are empty; \(\infty\)
when only the reference is empty. `mer_sentence(ref, hyp, agg=...)` uses
`"mean"` (per-token, default), `"sum"`, or `"micro"` (one sequence over all
tokens' best morphemes; used by \(R_{morph}\)). Lower is better.

**MCS** — fraction of token positions whose predicted `stage` equals Daba's
stage, divided by \(\max(|T_{pred}|, |T_{ref}|)\) (self-awareness: no fake
stage=1). `mcs_corpus` micro-averages over sentences. Unaware hallucination is
claiming a valid stage for a token Daba marks −1.

**U** (uncertainty-weighted cost error, UWEC) —
\(I(\text{stage}_{pred}\ne-1) +
\beta \log((\pi_\theta+\epsilon)/(\pi_{ref}+\epsilon))\). The stage indicator
is averaged over morphological JSON tokens; the log ratio is averaged over LM
completion tokens. They are not positionally zipped.
`uncertainty_weighted_cost_error` scores one token;
`uncertainty_weighted_cost_error_sentence` averages a sentence.

U is reward distrust, not a fifth reward and not a replacement for Φ. Torch
and JAX scale the update by \(1/(1+\operatorname{relu}(U))\). Each policy
step logs `u_indicator`, `u_kl`, and `uncertainty` to Trackio, and the last
values are written into `safety_snapshot.json` when the model is saved.
`TrackioMetricsCallback` also logs the running reward total as
`sebeni/reward`.

`MorphologyScorer.evaluate(predicted, reference)` returns `mer`, `mcs`, and
`phi` for one sentence, or corpus versions for lists of sentences.

## Rewards

| Term | Weight | Meaning |
| --- | --- | --- |
| \(R_{morph}\) | 0.4 | Annotation quality vs DabaX \(y^*\): mean of MCS, \(1-\operatorname{clip}(\mathrm{MER}_{micro}, 0, 1)\), and lemma overlap. 0 if either side has no tokens |
| \(R_{rule}\) | 0.4 | \(\tfrac12(\Phi + \text{lemma overlap})\): the completion's `text` is re-parsed with that row's active **G**, **D**, and its first-analysis lemmas are compared with \(y^*\) |
| \(R_{format}\) | 0.2 | Valid **JSON** object with a `tokens` list |
| \(R_{lang}\) | 0.2 | JSON `lang` resolves to the same group code as **that row**'s language |

Lemma overlap counts positions where the first analysis `form` matches a
non-empty reference form, divided by the longer token list. The reference
\(y^*\) comes from the dataset `reference` column (rebuilt from DabaX after a
promote); without it, the JSON in the prompt message is used.

Custom terms: `RewardManager.register_custom_reward(name, fn, weight)` (or
`register_reward` on a trainer) appends `fn` scaled by `weight` to
`get_reward_functions()`.

### Rule-encoded extras

The SAMPG design also describes finite-state checks in **G** — positional
\(\rho\) (e.g. `:n: [ :v: :n: ]` nominalization), lexical \(\lambda\)
(`\lx` / `\ps` / `\va` / `\fa` in **D**), and valence \(\nu\) (`\vl`
transitivity). These are **not** implemented in `reward_rule` yet; today
\(R_{rule}\) is Φ plus lemma overlap only.

### Format / self-awareness

Predicted `stage` matching Daba is rewarded through MCS inside
\(R_{morph}\); claiming `stage=1` where Daba says −1 earns nothing for that
token. Sebeni encodes this as JSON schema validity plus MCS when a reference
parse exists.

```json
{"text": "aw ka ne labato.", "lang": "bam", "tokens": [{"surface": "aw", "stage": 1, "analyses": []}]}
```

Invalid JSON, missing `tokens`, or a `lang` that is not this row's group
scores 0 for the corresponding term. A batch whose format scores are all 0
has its gradients zeroed — see [Safety](safety.md).

## YAML

```yaml
reward:
  format_weight: 0.2
  morph_weight: 0.4
  rule_weight: 0.4
  lang_weight: 0.2
  enable_format_reward: true
  enable_morph_reward: true
  enable_rule_reward: true
  enable_lang_reward: true
```

```python
from beni.core.compute.rewards import RewardManager

rm = RewardManager()
fmt = rm.reward_format(['{"text": "aw", "lang": "bam", "tokens": []}'])
lang = rm.reward_lang(
    ['{"text": "aw", "lang": "bam", "tokens": []}'],
    language=["bam"],
)
```
