# Hub model cards

Every `save_model` writes `README.md` (the model card,
`beni.core.hub.model_card.write_model_card`) and `safety_snapshot.json` into
`output_dir`. `push_to_hub` writes them first if either is missing, then asks
SafetyGovernor; with `safety.require_model_card` / `require_safety_snapshot`
(both default on) the push is refused if they are still absent.

The card is rendered from `beni/core/hub/templates/model_card.md.j2` and
includes HF YAML frontmatter (`language` list, `base_model`,
`library_name: transformers`, tags, `license: mit`), intended use (ELRL
morphological generation — not a chatbot), limitations, training recipe
(algorithm, base model, languages/groups (MKU not MLQ), τ, last Φ, baseline
**grammar/dictionary** checkpoint id, reward weights, safety gates that fired),
a YAML snapshot of the full config, an eval table (Phi, MER, MCS, format
validity, R_lang), and links to [seben.robotsmali.org](https://seben.robotsmali.org)
and the package docs ([mlsftwrs.github.io/sebeni](https://mlsftwrs.github.io/sebeni/)).

```python
from beni.core.hub import write_model_card
from beni.core.srl.config import MasterConfig

mc = MasterConfig.from_yaml("config.yaml")
write_model_card(mc.trainer.output_dir, config=mc)  # -> output_dir/README.md
```

Trained policies are Hub model repos. **Source code** stays on GitHub
([mlsftwrs/sebeni](https://github.com/mlsftwrs/sebeni)).

## Push

```bash
huggingface-cli login
sebeni push -c config.yaml --repo-id mlsftwrs/sebeni-manding-grpo
```

`--repo-id` defaults to `trainer.hub_model_id`; `hub_private_repo` and
`hub_token` are passed through. Token via `trainer.hub_token`, `HF_TOKEN`, or
`huggingface-cli login` (write access on org `mlsftwrs`). With
`trainer.push_to_hub: true`, GRPO training pushes automatically after saving
(DPO reads the same keys from the `dpo:` section).

```yaml
trainer:
  push_to_hub: true
  hub_model_id: mlsftwrs/sebeni-manding-grpo
  hub_private_repo: true   # set false for a public model
  # hub_token: null        # prefer HF_TOKEN / huggingface-cli login
```

`sebeni push` regenerates `README.md` (card) and `safety_snapshot.json` if
they are missing and refuses to upload if SafetyGovernor still cannot find them.

## Transfer into Hugging Face org `mlsftwrs`

The Hub org holds **models / datasets / Spaces**. It does not replace GitHub
for this source tree (Pages still need `mlsftwrs/sebeni` on GitHub). Hugging
Face does **not** move a GitHub repository.

1. Create [huggingface.co/organizations/new](https://huggingface.co/organizations/new) named `mlsftwrs`.
2. Invite pushers (**write**) and people who will move repos (**admin**).
3. Issue a token with write (and org) scope; `huggingface-cli login`. The token’s
   user must be a member of `mlsftwrs`.
4. **Existing** Hub repos: owner + org admin → **Settings → Transfer repository** → `mlsftwrs`.
5. **This git tree**: nothing to transfer on the Hub until `sebeni push` creates a model repo.
6. Then set `hub_model_id: mlsftwrs/<name>`. Re-add collaborators via the org if
   they were individual users on the old repo. Spaces that referenced the old
   model id need their env/secrets updated.

Optional Hub extras:

- Organization card at `https://huggingface.co/mlsftwrs`
- Dataset repo `mlsftwrs/<name>` if you publish the (text, lang) corpus
- Keep models private (`hub_private_repo: true`) until the card/snapshot look right
- Do not vendor GPL Daba sources into a Hub model repo; the card already credits
  [maslinych/daba](https://github.com/maslinych/daba)
