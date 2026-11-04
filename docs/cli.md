# CLI and config

Public docs: [https://seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

Console script: `sebeni` (Typer). Autodoc of the Typer `app` object is
intentionally omitted — it has no stable inspect signature and used to break
the API pages.

```
sebeni init     --lang bam --lang mku -w ./runs/manding-001
sebeni train    -c ./runs/manding-001/config.yaml --lr 1e-5 --lora-r 32
sebeni train    -c config.yaml -w /scratch/sebeni --max-steps 100 --beta 0.04
sebeni exp      -c configs/exp.yaml -w ./runs/exp-001
sebeni distill  -c config.yaml
sebeni wordfreq -c config.yaml
sebeni eval     -c config.yaml
sebeni push     -c config.yaml --repo-id mlsftwrs/sebeni-bam-grpo
sebeni generate -c config.yaml --prompt "Aw ka kɛnɛ wa?" --lang bam
```

YAML overlays `MasterConfig` dataclasses (no second config system). Examples:
`configs/exp.yaml`, `configs/grpo_bam.yaml`, `configs/grpo_multilang.yaml`,
`configs/dpo_bam.yaml`, `configs/apo_bam.yaml`. Full trainer / LoRA tables:
[Hyperparameters](hyperparams.md).

## Commands

| Command | Purpose | Important flags |
| --- | --- | --- |
| `sebeni init` | Write `config.yaml` + workdir (`data/`, `models/`, `runs/`, `exp/`, `runtime/`) | `--lang` repeatable (default `bam`), `-w` (default `./runs/sebeni-001`) |
| `sebeni train` | SAMPG (Φ / τ / Distiller) then policy-update plugin | `-c`, `-w`, `--lang`, `--hitl/--no-hitl`, `--lr`, `--batch-size`, `--grad-accum`, `--max-steps`, `--epochs`, `--beta`, `--num-generations`, `--max-prompt-length`, `--max-completion-length`, `--temperature`, warmup / weight-decay / seed / save / optim / scheduler, LoRA / 4-bit / PEFT / bf16 / fp16 / grad-checkpoint / CPU flags (full list: [Hyperparameters](hyperparams.md)) |
| `sebeni exp` | Packaged multilingual train + eval (`experiment` alias) | `-c` (defaults to packaged `exp.yaml`), `-w` |
| `sebeni distill` | Distiller only (no policy step) | `-c`, `-w`, `--lang`, `--hitl/--no-hitl` |
| `sebeni eval` | Φ per language → `{working_dir}/exp/eval.json` + safety snapshot | `-c`, `-w`, `--lang` |
| `sebeni wordfreq` | Surfaces / lemmas / morphemes / stages per language → `{working_dir}/exp/wordfreq/` (`wordfreq.json` index + one folder per language) | `-c`, `-w`, `--lang` |
| `sebeni generate` | Decode from saved policy; warn on `R_format` / `R_lang` | `-c`, `-w`, `--prompt`, `--lang` (single code), `--max-length` (default `128`) |
| `sebeni push` | Hub upload (card + snapshot required) | `-c`, `-w`, `--repo-id` (defaults to `trainer.hub_model_id`) |

`--lang` on `init`, `train`, `distill`, `eval`, and `wordfreq` is repeatable
or comma-separated (`--lang bam --lang mku` or `--lang bam,mku`).

Setting `--epochs` without `--max-steps` sets `max_steps: -1` so Hugging Face
runs by epoch count.

## `eval.json` shape

```json
{
  "phi": 0.63,
  "tau": 0.5,
  "n_sentences": 128,
  "languages": ["bam", "mku"],
  "by_language": {
    "bam": {"phi": 0.71, "n_sentences": 80, "checkpoint_id": "baseline_v3", "language": "bam"},
    "mku": {"phi": 0.48, "n_sentences": 48, "checkpoint_id": "baseline_v1", "language": "mku"}
  },
  "algorithm": "grpo"
}
```

## MasterConfig fields

| Field | Meaning |
| --- | --- |
| `project_name` | Trackio project (each run is named `run-<timestamp>`) |
| `algorithm` | `grpo` (default) / `dpo` / `apo` |
| `working_dir` | Relocatable root |
| `model` | `ModelConfig` (base, ref, optional `flax_model_name` for JAX, LoRA rank/alpha/dropout, 4-bit) |
| `processor` | `DabaXProcessorConfig` (DabaX `runtime_dir`, baseline `data_dir`, tokenizer, baseline file names) |
| `data` | `DataConfig` (`source`, `scheme`, `languages`, `default_lang`, file / Hugging Face loader keys) |
| `trainer` | `GRPOTrainerConfig` → TRL `GRPOConfig` (lr, batch, β, generations, Hub, `report_to`, `framework: torch` / `jax`) |
| `dpo` / `apo` | Policy-update sibling configs (shared optimizer fields plus `max_length`, `loss_type`, their own `report_to`) |
| `distillation` | `enabled`, `backend` (`algorithmic` default, `gguf`, `google`, `openai`, `groq`, `together`; `provider` is a deprecated alias), `model`, `tau` (0.5), `hitl`, `vertex`, `base_url`, `gguf_path`, `n_ctx` (4096), `max_input_chars` (8000), `batch_size`, `auto_update_baselines`, `working_dir` |
| `wordfreq` | `raw_inputs` for `sebeni wordfreq` (falls back to `data.source`, then packaged `beni/data/raw`) |
| `reward` | Weights and `enable_*` switches for R_format, R_morph, R_rule, R_lang; `custom_reward_weights` |
| `safety` | `SafetyConfig` → `SafetySpec` |
| `experiment` | `kveritas` / `kveritas_seal` for `sebeni exp` |

Env keys: `SEBENI_HOME`, `SEBENI_WORKING_DIR`, `SEBENI_ENV` (extra `.env`
file), `GOOGLE_API_KEY`, `GOOGLE_PROJECT_ID` / `GOOGLE_LOCATION` (Vertex),
`OPENAI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`. `.env` files
in the cwd, the config file's directory, and `{working_dir}` are loaded
without overriding variables that are already set.

Extras: `[train]`, `[wandb]`, `[distil]`, `[gguf]` (llama-cpp-python for the
local Distiller), `[jax]` (JAX / Flax / Optax / Orbax for
`trainer.framework: jax`), `[docs]`, `[dev]`. Install the
headless parser separately with
`pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps`;
`[gui]` is wxPython for upstream gparser only.

## Python equivalent of `sebeni train`

```python
from beni.core.srl.config import MasterConfig
from beni.core.srl.unified import SRLTrainer

cfg = MasterConfig.from_yaml("config.yaml")
trainer = SRLTrainer(cfg)
trainer.train([{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}])
```
