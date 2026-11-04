# Model and trainer hyperparameters

Every field on `ModelConfig`, `GRPOTrainerConfig`, `DPOTrainerConfig`, and
`APOTrainerConfig` is a YAML key under `model:`, `trainer:`, `dpo:`, or `apo:`.
`sebeni train` overlays the same knobs from the CLI (non-None flags win).

Docs site: [seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

```bash
sebeni train -c config.yaml \
  --lr 1e-5 --batch-size 4 --grad-accum 4 --max-steps 100 \
  --beta 0.04 --num-generations 8 --temperature 0.8 \
  --lora-r 32 --lora-alpha 64 --bf16 --grad-checkpoint
```

Setting `--epochs` without `--max-steps` sets `max_steps: -1` so Hugging Face
runs by epoch count.

## `model:` (LoRA / quantization)

| Key | Default | CLI |
| --- | --- | --- |
| `model_name` | `HuggingFaceTB/SmolLM2-135M` | (YAML) |
| `ref_model_name` | `null` | (YAML) |
| `flax_model_name` | `null` | (YAML; Flax checkpoint for `trainer.framework: jax`, else `model_name` is probed) |
| `load_in_4bit` | `true` | `--load-in-4bit` / `--no-load-in-4bit` |
| `bnb_4bit_quant_type` | `nf4` | (YAML) |
| `bnb_4bit_use_double_quant` | `true` | (YAML) |
| `use_peft` | `true` | `--peft` / `--no-peft` |
| `lora_r` | `16` | `--lora-r` |
| `lora_alpha` | `32` | `--lora-alpha` |
| `lora_dropout` | `0.1` | `--lora-dropout` |
| `lora_target_modules` | `q_proj v_proj k_proj o_proj` | (YAML list) |
| `lora_bias` | `none` | (YAML) |

Base (non-instruct) checkpoints such as `SmolLM2-135M` often ship without a
`chat_template`; Sebeni then sets a ChatML template on the tokenizer so TRL can
format prompts. 4-bit loading is only used when CUDA and bitsandbytes are
available.

## `trainer:` (GRPO / TRL `GRPOConfig`)

`framework: torch` (default) uses TRL. `framework: jax` (always read from
`trainer:`, even when `algorithm` is `dpo` or `apo`; needs the `[jax]` extra)
uses Flax/Optax for GRPO, DPO, or APO when Flax weights exist. If JAX or the
Flax weights cannot be loaded, Sebeni warns with the failed probe and continues
with torch/TRL; set `model.flax_model_name` to pin a separate Flax checkpoint.
Unknown TRL constructor names are dropped at train time so the same YAML
works on TRL 0.x (`max_prompt_length`, `warmup_ratio`) and TRL 1.x
(`max_completion_length`, `warmup_steps`). DPO maps `max_prompt_length`
onto `max_length` when the installed TRL no longer has the prompt field.

| Key | Default | CLI |
| --- | --- | --- |
| `learning_rate` | `5e-6` | `--lr` |
| `per_device_train_batch_size` | `2` | `--batch-size` |
| `gradient_accumulation_steps` | `8` | `--grad-accum` |
| `max_prompt_length` | `1024` | `--max-prompt-length` (ignored on TRL 1.x GRPO; kept for 0.x / DPO mapping) |
| `max_completion_length` | `1024` | `--max-completion-length` |
| `max_steps` | `10` | `--max-steps` |
| `num_train_epochs` | `1.0` | `--epochs` |
| `logging_steps` | `1` | (YAML) |
| `save_steps` | `50` | `--save-steps` |
| `save_total_limit` | `2` | (YAML) |
| `max_grad_norm` | `0.1` | (YAML) |
| `beta` | `0.1` | `--beta` |
| `num_generations` | `4` | `--num-generations` |
| `num_iterations` | `1` | (YAML) |
| `temperature` | `0.9` | `--temperature` |
| `top_p` | `1.0` | (YAML) |
| `top_k` | `50` | (YAML) |
| `warmup_ratio` | `0.0` | `--warmup-ratio` (ignored on TRL 1.x; use `warmup_steps`) |
| `warmup_steps` | `0` | `--warmup-steps` |
| `weight_decay` | `0.0` | `--weight-decay` |
| `lr_scheduler_type` | `cosine` | `--lr-scheduler` |
| `seed` | `42` | `--seed` |
| `optim` | `adamw_torch` | `--optim` |
| `bf16` | `false` | `--bf16` / `--no-bf16` |
| `fp16` | `false` | `--fp16` / `--no-fp16` |
| `gradient_checkpointing` | `false` | `--grad-checkpoint` / `--no-grad-checkpoint` |
| `dataloader_num_workers` | `0` | (YAML) |
| `use_cpu` | `false` | `--use-cpu` / `--no-use-cpu` |
| `framework` | `torch` | (YAML; `torch` or `jax`) |
| `output_dir` | `{working_dir}/models` | (YAML) |
| `report_to` | `trackio` | (YAML; `trackio`, `wandb`, or `none`. With `trackio`, TRL gets `project=<project_name>` and `run_name=run-<timestamp>` so Trainer logs into that Trackio project, not `huggingface`.) |
| `push_to_hub` | `false` | (YAML) |
| `hub_model_id` | `null` | (YAML; default for `sebeni push --repo-id`) |
| `hub_strategy` | `end` | (YAML) |
| `hub_token` | `null` | (YAML) |
| `hub_private_repo` | `true` | (YAML) |

## `dpo:` / `apo:` (TRL `DPOConfig`)

DPO uses the `dpo:` block (`loss_type: sigmoid`); APO uses `apo:`
(`loss_type: apo_zero` or `apo_down`). Both carry `learning_rate`,
`per_device_train_batch_size`, `gradient_accumulation_steps`,
`max_prompt_length` (`1024`), `max_length` (`2048`), `max_steps`,
`num_train_epochs`, `logging_steps`, `save_steps`, `output_dir`,
`max_grad_norm`, `report_to` (`trackio`), `beta`, `loss_type`, `warmup_ratio`,
`weight_decay`, `lr_scheduler_type`, `seed`, `bf16`, `fp16`, `use_cpu`, and the
`push_to_hub` / `hub_*` keys, with the same defaults as `trainer:`.

When `algorithm` is `dpo` or `apo`, the optimizer CLI flags (`--lr`,
`--batch-size`, `--grad-accum`, `--max-steps`, `--epochs`, `--beta`,
`--warmup-ratio`, `--weight-decay`, `--seed`, `--save-steps`,
`--max-prompt-length`, `--lr-scheduler`, `--bf16`, `--fp16`, `--use-cpu`)
overlay onto that block instead of `trainer:`. `--num-generations`,
`--max-completion-length`, `--temperature`, and `--grad-checkpoint` always
write to `trainer:`.

## `data:` (multilingual)

| Key | Meaning |
| --- | --- |
| `languages` | List or comma-separated ISO / group codes for this run |
| `default_lang` | Fallback for unlabeled rows |
| `source` | Path, glob, list of paths, or Hugging Face id |
| `scheme` | `completion` (GRPO) or `preference` (DPO/APO) |
| `encoding` | File encoding (`utf-8-sig`) |
| `file_text_key` / `file_lang_key` | JSON / JSONL key names (`text` / `lang`) |
| `hf_split` / `hf_text_key` / `hf_lang_key` / `hf_streaming` / `hf_kwargs` | Hugging Face dataset loading (`hf_split: train`) |

`--lang` on `init`, `train`, `distill`, `eval`, and `wordfreq` is repeatable.

`sebeni exp` ignores `data.source` and always uses packaged raw / test splits.
See [Experiments](experiments.md).

## What the CLI does *not* replace

Reward weights, SafetyGovernor gates, Distiller `backend` / `model` / `tau` /
`gguf_path` / `n_ctx` / `max_input_chars`, `trainer.framework`, `report_to`,
and Hub tokens stay in YAML (or the environment for API keys). Only
`distillation.hitl` has a CLI switch (`--hitl` / `--no-hitl`).
