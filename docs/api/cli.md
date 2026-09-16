# `beni.cli`

Console script `sebeni`. Autodoc of the Typer `app` object is omitted (it has
no stable inspect signature and broke the API pages). Commands:

| Command | Purpose |
| --- | --- |
| `sebeni init` | Write `config.yaml` + workdir (`--lang` repeatable) |
| `sebeni train` | SAMPG + policy-update plugin; `--lr`, `--lora-r`, … |
| `sebeni exp` | Packaged multilingual train + eval (`experiment` alias) |
| `sebeni distill` | Distiller only |
| `sebeni eval` | Φ per language → `exp/eval.json` |
| `sebeni wordfreq` | Surfaces / lemmas / morphemes / stages |
| `sebeni generate` | Policy decode + R_format / R_lang warnings |
| `sebeni push` | Hub upload (card + snapshot required) |

```bash
sebeni init --lang bam --lang mku -w ./runs/manding-001
sebeni train -c ./runs/manding-001/config.yaml --lr 1e-5 --max-steps 20
sebeni exp -c configs/exp.yaml -w ./runs/exp-001
sebeni --help
sebeni train --help
```

::: beni.cli.main.default_config_yaml
    options:
      show_root_heading: true
      heading_level: 2

See [CLI and config](../cli.md) and [Use cases](../use-cases.md).
