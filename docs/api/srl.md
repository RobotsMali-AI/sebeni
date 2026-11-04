# `beni.core.srl`

SAMPG driver and policy-update plugins. Import config **without** torch:

```python
from beni.core.srl.config import MasterConfig
from beni.core.srl.unified import SRLTrainer, register_algorithm

cfg = MasterConfig.from_yaml("config.yaml")
trainer = SRLTrainer(cfg)          # plugin = GRPO / DPO / APO, or JAX if trainer.framework: jax
trainer.train([{"text": "Aw ka kɛnɛ wa?", "lang": "bam"}])
# trainer.load_models / save_model / generate / push_to_hub → plugin
# trainer.register_reward(name, fn, weight) / register_hook(fn) → plugin
```

Register a custom policy-update plugin:

```python
register_algorithm("my_gc", MyPlugin)  # YAML algorithm: my_gc
```

::: beni.core.srl.config.MasterConfig
    options:
      members:
        - from_yaml
        - from_dict
        - languages
        - apply_cli_overrides
        - apply_workdir
      show_root_heading: true
      heading_level: 2
      merge_init_into_class: false
      show_if_no_docstring: false

::: beni.core.srl.unified.SRLTrainer
    options:
      members:
        - train
      show_root_heading: true
      heading_level: 2
      inherited_members: false
      show_if_no_docstring: false

::: beni.core.srl.unified.register_algorithm
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.srl.algorithm1.maybe_distill_languages
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.srl.algorithm1.maybe_distill_batch
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.srl.algorithm1.group_texts_by_language
    options:
      show_root_heading: true
      heading_level: 2

## Plugin base and callbacks

::: beni.core.srl.plugin.AlignmentPlugin
    options:
      members:
        - load_models
        - register_reward
        - register_hook
        - save_model
        - push_to_hub
      show_root_heading: true
      heading_level: 3
      show_if_no_docstring: false

::: beni.core.srl.grpo.callbacks.SelfAwareCallback
    options:
      members:
        - process_inputs
      show_root_heading: true
      heading_level: 3

::: beni.core.srl.grpo.callbacks.format_gradient_mask_hook
    options:
      show_root_heading: true
      heading_level: 3

::: beni.core.srl.grpo.callbacks.distillation_hook
    options:
      show_root_heading: true
      heading_level: 3

## JAX backend

Selected by `trainer.framework: jax` (install `sebeni[jax]`). Falls back to
the PyTorch/TRL plugin when Flax weights cannot be loaded.

::: beni.core.srl.jax.policy.JaxPolicyPlugin
    options:
      members:
        - load_models
        - train
        - save_model
      show_root_heading: true
      heading_level: 3
      show_if_no_docstring: false

::: beni.core.srl.jax.policy.jax_uncertainty_scale
    options:
      show_root_heading: true
      heading_level: 3
