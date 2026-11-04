# Morphotactic: Distiller and DabaX

Distiller in SAMPG proposes G, D when Φ < τ. DabaX wraps CLI `daba.mparser`
(no wxPython). Backends and their options are described in
[Distillation and DabaX](../distillation.md).

```python
from beni.core.morphotactic.distil.distillation import Distiller
from beni.core.morphotactic.dabax import DabaX, get_dabax

d = Distiller(lang_code="bam", backend="algorithmic", working_dir="./runs/bam")
d.handle_baselines()
phi = d.phi_on_texts(["Aw ka kɛnɛ wa?"])
proposal = d.propose(["Aw ka kɛnɛ wa?"])  # DistillProposal or None

dx = DabaX("bam", gram=d.gram_path, ldict=d.dict_path, process=True)
sentences = dx.loader("Aw ka kɛnɛ wa?")
dx = get_dabax("bam")  # cached; latest workdir checkpoint, else packaged
```

`backend` takes `algorithmic` (default), `google` / `gemini`, `openai`, `groq`,
`together` or `gguf`; `provider=` is an older alias. LLM backends also accept
`model`, `api_key`, `vertex`, `base_url`, `gguf_path`, `n_ctx` and
`max_input_chars`.

::: beni.core.morphotactic.distil.distillation.Distiller
    options:
      members:
        - handle_baselines
        - phi_on_texts
        - collect_misses
        - files_parseable
        - propose
        - write_checkpoint
        - run_distillation
        - is_scratch
        - is_first_create
        - checkpoint_id
      show_root_heading: true
      heading_level: 2
      inherited_members: false
      show_if_no_docstring: false

::: beni.core.morphotactic.distil.distillation.DistillProposal
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.morphotactic.dabax.DabaX
    options:
      members:
        - loader
        - parse
      show_root_heading: true
      heading_level: 2
      inherited_members: false
      show_if_no_docstring: false

::: beni.core.morphotactic.dabax.get_dabax
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.morphotactic.dabax.resolve_baseline_files
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.morphotactic.dabax.clear_dabax_cache
    options:
      show_root_heading: true
      heading_level: 2

::: beni.core.morphotactic.distil.providers.create_provider
    options:
      show_root_heading: true
      heading_level: 2
