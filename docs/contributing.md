# Contributing / code style

Match the existing dialect: dataclasses with type hints, TRL-wrapping classes
(`SebeniGrpo`, `RewardManager`, `Distiller`, `DabaX`), numpy-style or short class
docstrings, `beni.*` imports.

- Do not introduce a second config library or rename public APIs.
- Public functions and protocols get numpy-style docstrings (mkdocstrings).
- When editing a file, fix only the touched region. No repo-wide reformat.
- **Ruff**: `format` + `check`, Python 3.10, double quotes, line length 100.
  Run it on new/changed files. Optional `[dev]` extra.

Install for docs / tests:

```bash
pip install -e ".[docs,dev]"
# or from GitHub:
# pip install "sebeni[docs,dev] @ git+https://github.com/mlsftwrs/sebeni.git"
```

Docs: [seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

```bash
ruff format beni tests
ruff check beni tests
pytest
mkdocs serve
```
