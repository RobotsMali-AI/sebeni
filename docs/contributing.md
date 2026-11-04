# Contributing / code style

Match the existing dialect: dataclasses with type hints, TRL-wrapping classes
(`SebeniGrpo`, `RewardManager`, `Distiller`, `DabaX`), numpy-style or short class
docstrings, `beni.*` imports.

- Do not introduce a second config library or rename public APIs.
- Public functions and protocols get numpy-style docstrings (mkdocstrings).
- When editing a file, fix only the touched region. No repo-wide reformat.
- **Ruff**: `format` + `check`, Python 3.10, double quotes, line length 100,
  lint rules `E4`, `E7`, `E9`, `F` (see `[tool.ruff]` in `pyproject.toml`).
  Run it on new/changed files. Ships with the `[dev]` and `[docs]` extras.

Install for docs / tests. The full test suite imports the PyTorch/TRL
trainers, so include `[train]`; Daba is installed separately without its GUI
dependencies:

```bash
pip install -e ".[train,docs,dev]"
pip install "daba @ git+https://github.com/maslinych/daba.git" --no-deps
# or from GitHub:
# pip install "sebeni[train,docs,dev] @ git+https://github.com/mlsftwrs/sebeni.git"
```

Tests that need Daba's parser or `google-genai` are skipped when those are not
installed.

Docs: [seben.robotsmali.org/docs](https://seben.robotsmali.org/docs).

```bash
ruff format beni tests
ruff check beni tests
pytest                 # testpaths = tests, run from the repo root
mkdocs serve
mkdocs build --strict  # what CI runs (.github/workflows/docs.yml)
```

CI installs `pip install -e ".[docs]"` and runs `mkdocs build --strict`, so a
broken link or an unresolved mkdocstrings `:::` target fails the build.
