"""Hugging Face model-card writer (Jinja). Required before Hub push."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape

from beni.core.safety.governor import SafetySnapshot

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_NAME = "model_card.md.j2"

PROJECT_URL = "https://seben.robotsmali.org"
DOCS_URL = "https://mlsftwrs.github.io/sebeni/"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        keep_trailing_newline=True,
    )


def write_model_card(
    output_dir: Union[str, Path],
    *,
    snapshot: Optional[SafetySnapshot] = None,
    config: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render ``README.md`` into ``output_dir`` for Hub upload.

    Parameters
    ----------
    output_dir : path
        Trainer output directory (card is written as README.md).
    snapshot : SafetySnapshot, optional
        Φ, τ, checkpoint id, eval table.
    config : MasterConfig, optional
        Used for algorithm, reward weights, and a YAML snapshot.
    extra : dict, optional
        Template overrides.

    Returns
    -------
    Path
        Path to the written README.md.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snap = snapshot.to_dict() if snapshot is not None else {}
    algorithm = getattr(config, "algorithm", None) if config is not None else None
    model_name = None
    reward = None
    if config is not None:
        model_name = getattr(getattr(config, "model", None), "model_name", None)
        reward = getattr(config, "reward", None)
        algorithm = algorithm or getattr(config, "algorithm", "grpo")

    config_snapshot = ""
    if config is not None:
        try:
            import yaml
            from dataclasses import asdict, is_dataclass

            payload = asdict(config) if is_dataclass(config) else {}
            payload.pop("_config_file_dir", None)
            config_snapshot = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        except Exception:
            config_snapshot = str(config)

    ctx: Dict[str, Any] = {
        "language": snap.get("language") or getattr(getattr(config, "data", None), "default_lang", "bam"),
        "languages": snap.get("languages") or [],
        "group_code": snap.get("group_code"),
        "base_model": model_name or "unknown",
        "algorithm": snap.get("algorithm") or algorithm or "grpo",
        "license": "mit",
        "model_name": getattr(getattr(config, "trainer", None), "hub_model_id", None) or model_name or "sebeni-model",
        "tau": snap.get("tau", getattr(getattr(config, "distillation", None), "tau", 0.5)),
        "phi": snap.get("phi"),
        "mer": snap.get("mer"),
        "mcs": snap.get("mcs"),
        "format_validity": snap.get("format_validity"),
        "r_lang": snap.get("r_lang"),
        "checkpoint_id": snap.get("checkpoint_id") or "baseline",
        "format_weight": getattr(reward, "format_weight", None),
        "morph_weight": getattr(reward, "morph_weight", None),
        "rule_weight": getattr(reward, "rule_weight", None),
        "lang_weight": getattr(reward, "lang_weight", None),
        "gates_fired": ", ".join(snap.get("gates_fired") or []) or "(none)",
        "config_snapshot": config_snapshot.strip(),
        "project_url": PROJECT_URL,
        "docs_url": DOCS_URL,
    }
    if config is not None and hasattr(config, "languages"):
        ctx["languages"] = ctx.get("languages") or config.languages()
    if extra:
        ctx.update(extra)
    if not ctx.get("languages"):
        lang = ctx.get("language") or "bam"
        ctx["languages"] = [s.strip() for s in str(lang).split(",") if s.strip()]

    text = _env().get_template(_TEMPLATE_NAME).render(**ctx)
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path
