"""Unified SAMPG driver wrapping policy-update plugins.

``SRLTrainer`` is the SAMPG driver: YAML → HITL/initial distill →
SelfAwareCallback (Φ, τ, Distiller) → policy plugin (GRPO/DPO/APO) → save θ, G, D
→ model card → optional Hub push.

Plugins only replace the policy-update step. Register a new method
with :func:`register_algorithm`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from beni.core.safety.governor import SafetyGovernor
from beni.core.srl.algorithm1 import confirm_hitl
from beni.core.srl.config import MasterConfig
from beni.utils import config as cfg


class AlignmentAlgorithm(Protocol):
    """Policy-update plugin: load models, train, save, push."""

    name: str

    def load_models(self, *args, **kwargs):
        ...

    def train(self, data, **kwargs):
        ...

    def save_model(self, output_dir=None):
        ...

    def push_to_hub(self, repo_id=None, **kwargs):
        ...


class DataSource(Protocol):
    """SAMPG samples o ~ π_θ(·|B). Offline pairs use scheme=preference."""

    def ds(self, *args, **kwargs):
        ...


class RewardFunction(Protocol):
    """π_rf: R_morph + R_format (JSON), plus optional R_rule / R_lang."""

    def get_reward_functions(self) -> list:
        ...


_ALGORITHM_REGISTRY: Dict[str, type] = {}


def register_algorithm(name: str, cls: type) -> None:
    """Register a policy-update plugin (one class + config, not a new outer loop)."""
    _ALGORITHM_REGISTRY[str(name).lower()] = cls


def get_algorithm(name: str) -> type:
    key = str(name or "grpo").lower()
    if key not in _ALGORITHM_REGISTRY:
        _load_builtins()
    if key not in _ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm {name!r}. Registered: {sorted(_ALGORITHM_REGISTRY)}"
        )
    return _ALGORITHM_REGISTRY[key]


def _load_builtins() -> None:
    if "grpo" not in _ALGORITHM_REGISTRY:
        from beni.core.srl.grpo.grpo import SebeniGrpo

        register_algorithm("grpo", SebeniGrpo)
    if "dpo" not in _ALGORITHM_REGISTRY:
        from beni.core.srl.dpo.dpo import SebeniDpo

        register_algorithm("dpo", SebeniDpo)
    if "apo" not in _ALGORITHM_REGISTRY:
        from beni.core.srl.apo.apo import SebeniApo

        register_algorithm("apo", SebeniApo)


class SRLTrainer:
    """SAMPG driver. Not an alias of ``SebeniGrpo``.

    Parameters
    ----------
    config : MasterConfig
        Loaded from YAML via ``MasterConfig.from_yaml``.
    """

    def __init__(self, config: Optional[MasterConfig] = None, **kwargs: Any):
        self.config = config or MasterConfig()
        if kwargs.get("model_name"):
            self.config.model.model_name = kwargs["model_name"]
        spec = self.config.safety.to_spec(
            tau=self.config.distillation.tau,
            kl_beta=self.config.trainer.beta,
        )
        self.governor = SafetyGovernor(spec)
        plugin_cls = get_algorithm(self.config.algorithm)
        self.plugin = plugin_cls(self.config)
        self.plugin.governor = self.governor

    def __getattr__(self, name: str):
        return getattr(self.plugin, name)

    def _records(self, data) -> List[Dict[str, Any]]:
        if data is None:
            return []
        if isinstance(data, list):
            return data
        try:
            return [{"text": row.get("text", ""), "lang": row.get("language") or row.get("lang")} for row in data]
        except TypeError:
            return []

    def _initial_distill(self, records: List[Dict[str, Any]]) -> None:
        if not self.config.distillation.enabled or not records:
            return
        from beni.core.morphotactic.distil.distillation import Distiller
        from beni.core.srl.algorithm1 import group_texts_by_language, records_text_langs

        default_lang = self.config.data.default_lang or "bam"
        lang_groups = group_texts_by_language(
            records_text_langs(records, default_lang),
            default_lang=default_lang,
        )
        allowed = set(self.config.languages()) if self.config.data.languages else None
        for group, texts in lang_groups.items():
            if allowed is not None and group not in allowed:
                continue
            distiller = Distiller(
                lang_code=group,
                provider=self.config.distillation.provider,
                model=self.config.distillation.model,
                working_dir=self.config.distillation.working_dir or self.config.working_dir,
            )
            distiller.handle_baselines()
            proposal = distiller.propose(texts)
            if proposal is None:
                continue
            if not confirm_hitl(
                proposal.gram_text, proposal.dict_text, self.config.distillation.hitl
            ):
                continue
            decision = self.governor.allow_promote(
                proposal.phi,
                proposal.phi_prime,
                parseable=proposal.parseable,
                first_create=proposal.first_create,
                language_ok=proposal.language_ok,
            )
            if decision.allowed:
                distiller.write_checkpoint(proposal.gram_text, proposal.dict_text)

    def train(self, data=None, project_name: Optional[str] = None, **kwargs):
        """HITL/initial distill → SelfAwareCallback → GC plugin → card / Hub."""
        if self.config.working_dir:
            cfg.set_working_dir(self.config.working_dir)
        records = self._records(data)
        self._initial_distill(records)
        return self.plugin.train(
            data,
            project_name=project_name,
            run_distillation_first=False,
            extra_callbacks=kwargs.get("extra_callbacks"),
        )
