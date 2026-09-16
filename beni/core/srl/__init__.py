from beni.core.srl.config import (
    ModelConfig,
    DabaXProcessorConfig,
    DataConfig,
    GRPOTrainerConfig,
    DPOTrainerConfig,
    APOTrainerConfig,
    DistillationConfig,
    RewardConfig,
    SafetyConfig,
    MasterConfig,
    SRLGrpoPrompt,
)
from beni.core.srl.unified import SRLTrainer, get_algorithm, register_algorithm

GrpoTrainerConfig = GRPOTrainerConfig


def __getattr__(name: str):
    """Lazy-load the GRPO plugin so ``import beni.core.srl`` works without torch."""
    if name == "SebeniGrpo":
        from beni.core.srl.grpo.grpo import SebeniGrpo
        return SebeniGrpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModelConfig",
    "DabaXProcessorConfig",
    "DataConfig",
    "GRPOTrainerConfig",
    "GrpoTrainerConfig",
    "DPOTrainerConfig",
    "APOTrainerConfig",
    "DistillationConfig",
    "RewardConfig",
    "SafetyConfig",
    "MasterConfig",
    "SRLGrpoPrompt",
    "SebeniGrpo",
    "SRLTrainer",
    "get_algorithm",
    "register_algorithm",
]
