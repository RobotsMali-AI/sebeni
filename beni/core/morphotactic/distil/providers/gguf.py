"""Local GGUF Distiller provider backed by llama-cpp-python."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

from beni.core.morphotactic.distil.providers.base import BaseProvider, ProviderCapability
from beni.core.morphotactic.distil.providers.openai_compat import _parse_distil_output

logger = logging.getLogger(__name__)


class GGUFProvider(BaseProvider):
    """Run a local GGUF model for optional miss-conditioned refinement."""

    cache = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        gguf_path: Optional[str] = None,
        n_ctx: int = 4096,
        **kwargs: Any,
    ):
        path = gguf_path or model
        if not path:
            raise ValueError("distillation.gguf_path is required for backend: gguf")
        super().__init__("", str(path))
        self.n_ctx = int(n_ctx)
        message = (
            f"Local GGUF Distiller selected (n_ctx={self.n_ctx}). Small context windows "
            "and lack of hosted-model caching/optimization can reduce grammar quality and "
            "throughput. Prefer backend: algorithmic for training or an API/ADC model for "
            "HITL bootstrap."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        logger.warning(message)
        self._client = self.initialize()

    def initialize(self):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "GGUF support requires `pip install 'sebeni[gguf]'`."
            ) from exc
        return Llama(model_path=self.model, n_ctx=self.n_ctx, verbose=False)

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability.NONE

    def create_cache(self, *args, **kwargs):
        return None

    def generate(
        self,
        prompt: str,
        sys_instruct: Optional[str] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        from beni.core.morphotactic.distil import DistilOutput

        system = sys_instruct or system_instruction or ""
        response = self._client.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"]["content"]
        parsed = _parse_distil_output(content)
        return parsed or DistilOutput(sebeni_gram="", sebeni_dict="")
