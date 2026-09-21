"""OpenAI-compatible distillation providers (OpenAI, Groq, Together)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from beni.core.morphotactic.distil.providers.base import BaseProvider, ProviderCapability
from beni.utils import config as cfg


def _parse_distil_output(payload: Any):
    from beni.core.morphotactic.distil import DistilOutput

    if isinstance(payload, DistilOutput):
        return payload
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    try:
        return DistilOutput.model_validate(payload)
    except Exception:
        gram = payload.get("sebeni_gram") or payload.get("gram") or ""
        ldict = payload.get("sebeni_dict") or payload.get("dict") or ""
        if gram or ldict:
            return DistilOutput(sebeni_gram=str(gram), sebeni_dict=str(ldict))
        return None


class OpenAICompatibleProvider(BaseProvider):
    """HTTP Chat Completions provider (OpenAI-compatible JSON).

    Cache/upload is ``ProviderCapability.NONE``; Distiller skips cache when
    the capability is not cache/both.
    """

    base_url: str = "https://api.openai.com/v1"
    env_key: str = "OPENAI_API_KEY"
    cache = None

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", **kwargs: Any):
        key = api_key or os.getenv(self.env_key) or cfg.provider_api_key(self.env_key.split("_")[0].lower())
        configured_url = kwargs.get("base_url")
        if configured_url:
            self.base_url = str(configured_url)
            key = key or "local"
        super().__init__(key or "", model)
        self.language = kwargs.get("language")
        self.temperature = float(kwargs.get("temperature", cfg.TEMPERATURE))
        self.cache = None

    def initialize(self) -> Any:
        return None

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability.NONE

    def create_cache(self, *args, **kwargs) -> None:
        return None

    def generate(
        self,
        prompt: str,
        sys_instruct: Optional[str] = None,
        indicator: bool = False,
        system_instruction: Optional[str] = None,
        **kwargs,
    ):
        """Generate a DistilOutput JSON object via Chat Completions."""
        from beni.core.morphotactic.distil import DistilOutput

        if not self.api_key:
            raise ValueError(f"{self.env_key} is required for {self.__class__.__name__}")

        system = sys_instruct or system_instruction or ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        def _call():
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            data = self._retry_generate(_call)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{self.__class__.__name__} HTTP {exc.code}: {exc.read()[:400]!r}") from exc

        content = ""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected provider payload: {data!r}") from exc

        parsed = _parse_distil_output(content)
        if parsed is None:
            return DistilOutput(sebeni_gram="", sebeni_dict="")
        return parsed


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI Chat Completions (``OPENAI_API_KEY``)."""

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    env_key = "OPENAI_API_KEY"


class GroqProvider(OpenAICompatibleProvider):
    """Groq OpenAI-compatible API (``GROQ_API_KEY``)."""

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    env_key = "GROQ_API_KEY"


class TogetherProvider(OpenAICompatibleProvider):
    """Together AI OpenAI-compatible API (``TOGETHER_API_KEY``)."""

    base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
    env_key = "TOGETHER_API_KEY"
