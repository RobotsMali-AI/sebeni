
from beni.core.morphotactic.distil.providers import base


PROVIDER_REGISTRY = {
    "google": "beni.core.morphotactic.distil.providers.google:GoogleProvider",
    "gemini": "beni.core.morphotactic.distil.providers.google:GoogleProvider",
    "openai": "beni.core.morphotactic.distil.providers.openai_compat:OpenAIProvider",
    "groq": "beni.core.morphotactic.distil.providers.openai_compat:GroqProvider",
    "together": "beni.core.morphotactic.distil.providers.openai_compat:TogetherProvider",
    "gguf": "beni.core.morphotactic.distil.providers.gguf:GGUFProvider",
}


def _load_class(spec: str):
    module_name, cls_name = spec.split(":")
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def create_provider(name: str, api_key: str = None, model: str = None, **kw) -> base.BaseProvider:
    """Create a Distiller provider from the registry.

    Parameters
    ----------
    name : str
        ``google`` / ``gemini`` / ``openai`` / ``groq`` / ``together`` / ``gguf``.
    api_key : str, optional
        Falls back to the provider's environment variable.
    model : str, optional
        Provider model id.
    """
    spec = PROVIDER_REGISTRY.get(str(name or "").lower())
    if not spec:
        raise ValueError(f"Unsupported provider: {name}. Known: {sorted(PROVIDER_REGISTRY)}")
    provider_class = _load_class(spec)
    return provider_class(api_key, model, **kw)
