"""
LLM provider abstractions for the reference generator.
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


class ProviderCapability(Enum):
    """Provider capabilities."""
    NONE = "none"
    UPLOAD = "upload"
    CACHE = "cache"
    BOTH = "both"


class BaseProvider(ABC):
    """Base LLM provider interface."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None
        self.retry_wait = 3
    
    @abstractmethod
    def initialize(self) -> Any:
        """Initialize provider client."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        cached_content: Optional[str] = None,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        """Generate JSON response."""
    
    @abstractmethod
    def create_cache(self, system_instruction: str, contents: List[Any]) -> Any:
        """Create cached context."""
    
    @property
    @abstractmethod
    def capability(self) -> ProviderCapability:
        """Provider capability level."""
    
    def get_client(self):
        """Lazy-initialize client."""
        if self._client is None:
            self._client = self.initialize()
        return self._client
    
    def _retry_generate(self, generate_fn, retries: int = 3) -> Dict[str, Any]:
        """Retry generation with exponential backoff."""
        retryable_keywords = [
            "ssl", "eof", "timeout", "connection",
            "temporarily", "unavailable", "deadline",
            "429", "rate", "resource exhausted",
        ]
        
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return generate_fn()
            except Exception as e:
                last_error = e
                err_lower = str(e).lower()
                if not any(kw in err_lower for kw in retryable_keywords):
                    raise
                if attempt == retries:
                    raise
                time.sleep(self.retry_wait * attempt)
        raise last_error


class PlaceholderProvider(BaseProvider):
    """Placeholder for unimplemented providers."""

    def initialize(self):
        raise NotImplementedError(f"{self.__class__.__name__} not implemented")

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def generate_json(self, *args, **kwargs):
        raise NotImplementedError

    def create_cache(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def capability(self) -> ProviderCapability:
        return ProviderCapability.NONE

