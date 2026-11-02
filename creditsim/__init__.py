"""Synthetic credit data generator and probability-of-default models."""

from .generator import GeneratorConfig, generate
from .validate import validate

__all__ = ["GeneratorConfig", "generate", "validate"]
__version__ = "0.1.0"
