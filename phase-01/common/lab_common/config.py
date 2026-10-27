"""Model provider configuration from environment variables.

Both agents use the same provider via an OpenAI-compatible endpoint, so only
MODEL_PROVIDER / MODEL_NAME / API_KEY (and optionally BASE_URL) matter.
Reuse decision: no custom provider abstraction beyond a base-URL lookup table;
LiteLLM was not added because all four target providers expose OpenAI-compatible APIs.
"""
import os
from dataclasses import dataclass

from .envfile import load_env

# Base URLs must be re-verified against provider docs when first used.
PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama-cloud": "https://ollama.com/v1",
    "openai": "https://api.openai.com/v1",
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    api_key: str
    base_url: str


class ConfigError(RuntimeError):
    pass


def load_model_config() -> ModelConfig:
    load_env()
    provider = os.environ.get("MODEL_PROVIDER", "").strip().lower()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    api_key = os.environ.get("API_KEY", "").strip()
    base_url = os.environ.get("BASE_URL", "").strip() or PROVIDER_BASE_URLS.get(provider, "")
    missing = [n for n, v in [("MODEL_PROVIDER", provider), ("MODEL_NAME", model_name),
                              ("API_KEY", api_key), ("BASE_URL (or known MODEL_PROVIDER)", base_url)] if not v]
    if missing:
        raise ConfigError(f"Missing env config: {', '.join(missing)}. See .env.<APP_ENV>.example")
    return ModelConfig(provider, model_name, api_key, base_url)


def model_configured() -> bool:
    try:
        load_model_config()
        return True
    except ConfigError:
        return False
