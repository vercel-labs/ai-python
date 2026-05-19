"""Helpers for models.dev metadata."""

from __future__ import annotations

import re

import modelsdotdev

_ENV_REFERENCE_RE = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)\}?")
_SECRET_ENV_MARKERS = ("API_KEY", "TOKEN", "SECRET", "BEARER")
_PROVIDER_ID_ALIASES = {"ai-gateway": "vercel", "gateway": "vercel"}


def parse_model_id(model_id: str) -> modelsdotdev.ModelRef:
    return modelsdotdev.parse_model_id(_canonical_model_id(model_id))


def get_provider_by_id(provider_id: str) -> modelsdotdev.Provider | None:
    return modelsdotdev.get_provider_by_id(_canonical_provider_id(provider_id))


def get_model_by_id(model_id: str) -> modelsdotdev.Model | None:
    return modelsdotdev.get_model_by_id(_canonical_model_id(model_id))


def _canonical_provider_id(provider_id: str) -> str:
    return _PROVIDER_ID_ALIASES.get(provider_id, provider_id)


def _canonical_model_id(model_id: str) -> str:
    for separator in (":", "/"):
        prefix, sep, rest = model_id.partition(separator)
        if sep and prefix in _PROVIDER_ID_ALIASES:
            return f"{_PROVIDER_ID_ALIASES[prefix]}{separator}{rest}"
    return model_id


def provider_base_url(
    provider: modelsdotdev.Provider,
    model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
) -> str | None:
    if (
        model_provider_config is not None
        and model_provider_config.api is not None
    ):
        return model_provider_config.api
    return provider.api


def provider_config(
    provider: modelsdotdev.Provider,
    model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Return API key and config envs from models.dev data."""
    api = provider_base_url(provider, model_provider_config)
    envs = _provider_envs(provider, api)
    api_key_env = _api_key_env(envs, api)
    config_envs = tuple(env for env in envs if env != api_key_env)
    return api_key_env, config_envs


def provider_npm(
    provider: modelsdotdev.Provider,
    model_provider_config: modelsdotdev.ModelProviderConfig | None = None,
) -> str:
    if (
        model_provider_config is not None
        and model_provider_config.npm is not None
    ):
        return model_provider_config.npm
    return provider.npm


def _provider_envs(
    provider: modelsdotdev.Provider, api: str | None
) -> tuple[str, ...]:
    envs = list(provider.env)
    for env in _ENV_REFERENCE_RE.findall(api or ""):
        if env not in envs:
            envs.append(env)
    return tuple(envs)


def _api_key_env(envs: tuple[str, ...], api: str | None) -> str | None:
    if not envs:
        return None

    referenced_envs = set(_ENV_REFERENCE_RE.findall(api or ""))
    candidates = [env for env in envs if env not in referenced_envs]
    if not candidates:
        candidates = list(envs)

    for marker in _SECRET_ENV_MARKERS:
        for env in candidates:
            if marker in env:
                return env
    return candidates[0]
