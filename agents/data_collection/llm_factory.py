from __future__ import annotations

from typing import Any


def build_llm(config: dict[str, Any] | None) -> Any | None:
    """Build an LLM client from config."""

    if not config:
        return None

    backend = config.get("backend")
    if not backend:
        return None

    if backend == "openai_compatible":
        return _build_openai_compatible(config)
    if backend == "google_genai":
        return _build_google_genai(config)

    raise ValueError(f"Unsupported LLM backend: {backend!r}.")


def _build_openai_compatible(config: dict[str, Any]) -> Any:
    """Build a ChatOpenAI client."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "openai_compatible backend requires 'langchain-openai'. "
            "Install it with: pip install langchain-openai"
        ) from exc

    extra_kwargs = dict(config.get("kwargs") or config.get("extra_kwargs") or {})
    kwargs = {
        "model": config.get("model"),
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
        "temperature": config.get("temperature"),
        "max_tokens": config.get("max_tokens"),
        "timeout": config.get("timeout"),
        **extra_kwargs,
    }
    return ChatOpenAI(**_drop_none(kwargs))


def _build_google_genai(config: dict[str, Any]) -> Any:
    """Build a ChatGoogleGenerativeAI client."""

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "google_genai backend requires 'langchain-google-genai'. "
            "Install it with: pip install langchain-google-genai"
        ) from exc

    extra_kwargs = dict(config.get("kwargs") or config.get("extra_kwargs") or {})
    client_options = dict(extra_kwargs.pop("client_options", {}))
    if config.get("base_url"):
        client_options.setdefault("api_endpoint", config["base_url"])

    kwargs = {
        "model": config.get("model"),
        "google_api_key": config.get("google_api_key") or config.get("api_key"),
        "temperature": config.get("temperature"),
        "max_output_tokens": config.get("max_tokens"),
        "timeout": config.get("timeout"),
        **extra_kwargs,
    }
    if client_options:
        kwargs["client_options"] = client_options

    return ChatGoogleGenerativeAI(**_drop_none(kwargs))


def _drop_none(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop None values from kwargs."""

    return {key: value for key, value in kwargs.items() if value is not None}
