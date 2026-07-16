"""Shared exception types."""


class ProviderError(RuntimeError):
    """An upstream provider (embeddings API or LLM) failed.

    Raised on network errors, timeouts and non-2xx responses from
    OpenAI-compatible backends. The API layer maps it to HTTP 502.
    """
