# LLM Gateway Patterns

An LLM gateway is a thin service that sits between applications and model
providers, exposing one stable, OpenAI-compatible API while hiding provider
differences behind it. It is the traffic-control layer of an AI platform.

## Why a gateway

Calling providers directly scatters API keys, retry logic and cost tracking
across every application. A gateway centralizes those cross-cutting concerns:
applications talk to `/v1/chat/completions` on the gateway, and the gateway
decides which upstream provider actually serves the request.

## Core responsibilities

- **Unified API surface.** The gateway speaks the OpenAI chat completions
  schema (`model`, `messages`, `temperature`, `stream`) so existing SDKs work
  unchanged. Provider-specific formats are adapted behind the scenes.
- **Authentication.** Clients authenticate to the gateway with virtual API
  keys; real provider credentials never leave the gateway. Keys can carry
  per-team budgets and model allowlists.
- **Rate limiting.** A token-bucket limiter per API key smooths bursts and
  protects upstream quotas. Buckets refill at a fixed rate; a request that
  finds the bucket empty receives HTTP 429 with a `Retry-After` hint.
- **Routing and fallback.** A model registry maps public model names to
  provider deployments. If the primary provider times out or returns 5xx,
  the gateway retries with exponential backoff and jitter, then fails over
  to the next provider in the route.
- **Streaming.** Token streaming uses Server-Sent Events (SSE): the gateway
  proxies `data:` chunks as they arrive and terminates the stream with
  `data: [DONE]`, so clients see the same wire format regardless of provider.

## Cost and usage tracking

Every response carries a `usage` block (`prompt_tokens`,
`completion_tokens`, `total_tokens`). The gateway multiplies token counts by
per-model prices to attribute spend to API keys and teams, which makes
per-feature cost dashboards and budget alerts possible.

## Observability

Structured request logs (model, latency, tokens, status, provider chosen)
plus metrics for p50/p95 latency and error rate per provider are the minimum.
Traces that span gateway -> provider make timeout tuning far less painful.

## Failure modes to design for

Provider outages, silent quality degradation, thundering-herd retries after
an incident, and clients that never handle 429. Circuit breakers per provider
and load-shedding at the gateway keep a bad hour from becoming a bad day.
