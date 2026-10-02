"""OpenAPI descriptions shared by Scalar, Swagger, and generated clients."""

TAGS = [
    {
        "name": "SystemOne",
        "description": "Evaluate typed questions against shared text or structured state.",
    },
    {"name": "Models", "description": "Discover the model names accepted by this deployment."},
    {
        "name": "Limits",
        "description": "Inspect request size, answer count, and concurrency limits.",
    },
    {"name": "Health", "description": "Readiness and liveness checks for monitoring."},
]

INTRODUCTION = (
    "Jev-compatible API endpoints, built on SGLang radix tree prefix reuse "
    "and parallel requests to open-weight models."
    "\n\n[Source code on GitHub](https://github.com/ekzhang/openjev-sglang)"
)

SYSTEMONE_DESCRIPTION = """Evaluate **1–64 independent questions** against one shared `state`.

### Choose an answer type

- **Noul** (`noul`): ask a yes/no question. The answer's `noul` value is the
  probability of `true`, from 0 to 1. Optionally define what `true` and `false`
  mean using `criteria`.
- **Choice** (`choice`): supply 2–64 named options in `criteria`. The response
  contains the winning key and a probability for every option; probabilities sum to 1.
  The model sees the description, or the option name only when its description is null.
- **Score** (`score`): supply 2–64 rubric descriptions in ascending order.
  The score is the probability-weighted **zero-based index**, so three levels
  yield a score between 0 and 2. `legend` maps each index to its description.

### State and instructions

`instructions` is optional per question; omitting it evaluates the options against
the shared `state` alone. Both fields accept text, a JSON object, or a JSON array.
For a conversation, pass a list of chat messages with `role` and `content`, or an
object containing only `messages`. Supported roles are `system`, `user`, `assistant`,
and `tool`. Other structured state is evaluated as JSON. Audio and video content are
not supported. Each question is evaluated independently, without seeing other
questions or their answers.

#### Images

A `user` message's `content` may be a list of parts, mixing text and images in the
OpenAI image shape:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Is the signature present and legible?"},
  {"type": "image_url", "image_url": {"url": "https://example.com/scan.png"}}
]}
```

Each part becomes exactly one position in the prompt, so `text` and image parts can be
interleaved to describe several pictures. A `url` may be an `http(s)://` URL or a
`data:image/...;base64,...` payload. Images belong to the request's state, are shared
by every question, and count toward the same token budget as text.

Image state requires a **vision checkpoint** (for example Qwen3-VL) and a positive
image allowance. A text-only deployment answers **422** saying the model cannot
evaluate images.

### Response and usage

`answers` preserves your question keys. Choice and Score include `confidence`,
defined as `1 − H(probabilities) / log(option_count)`: 0 is uniform and 1 is a
point mass. This measures concentration, not correctness.

`usage.input_tokens` counts all submitted prompt tokens, including cache hits
and the shared warm-up. `usage.output_tokens` is **number of questions + 1**:
one discarded warm-up token and one token per question. No thinking tokens or
multi-token answers are generated.

### Limits and retries

The defaults allow 2 MiB JSON, 32,768 tokens per branch including its output,
and 262,144 total submitted input tokens. Check **Limits** for this deployment.
Validation failures return 422; oversized bodies return 413. A busy server
returns 529 with `Retry-After`. Retry 503 or 529 after the suggested delay.
On a cold start, Modal can return 503 while the model loads.

Response headers include `x-typesafe-request-id`, `x-openjev-model`,
`x-openjev-prefix-tokens`, and `Server-Timing`. `x-openjev-cached-tokens` appears
only when the backend supplies cache counts; its absence does not mean zero hits.
"""

ERROR_DESCRIPTIONS = {
    401: "Missing or invalid Bearer token when optional API authentication is configured.",
    413: "The JSON request body exceeds the deployment's byte limit.",
    422: "Invalid fields, unknown model, too many questions/options, or token budget exceeded.",
    502: "The inference backend returned an error or an invalid logprob response.",
    503: "Backend unavailable, or Modal is starting a cold container.",
    504: "The evaluation exceeded its deadline.",
    529: "The API is at capacity. Wait for Retry-After before retrying.",
}
