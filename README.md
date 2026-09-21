# openjev-sglang

A server implementing the [TypeSafe/Jev HTTP API](https://docs.typesafe.ai/api)
with **Qwen3.6-35B-A3B on SGLang**.

![](https://i.imgur.com/wHM3jxV.gif)

Each container one B200 with SGLang **0.5.19's Rust frontend**,
radix caching, and **breakable prefill CUDA graphs**. A separate Python API process
uses FastAPI, uvloop, the Rust-backed HF tokenizer, and pooled asynchronous HTTP
connections to SGLang on localhost. CUDA dependencies stay in SGLang's container;
`uv sync` on your laptop installs only the API, deployment tools, and tests.

## Run on Modal

```sh
uv sync
# Only if you haven't authenticated Modal on this machine:
uv run modal setup

# Start a temporary Server, run actual inference checks, then shut it down:
uv run modal run modal_app.py

# Deploy a stable public endpoint:
uv run modal deploy modal_app.py
```

The deployment prints a `https://...us-west.modal.direct` URL. It uses a
[Modal Server](https://modal.com/docs/guide/servers), `unauthenticated=True`,
`routing_region="us-west"`, and `compute_region=["us-west", "us-central", "us"]`.
Autoscaling has no explicit container cap and scales to zero after five idle minutes.
Set `min_containers=1` in `modal_app.py` to keep a B200 warm.
If SGLang exits unexpectedly, the API exits too. The Modal launcher watches the
API and exits the container so Modal can replace it, rather than leaving a live
HTTP process with a dead inference backend. Normal shutdown disarms both watchers.

Cache warmups also request one unused token probability to avoid SGLang's
[mixed-logprob batch crash](https://github.com/sgl-project/sglang/issues/34719).
This keeps warmups and scoring requests batch-compatible without patching SGLang.

The first build imports a large SGLang image. The first GPU start also downloads
weights and compiles/captures kernels. Model weights persist in the
`openjev-huggingface` Modal Volume, alongside SGLang's tuning cache and Triton
compilation cache. Later starts reuse these files; CUDA graph capture still runs
at startup. The Rust frontend receives an explicit local tokenizer directory to
avoid remote-name lookup issues with revision-pinned snapshots.
A scaled-to-zero Server returns **503** while it
starts; the included smoke command retries startup responses.

```sh
uv run openjev smoke https://YOUR-SERVER.us-west.modal.direct
```

The smoke test covers all three answer types, a 64-answer question, basic semantic
sanity checks, and rejection of 65 answers. It reports startup wait, inference
latency, and cache usage. `modal run` saves this report as `smoke-result.json`.

## Request

```sh
curl "$OPENJEV_URL/v1/systemone" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": [
      {"role": "system", "content": "You are a support assistant."},
      {"role": "user", "content": "I was charged twice. Please refund the duplicate."}
    ],
    "questions": {
      "refund": {
        "type": "noul",
        "instructions": "Does the user request a refund?"
      },
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["Routine", "Urgent", "Emergency"]
      }
    }
  }'
```

Or use `curl "$OPENJEV_URL/v1/systemone" -H 'Content-Type: application/json' --data-binary @examples/request.json`.

`state` and `instructions` accept strings, JSON objects, or arrays. A state that is
a list of chat messages, or exactly `{"messages": [...]}`, is rendered using the
model's native chat template. Original roles and message objects are retained;
the classification question becomes an additional user turn, even after another
user turn. Other structured state is serialized intact into a user message.
Objects containing `messages` plus additional fields are kept intact so metadata
isn't silently discarded. Chat state supports text and `image_url` parts, not
audio/video content.

A `user` message's `content` may also be a list of parts, mixing text and images.
This follows the OpenAI image shape, which is what Qwen's chat template matches on,
so each part becomes exactly one position in the prompt and can be used to describe
several pictures:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Is the signature present and legible?"},
  {"type": "image_url", "image_url": {"url": "https://example.com/scan.png"}}
]}
```

`url` accepts an `http(s)://` URL or a `data:image/...;base64,...` payload. Images
belong to the request's state and are shared by every question, so they are sent on
the cache-warming prefill and on each branch. Image state needs a vision checkpoint
(for example Qwen3-VL) and a nonzero `OPENJEV_MAX_IMAGES`; a text-only deployment
returns **422** explaining that the model cannot evaluate images. Because images
expand to many tokens, expect a heavier shared prefix and reduce
`OPENJEV_MAX_TOTAL_INPUT_TOKENS` when 64 branches each carry pictures.

The image path reuses SGLang's `/generate` unchanged: it accepts `image_data`
alongside `input_ids`, decodes the ids back to text for its multimodal processor,
expands the one `<|image_pad|>` placeholder per image, and then restores the
caller's original tokens (`SGLANG_MM_AVOID_RETOKENIZE`, on by default).

| Route                | Purpose                                                          |
| -------------------- | ---------------------------------------------------------------- |
| `POST /v1/systemone` | Noul, Choice, and Score evaluation                               |
| `GET /v1/models`     | Model catalogue with TypeSafe and OpenAI-style fields            |
| `GET /v1/limits`     | Admission limits                                                 |
| `GET /health`        | Readiness, including SGLang health and startup duration          |
| `GET /health/live`   | API process liveness                                             |
| `GET /`              | Scalar API reference with an editable example and request client |
| `GET /docs`          | Built-in Swagger UI                                              |
| `GET /openapi.json`  | Generated API schema                                             |

`jev-latest` is a compatibility alias for the configured Qwen model. The public
model ID is `Qwen/Qwen3.6-35B-A3B`;
NVIDIA's repository is only the internal weight source. Set
`OPENJEV_SERVED_MODEL_NAME` or `openjev serve --served-model-name NAME` to override
the public ID. It is also passed to SGLang as `--served-model-name`.
No requests go to TypeSafe. The public server is
the evaluation API; SGLang's generation and administration routes remain on
localhost and aren't forwarded publicly.

## How inference works

1. Validate the schema, answer count, body size, context length, and total token budget.
2. Render the native chat template **once**, with thinking disabled. Split out a
   common prefix and independently tokenize each question suffix.
3. Send the common prefix to `/generate` with `max_new_tokens=1`, await completion,
   and discard the sampled token. This warms SGLang's radix cache.
4. Concurrently send `prefix + question suffix + assistant header + "Answer:\n"`
   for each question. Every call again has **`max_new_tokens=1`**. Request
   `token_ids_logprob` for every answer label and `logprob_start_len=-1`, so there
   is no need to recompute prompt logprobs. The sampled token itself is ignored.
5. Renormalize the requested label logprobs with stable softmax. Noul returns
   `P(yes)`, Choice returns the argmax and full distribution, and Score returns
   `sum(level_index * probability)` with zero-based levels and a legend.

Options are rendered as `A: description`, `B: description`, etc., without JSON
wrappers. Choice keys identify response fields and are hidden from the model,
except when a description is `null`: then the option key supplies its meaning,
matching Jev's nullable description schema.

This is a prefill plus first-token-readout workload: there is no generated chain
of thought and no autoregressive continuation after the first token. There are
**N+1 one-token calls for N questions**, including the cache-warming call.
Speculative decoding is not enabled.

Qwen tokenizes `10` and `64` as multiple tokens. Answer labels are therefore
`A`–`Z`, followed by verified single-token letter combinations (`AA`, `AB`, ...).
All 64 labels are checked against the actual tokenizer at startup. Your original
option names are preserved in the returned distribution. This keeps 64-way
classification an exact one-token readout instead of comparing only the first
digit of a multi-token number.

Radix reuse is opportunistic, not a pinned per-request KV session. Hybrid Qwen's
recurrent state, cache page boundaries, cache pressure, and concurrent requests
can reduce hits. The backend uses `--mamba-radix-cache-strategy extra_buffer`.
`x-openjev-prefix-tokens` exposes the requested common prefix size. When SGLang
reports cache counts, `x-openjev-cached-tokens` sums the branch cache hits. SGLang
0.5.19's Rust frontend omits these counts: the header is absent and smoke reports
`null`, rather than a misleading zero. Scheduler logs still show actual cache
hits (verified on the live B200 deployment). `Server-Timing` separates
prompt preparation, the shared prefill, and branch inference.

`usage.input_tokens` sums SGLang's full prompt counts across the warm-up and all
branches, including cached tokens. `usage.output_tokens` is N+1. These are backend
usage counts, not TypeSafe billing estimates or unique tokens actually computed.

## Limits and configuration

Defaults are **64 questions**, **2–64 answers per Choice/Score**, **2 MiB JSON**,
**32,768 tokens per branch including its output**, **262,144 total submitted input
tokens**, **16 simultaneous evaluations**, and **64 simultaneous backend calls**.
Invalid requests return 422 before inference; oversized bodies return 413;
overload returns 529 with `Retry-After`. Backend timeouts return 504. Failed or
cancelled evaluations cancel sibling requests and attempt to abort them in SGLang.

All settings can be provided as `OPENJEV_*` environment variables; see
`src/openjev/config.py`. Common settings:

| Variable                          | Default                                                 |
| --------------------------------- | ------------------------------------------------------- |
| `OPENJEV_MODEL`                   | `nvidia/Qwen3.6-35B-A3B-NVFP4`                          |
| `OPENJEV_SERVED_MODEL_NAME`       | `Qwen/Qwen3.6-35B-A3B` (profile-specific public name)   |
| `OPENJEV_REVISION`                | Pinned NVIDIA checkpoint revision for the default model |
| `OPENJEV_FRONTEND`                | `rust` (`python` is an explicit fallback)               |
| `OPENJEV_MAX_INPUT_TOKENS`        | `32768`                                                 |
| `OPENJEV_MAX_TOTAL_INPUT_TOKENS`  | `262144`                                                |
| `OPENJEV_MAX_CONCURRENT_REQUESTS` | `16`                                                    |
| `OPENJEV_MAX_CONCURRENT_BRANCHES` | `64`                                                    |
| `OPENJEV_MAX_IMAGES`              | `0` (text only); images need a vision model            |
| `OPENJEV_REQUEST_TIMEOUT`         | `120` seconds                                           |
| `OPENJEV_TEMPERATURE`             | `1.0`, applied during label normalization               |
| `OPENJEV_API_KEY`                 | Unset; optional Bearer authentication for the API       |
| `OPENJEV_BACKEND_API_KEY`         | Unset; optional separate SGLang Bearer key              |

The Modal launch script forwards `OPENJEV_PROFILE`, `OPENJEV_FRONTEND`, and
`OPENJEV_SERVED_MODEL_NAME` from the local environment. To customize other remote settings,
add them to `image.env(...)` or
use a Modal Secret for keys. The default Modal endpoint intentionally has no auth.

OpenJev defines
`confidence = 1 - H(probabilities) / log(number_of_options)`, clamped to [0, 1].
This is zero for a uniform distribution and one for a point mass. Probabilities
are conditioned on the supplied options, depend on prompt and label ordering,
and are not calibrated estimates of correctness.

## Local development / existing SGLang

```sh
uv sync
uv run pytest                         # offline unit + API tests
uv run pytest -m integration          # real tokenizer, small HF download, no GPU
uv run ruff check .
uv run openjev schema                 # no GPU or model download

# Connect to an existing backend; it must have matching model/tokenizer revision,
# selected-token logprobs, radix cache, and a sufficient context length:
uv run openjev serve --connect http://127.0.0.1:30000

# On a B200 host/container with SGLang 0.5.19 installed in another environment:
uv run openjev serve --sglang-python /path/to/sglang/bin/python
```
