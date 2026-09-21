import asyncio
import secrets
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import orjson
from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .backend import BackendError, SGLangClient
from .config import MAX_ANSWERS, MAX_QUESTIONS, Settings
from .documentation import ERROR_DESCRIPTIONS, INTRODUCTION, SYSTEMONE_DESCRIPTION, TAGS
from .models import (
    ErrorResponse,
    HealthResponse,
    LimitsResponse,
    LiveResponse,
    ModelsResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from .prompts import PromptCompiler
from .service import EvaluationService, RequestError


class ORJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return orjson.dumps(content)


class RequestGuard:
    """Authenticate and bound even chunked request bodies before JSON parsing."""

    def __init__(self, app: ASGIApp, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        key = self.settings.api_key
        if key and scope["path"] not in {"/health", "/health/live"}:
            expected = f"Bearer {key.get_secret_value()}".encode()
            if not secrets.compare_digest(headers.get(b"authorization", b""), expected):
                return await ORJSONResponse(
                    {"error": {"message": "Missing or invalid API key"}},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )(scope, receive, send)
        if scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > self.settings.max_body_bytes:
                return await ORJSONResponse(
                    {"error": {"message": "Request body is too large"}},
                    status_code=413,
                )(scope, receive, send)
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(
    settings: Settings | None = None,
    *,
    service: EvaluationService | None = None,
    launch_backend: bool = False,
    sglang_args: list[str] | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.monotonic()
        if service is not None:
            app.state.service = service
            yield
            return
        from .runtime import backend_process, load_tokenizer, wait_ready

        headers = {}
        if settings.backend_api_key:
            headers["Authorization"] = f"Bearer {settings.backend_api_key.get_secret_value()}"
        async with backend_process(settings, launch_backend, sglang_args or []) as process:
            async with httpx.AsyncClient(
                base_url=settings.backend_url,
                headers=headers,
                timeout=httpx.Timeout(settings.request_timeout, connect=5),
                limits=httpx.Limits(
                    max_connections=settings.max_concurrent_branches + 8,
                    max_keepalive_connections=settings.max_concurrent_branches + 8,
                ),
            ) as client:
                backend = SGLangClient(settings, client)
                await wait_ready(backend, process, settings.startup_timeout)
                compiler = PromptCompiler(
                    await asyncio.to_thread(load_tokenizer, settings),
                    max_images=settings.max_images,
                )
                app.state.service = EvaluationService(settings, compiler, backend)
                app.state.startup_seconds = round(time.monotonic() - started, 2)
                yield

    app = FastAPI(
        title="OpenJev",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        default_response_class=ORJSONResponse,
        description=INTRODUCTION,
        openapi_tags=TAGS,
    )
    app.add_middleware(RequestGuard, settings=settings)

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def reference():
        return HTMLResponse("""<!doctype html>
<html lang="en">
  <head>
    <title>OpenJev API Reference</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
  </head>
  <body>
    <div id="app"></div>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
    <script>
      Scalar.createApiReference('#app', { url: '/openapi.json' });
    </script>
  </body>
</html>""")

    @app.exception_handler(RequestError)
    @app.exception_handler(BackendError)
    async def error_handler(request: Request, exc: RequestError | BackendError):
        return ORJSONResponse(
            {"error": {"message": str(exc)}},
            status_code=exc.status,
            headers={"Retry-After": "1"} if exc.status in {429, 503, 529} else {},
        )

    @app.post(
        "/v1/systemone",
        tags=["SystemOne"],
        summary="Evaluate questions",
        description=SYSTEMONE_DESCRIPTION,
        response_model=SystemOneResponse,
        responses={
            code: {
                "description": description,
                **(
                    {"model": ErrorResponse}
                    if code != 422
                    else {
                        "content": {
                            "application/json": {
                                "examples": {
                                    "validation": {
                                        "summary": "Invalid request field",
                                        "value": {
                                            "detail": [
                                                {
                                                    "type": "missing",
                                                    "loc": ["body", "state"],
                                                    "msg": "Field required",
                                                }
                                            ]
                                        },
                                    },
                                    "admission": {
                                        "summary": "Unknown model or token limit",
                                        "value": {"error": {"message": "Unknown model: example"}},
                                    },
                                }
                            }
                        }
                    }
                ),
            }
            for code, description in ERROR_DESCRIPTIONS.items()
        },
    )
    async def systemone(payload: SystemOneRequest, request: Request):
        async def disconnect():
            while True:
                event = await request.receive()
                if event["type"] == "http.disconnect":
                    return

        evaluation = asyncio.create_task(request.app.state.service.evaluate(payload))
        watcher = asyncio.create_task(disconnect())
        try:
            await asyncio.wait([evaluation, watcher], return_when=asyncio.FIRST_COMPLETED)
            if not evaluation.done():
                raise RequestError("Client disconnected", 499)
            result = await evaluation
            return ORJSONResponse(
                result.response.model_dump(),
                headers={
                    "x-typesafe-request-id": uuid4().hex,
                    "x-openjev-prefix-tokens": str(result.prefix_tokens),
                    **(
                        {"x-openjev-cached-tokens": str(result.cached_tokens)}
                        if result.cached_tokens is not None
                        else {}
                    ),
                    "x-openjev-model": settings.served_model_name,
                    "Server-Timing": (
                        f"prepare;dur={result.prepare_ms:.2f}, "
                        f"prefill;dur={result.prefill_ms:.2f}, "
                        f"branches;dur={result.branches_ms:.2f}"
                    ),
                },
            )
        finally:
            for task in (evaluation, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(evaluation, watcher, return_exceptions=True)

    @app.get(
        "/v1/models",
        tags=["Models"],
        summary="List available model names",
        description=(
            "Lists the names accepted by `POST /v1/systemone` on this deployment. "
            "`jev-latest` and the checkpoint ID refer to the same loaded model. "
            "`models` uses the TypeSafe catalogue format; `data` exposes the same IDs "
            "in OpenAI format. Model selection happens at deployment time, not by "
            "loading a new model for each request."
        ),
        response_model=ModelsResponse,
    )
    async def models():
        names = list(dict.fromkeys([settings.model_alias, settings.served_model_name]))
        # Both SDK conventions in one additive response.
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "owned_by": "openjev"} for name in names],
            "models": [
                {
                    "name": name,
                    "description": f"OpenJev classification using {settings.served_model_name}",
                    "release_date": "2026-09-17",
                }
                for name in names
            ],
        }

    @app.get(
        "/v1/limits",
        tags=["Limits"],
        summary="Get request and concurrency limits",
        description=(
            "Returns the effective admission limits for this deployment. Choice and "
            "Score require at least two options. Body sizes are bytes; token budgets "
            "use the configured model's tokenizer. Concurrency is per container, and "
            "Modal can scale to additional containers."
        ),
        response_model=LimitsResponse,
    )
    async def limits():
        return {
            "max_answers_per_question": MAX_ANSWERS,
            "max_questions": MAX_QUESTIONS,
            "max_body_bytes": settings.max_body_bytes,
            "max_input_tokens": settings.max_input_tokens,
            "max_total_input_tokens": settings.max_total_input_tokens,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "max_images": settings.max_images,
        }

    @app.get(
        "/health",
        tags=["Health"],
        summary="Check inference readiness",
        description=(
            "Checks the connection to SGLang. Returns 200 when healthy or 503 when "
            "unavailable. `startup_seconds` records model and API initialization time "
            "for this container, excluding image build and scheduling. Modal may "
            "return its own 503 response before the API starts."
        ),
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "Inference backend unavailable."}},
    )
    async def health(request: Request):
        healthy = await request.app.state.service.backend.health()
        return ORJSONResponse(
            {
                "status": "ok" if healthy else "unavailable",
                "startup_seconds": getattr(request.app.state, "startup_seconds", None),
            },
            status_code=200 if healthy else 503,
        )

    @app.get(
        "/health/live",
        tags=["Health"],
        summary="Check API liveness",
        description=(
            "Returns 200 while the API process is serving requests. This does not "
            "test the inference backend; use `/health` for readiness."
        ),
        response_model=LiveResponse,
    )
    async def live():
        return {"status": "ok"}

    return app
