import asyncio
import time
from dataclasses import dataclass

from .backend import BackendError, SGLangClient
from .config import Settings
from .models import SystemOneRequest, SystemOneResponse, Usage
from .prompts import PromptCompiler
from .scoring import answer


class RequestError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass
class Evaluation:
    response: SystemOneResponse
    prefix_tokens: int
    cached_tokens: int | None
    prepare_ms: float
    prefill_ms: float
    branches_ms: float


class EvaluationService:
    def __init__(self, settings: Settings, compiler: PromptCompiler, backend: SGLangClient):
        self.settings = settings
        self.compiler = compiler
        self.backend = backend
        self.active = 0

    async def evaluate(self, request: SystemOneRequest) -> Evaluation:
        # Any "jev-*" name is accepted as an alias, so pinning a tuned version
        # (e.g. "jev-1.13.0") is a drop-in swap, not just "jev-latest".
        known = request.model in {self.settings.served_model_name, self.settings.model_alias}
        if not known and not request.model.startswith("jev-"):
            raise RequestError(f"Unknown model: {request.model}")
        # No await between checking and reserving capacity on this event loop.
        if self.active >= self.settings.max_concurrent_requests:
            raise RequestError("Server is at capacity; retry shortly", 529)
        self.active += 1
        try:
            return await asyncio.wait_for(
                self._evaluate(request), self.settings.request_timeout
            )
        except asyncio.TimeoutError as exc:
            raise BackendError("Evaluation deadline exceeded", 504) from exc
        finally:
            self.active -= 1

    async def _evaluate(self, request: SystemOneRequest) -> Evaluation:
        started = time.perf_counter()
        try:
            prepared = await asyncio.to_thread(self.compiler.prepare, request)
        except (ValueError, TypeError) as exc:
            raise RequestError(str(exc)) from exc
        limit = self.settings.max_input_tokens
        # Include the single output token in context admission, before any GPU work.
        if any(len(branch.input_ids) + 1 > limit for branch in prepared.branches):
            raise RequestError(f"A question branch exceeds the {limit}-token context limit")
        total_input = len(prepared.prefix_ids) + sum(
            len(branch.input_ids) for branch in prepared.branches
        )
        if total_input > self.settings.max_total_input_tokens:
            raise RequestError("Request exceeds the total input-token budget across branches")

        # Barrier: cache the common prefix completely before submitting any branch.
        # The warm-up token is discarded; it is never appended to the branches.
        prepared_at = time.perf_counter()
        warmup = await self.backend.generate(prepared.prefix_ids)
        warmed_at = time.perf_counter()
        tasks = [
            asyncio.create_task(self.backend.generate(branch.input_ids, branch.label_ids))
            for branch in prepared.branches
        ]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            # Also clean up siblings on disconnect, timeout, or a failed branch.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        answers = {
            branch.question_id: answer(branch, result.logprobs, self.settings.temperature)
            for branch, result in zip(prepared.branches, results, strict=True)
        }
        return Evaluation(
            response=SystemOneResponse(
                model=request.model,
                answers=answers,
                usage=Usage(
                    input_tokens=warmup.input_tokens + sum(r.input_tokens for r in results),
                    output_tokens=warmup.output_tokens + sum(r.output_tokens for r in results),
                ),
            ),
            prefix_tokens=len(prepared.prefix_ids),
            cached_tokens=(
                sum(r.cached_tokens for r in results)
                if all(r.cached_tokens is not None for r in results)
                else None
            ),
            prepare_ms=(prepared_at - started) * 1000,
            prefill_ms=(warmed_at - prepared_at) * 1000,
            branches_ms=(time.perf_counter() - warmed_at) * 1000,
        )
