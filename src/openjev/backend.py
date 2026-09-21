import asyncio
import math
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import orjson

from .config import Settings


class BackendError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Generation:
    logprobs: list[float]
    input_tokens: int
    output_tokens: int
    cached_tokens: int | None


class SGLangClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.client = client
        self.slots = asyncio.Semaphore(settings.max_concurrent_branches)

    async def health(self) -> bool:
        try:
            response = await self.client.get("/health", timeout=5)
            return response.is_success
        except httpx.HTTPError:
            return False

    async def generate(
        self,
        input_ids: list[int],
        label_ids: list[int] | None = None,
        image_data: list[str] | None = None,
    ):
        rid = f"openjev-{uuid4().hex}"
        payload: dict[str, Any] = {
            "rid": rid,
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": True,
            },
            "stream": False,
            # SGLang 0.5.19 crashes when selected-logprob and plain requests
            # share a batch (sgl-project/sglang#34719). Warmups request one
            # unused token's probability so every request takes the same path.
            "return_logprob": True,
            "token_ids_logprob": label_ids or [0],
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
        }
        if image_data:
            # SGLang accepts image_data alongside input_ids: it decodes the ids back to
            # text for the HF processor, expands the single <|image_pad|> placeholder per
            # image, then restores the caller's original tokens
            # (SGLANG_MM_AVOID_RETOKENIZE, default on). Exactly one placeholder per image
            # is required, so PromptCompiler verifies the render before we get here.
            payload["image_data"] = image_data
        async with self.slots:
            try:
                response = await self.client.post(
                    "/generate",
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                )
            except asyncio.CancelledError:
                await self._abort(rid)
                raise
            except httpx.TimeoutException as exc:
                await self._abort(rid)
                raise BackendError("SGLang request timed out", 504) from exc
            except httpx.HTTPError as exc:
                raise BackendError("Cannot reach SGLang", 503) from exc
        if not response.is_success:
            status = response.status_code
            if status not in {429, 503, 529}:
                status = 502
            raise BackendError(f"SGLang returned HTTP {response.status_code}", status)
        try:
            return parse_generation(response.json(), label_ids or [])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise BackendError(f"Invalid SGLang logprob response: {exc}") from exc

    async def _abort(self, rid: str):
        try:
            await self.client.post("/abort_request", json={"rid": rid}, timeout=2)
        except httpx.HTTPError:
            pass


def parse_generation(data: dict, label_ids: list[int]) -> Generation:
    meta = data["meta_info"]
    reason = meta.get("finish_reason") or {}
    if reason.get("type") == "abort":
        raise ValueError("generation was aborted")
    output_tokens = meta["completion_tokens"]
    if output_tokens != 1:
        raise ValueError(f"expected exactly one output token, got {output_tokens}")
    logprobs = []
    if label_ids:
        positions = meta["output_token_ids_logprobs"]
        if len(positions) != 1:
            raise ValueError("expected one position of selected-token logprobs")
        by_id = {}
        for entry in positions[0]:
            value, token_id = entry[:2]
            if token_id in by_id:
                raise ValueError("duplicate token ID in logprobs")
            if value is None or math.isnan(value) or value == math.inf:
                raise ValueError("non-numeric or invalid logprob")
            by_id[token_id] = float(value)
        logprobs = [by_id[token_id] for token_id in label_ids]
        if not any(math.isfinite(value) for value in logprobs):
            raise ValueError("all label probabilities are zero")
    return Generation(
        logprobs=logprobs,
        input_tokens=int(meta["prompt_tokens"]),
        output_tokens=int(output_tokens),
        # The Rust frontend currently omits this field even when radix hits occur.
        cached_tokens=int(meta["cached_tokens"]) if meta.get("cached_tokens") is not None else None,
    )
