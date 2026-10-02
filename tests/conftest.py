import asyncio
import json

import httpx
import pytest

from openjev.api import create_app
from openjev.backend import SGLangClient
from openjev.config import Settings
from openjev.prompts import PromptCompiler
from openjev.service import EvaluationService

VISION_TOKENS = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
_PIECES = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>")
_PIECE_IDS = {piece: 50000 + index for index, piece in enumerate(_PIECES)}
_CHAR_BASE = 100000


def _encode_text(text, vision):
    """Encode by known multi-character pieces, then one id per remaining character.

    With ``vision`` off, the vision pieces are not recognised, mirroring a text-only
    checkpoint whose tokenizer has no <|image_pad|> token at all.
    """
    pieces = _PIECES if vision else _PIECES[3:]
    ids = []
    index = 0
    while index < len(text):
        for piece in pieces:
            if text.startswith(piece, index):
                ids.append(_PIECE_IDS[piece])
                index += len(piece)
                break
        else:
            ids.append(_CHAR_BASE + ord(text[index]))
            index += 1
    return ids


def _decode_ids(ids, vision):
    lookup = (
        {value: key for key, value in _PIECE_IDS.items()}
        if vision
        else {value: key for key, value in _PIECE_IDS.items() if key not in VISION_TOKENS}
    )
    return "".join(lookup.get(i, chr(i - _CHAR_BASE) if i >= _CHAR_BASE else "?") for i in ids)


class FakeTokenizer:
    def __init__(self, vision=False):
        self.vision = vision

    def encode(self, text, add_special_tokens=False):
        if text.isalpha() and text.isupper() and len(text) <= 2:
            if len(text) == 1:
                return [10000 + ord(text)]
            return [20000 + ord(text[0]) * 100 + ord(text[1])]
        return _encode_text(text, self.vision)

    def decode(self, ids):
        if len(ids) == 1 and ids[0] >= 20000:
            return chr((ids[0] - 20000) // 100) + chr((ids[0] - 20000) % 100)
        if len(ids) == 1 and ids[0] >= 10000:
            return chr(ids[0] - 10000)
        return _decode_ids(ids, self.vision)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        rendered = []
        for message in messages:
            rendered.append(f"<|im_start|>{message['role']}\n")
            content = message.get("content")
            if isinstance(content, list):
                # Mirrors Qwen's template: an image part becomes exactly one
                # <|vision_start|><|image_pad|><|vision_end|> run.
                for part in content:
                    if part.get("type") == "image_url":
                        rendered.append("<|vision_start|><|image_pad|><|vision_end|>")
                    elif part.get("type") == "text":
                        rendered.append(part["text"])
                    else:
                        raise ValueError(f"unexpected content part: {part}")
            else:
                rendered.append(content or "")
            rendered.append("<|im_end|>\n")
        rendered.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(rendered)


@pytest.fixture
def compiler():
    return PromptCompiler(FakeTokenizer())


@pytest.fixture
def vision_compiler():
    return PromptCompiler(FakeTokenizer(vision=True), max_images=4)


@pytest.fixture
def payload():
    return {
        "model": "jev-latest",
        "state": "I was charged twice, please refund.",
        "questions": {
            "yes": {"type": "noul", "instructions": "Refund?"},
            "team": {
                "type": "choice",
                "instructions": "Team?",
                "criteria": {"billing": "Payments", "technical": "Software"},
            },
            "level": {
                "type": "score",
                "instructions": "Urgency?",
                "criteria": ["Low", "Medium", "High"],
            },
        },
    }


@pytest.fixture
async def api(compiler):
    calls = []
    warmed = False
    settings = Settings()

    async def handler(request):
        nonlocal warmed
        if request.url.path == "/health":
            return httpx.Response(200, json={})
        data = json.loads(request.content)
        calls.append(data)
        assert data["sampling_params"]["max_new_tokens"] == 1
        labels = data.get("token_ids_logprob")
        is_warmup = labels == [0]
        if not is_warmup:
            assert warmed, "A branch ran before the prefix barrier"
        else:
            await asyncio.sleep(0.005)
            warmed = True
        meta = {
            "prompt_tokens": len(data["input_ids"]),
            "completion_tokens": 1,
            "cached_tokens": 0 if is_warmup else 100,
        }
        if labels:
            # Return reordered entries, so correctness depends on IDs, not tuple order.
            meta["output_token_ids_logprobs"] = [
                [[-float(i + 1), label, None] for i, label in enumerate(labels)][::-1]
            ]
        return httpx.Response(200, json={"text": "unused", "meta_info": meta})

    async with httpx.AsyncClient(
        base_url="http://sglang", transport=httpx.MockTransport(handler)
    ) as client:
        service = EvaluationService(settings, compiler, SGLangClient(settings, client))
        app = create_app(settings, service=service)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                base_url="http://test", transport=httpx.ASGITransport(app=app)
            ) as api_client:
                yield api_client, calls, service
