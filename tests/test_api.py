import math
from dataclasses import replace

import httpx
import pytest

from openjev.api import create_app


async def test_full_request_prefills_then_branches(api, payload):
    client, calls, _ = api
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(calls) == 4
    prefix = calls[0]["input_ids"]
    for branch in calls[1:]:
        assert branch["input_ids"][: len(prefix)] == prefix
        assert branch["logprob_start_len"] == -1
        assert branch["sampling_params"]["top_k"] == -1
    assert data["usage"]["output_tokens"] == 4
    assert data["usage"]["input_tokens"] == sum(len(c["input_ids"]) for c in calls)
    assert data["answers"]["yes"]["noul"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert data["answers"]["team"]["choice"] == "billing"
    score = data["answers"]["level"]
    assert score["legend"] == {"0": "Low", "1": "Medium", "2": "High"}
    assert score["score"] == pytest.approx(
        sum(int(i) * p for i, p in score["probabilities"].items())
    )
    assert response.headers["x-openjev-cached-tokens"] == "300"
    assert "x-typesafe-request-id" in response.headers


async def test_public_model_name_in_catalogue_requests_and_headers(api, payload):
    client, _, _ = api
    catalogue = await client.get("/v1/models")
    assert "nvidia/" not in catalogue.text
    assert {m["name"] for m in catalogue.json()["models"]} == {"jev-latest", "Qwen/Qwen3.6-35B-A3B"}
    payload["model"] = "Qwen/Qwen3.6-35B-A3B"
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == payload["model"]
    assert response.headers["x-openjev-model"] == payload["model"]


@pytest.mark.parametrize("model", ["jev-1.13.0", "jev-anything"])
async def test_pinned_jev_model_names_are_accepted_as_aliases(api, payload, model):
    client, _, _ = api
    payload["model"] = model
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == model


@pytest.mark.parametrize("count,status", [(1, 422), (2, 200), (64, 200), (65, 422)])
async def test_answer_limits(api, count, status):
    client, calls, _ = api
    response = await client.post(
        "/v1/systemone",
        json={
            "model": "jev-latest",
            "state": "test",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {str(i): f"Option {i}" for i in range(count)},
                },
            },
        },
    )
    assert response.status_code == status
    if status == 422:
        assert not calls
    else:
        assert len(response.json()["answers"]["q"]["probabilities"]) == count


@pytest.mark.parametrize(
    "change",
    [
        {"state": None},
        {"model": "other"},
        {"model": 1},
        {"questions": {}},
        {"questions": {"q": {"type": "invalid", "instructions": "x"}}},
        {"questions": {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}},
        {"questions": {str(i): {"type": "noul", "instructions": "x"} for i in range(65)}},
    ],
)
async def test_invalid_requests_do_not_hit_gpu(api, payload, change):
    client, calls, _ = api
    response = await client.post("/v1/systemone", json=payload | change)
    assert response.status_code == 422
    assert not calls


@pytest.mark.parametrize("description", [None, ""])
async def test_choice_accepts_nullable_descriptions(api, payload, description):
    client, calls, _ = api
    payload["questions"]["team"]["criteria"]["technical"] = description
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    assert response.json()["answers"]["team"]["probabilities"].keys() == {"billing", "technical"}


async def test_structured_criteria_are_accepted_and_score_legend_is_preserved(api, payload):
    client, calls, _ = api
    score_criteria = [{"level": "Low"}, ["High", {"threshold": 10}]]
    payload["questions"] = {
        "binary": {
            "type": "noul",
            "criteria": {"true": {"meaning": "Match"}, "false": ["Different"]},
        },
        "choice": {
            "type": "choice",
            "criteria": {"object": {"meaning": "Object"}, "array": ["Array"]},
        },
        "score": {"type": "score", "criteria": score_criteria},
    }

    response = await client.post("/v1/systemone", json=payload)

    assert response.status_code == 200, response.text
    assert len(calls) == 4
    assert response.json()["answers"]["score"]["legend"] == {
        "0": score_criteria[0],
        "1": score_criteria[1],
    }


async def test_question_without_instructions_is_accepted(api, payload):
    client, calls, _ = api
    del payload["questions"]["yes"]["instructions"]
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    assert "yes" in response.json()["answers"]


async def test_token_limit_checked_before_prefill(api, payload):
    client, calls, service = api
    service.settings.max_input_tokens = 256
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    assert not calls


async def test_total_token_limit_checked_before_prefill(api, payload):
    client, calls, service = api
    service.settings.max_total_input_tokens = 1
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    assert not calls


async def test_overload_returns_retryable_error(api, payload):
    client, calls, service = api
    service.active = service.settings.max_concurrent_requests
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 529
    assert response.headers["retry-after"] == "1"
    assert not calls


async def test_auth_and_body_limits(api, payload):
    _, calls, service = api
    settings = service.settings.model_copy(update={"max_body_bytes": 50})
    from pydantic import SecretStr

    settings.api_key = SecretStr("test-key")
    app = create_app(settings, service=service)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.ASGITransport(app=app)
        ) as client:
            assert (await client.post("/v1/systemone", json=payload)).status_code == 401
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/docs")).status_code == 401
            response = await client.post(
                "/v1/systemone", json=payload, headers={"Authorization": "Bearer test-key"}
            )
            assert response.status_code == 413
    assert not calls


async def test_models_and_limits(api):
    client, _, _ = api
    data = (await client.get("/v1/models")).json()
    assert data["models"][0]["name"] == "jev-latest"
    assert data["data"][0]["id"] == "jev-latest"
    limits = (await client.get("/v1/limits")).json()
    assert limits["max_answers_per_question"] == 64


async def test_root_scalar_ui_and_example(api):
    client, calls, _ = api
    page = await client.get("/")
    assert page.status_code == 200
    assert "Scalar.createApiReference" in page.text
    assert "/openapi.json" in page.text
    assert "SwaggerUIBundle" in (await client.get("/docs")).text
    schema = (await client.get("/openapi.json")).json()
    example = schema["components"]["schemas"]["SystemOneRequest"]["example"]
    result = await client.post("/v1/systemone", json=example)
    assert result.status_code == 200
    assert len(result.json()["answers"]) == 3


async def test_missing_backend_cache_counts_are_not_reported_as_zero(api, payload, monkeypatch):
    client, _, service = api
    generate = service.backend.generate

    async def without_cache_counts(*args):
        return replace(await generate(*args), cached_tokens=None)

    monkeypatch.setattr(service.backend, "generate", without_cache_counts)
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    assert "x-openjev-cached-tokens" not in response.headers
