# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1", "pyarrow>=20,<24"]
# ///
"""Collect resumable BoolQ probabilities: uv run evals/boolq.py --help."""

import argparse
import asyncio
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from run_files import load_predictions, save_manifest

REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
DATA_URL = (
    f"https://huggingface.co/datasets/google/boolq/resolve/{REVISION}/"
    "data/validation-00000-of-00001.parquet"
)
DEFAULT_URL = "https://ekzhang--openjev-sglang-openjev.us-west.modal.direct"
MODEL = "Qwen/Qwen3.6-35B-A3B"
PROMPT = "Based on the passage, answer this yes/no question:\n{question}"


def payload(row, model):
    return {
        "model": model,
        "state": row["passage"],
        "questions": {
            "answer": {
                "type": "noul",
                "instructions": PROMPT.format(**row),
                "criteria": {"true": "Yes", "false": "No"},
            }
        },
    }


async def collect(args):
    is_jev = args.provider == "jev"
    model = "~typesafe/jev-latest" if is_jev else MODEL
    endpoint = "https://openrouter.ai" if is_jev else args.url.rstrip("/")
    route = "/api/alpha/decisions" if is_jev else "/v1/systemone"
    args.output.mkdir(parents=True, exist_ok=True)
    data_path = Path(__file__).parent / "data/boolq-validation.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as downloader:
        if not data_path.exists():
            response = await downloader.get(DATA_URL)
            response.raise_for_status()
            data_path.write_bytes(response.content)
    rows = pq.read_table(data_path).to_pylist()
    assert len(rows) == 3270 and all(type(row["answer"]) is bool for row in rows)
    if args.limit:
        rows = rows[: args.limit]
    manifest = {
        "dataset": "google/boolq",
        "split": "validation",
        "revision": REVISION,
        "data_url": DATA_URL,
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "endpoint": endpoint,
        "route": route,
        "model": model,
        "count": len(rows),
        "instructions": PROMPT,
        "state": "{passage}",
        "criteria": {"true": "Yes", "false": "No"},
        "concurrency": args.concurrency,
        "delay_per_worker_seconds": args.delay,
    }
    save_manifest(
        args.output / "manifest.json",
        manifest,
        mutable=("concurrency", "delay_per_worker_seconds"),
    )
    predictions = args.output / "predictions.jsonl"
    done = {r["index"]: r for r in load_predictions(predictions, range(len(rows)))}
    queue = asyncio.Queue()
    for i, row in enumerate(rows):
        if i not in done:
            queue.put_nowait((i, row))
    print(f"Dataset: {len(rows)} rows; resuming with {len(done)} completed", flush=True)
    with (args.output / "execution.jsonl").open("a") as events:
        events.write(
            json.dumps(
                {
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "concurrency": args.concurrency,
                    "delay_per_worker_seconds": args.delay,
                    "resumed_rows": len(done),
                }
            )
            + "\n"
        )
    started = time.monotonic()
    headers = {"Modal-Session-ID": "openjev-boolq-eval"}
    if is_jev:
        headers = {
            "Authorization": "Bearer "
            + Path("~/.openrouter-api-key").expanduser().read_text().strip(),
            "X-OpenRouter-Title": "OpenJev BoolQ evaluation",
        }
    async with httpx.AsyncClient(base_url=endpoint, timeout=150, headers=headers) as client:
        # Readiness can require a full cold start. Avoid filling an inference queue meanwhile.
        deadline = time.monotonic() + 1200
        while not is_jev:
            try:
                ready = await client.get("/health", timeout=10)
                if ready.is_success:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Backend did not become ready in 20 minutes")
            await asyncio.sleep(5)
        if not is_jev:
            (args.output / "health.json").write_text(ready.text + "\n")
        with predictions.open("a", buffering=1) as stream:

            async def worker():
                while not queue.empty():
                    i, row = queue.get_nowait()
                    begin = time.monotonic()
                    for attempt in range(9):
                        try:
                            response = await client.post(route, json=payload(row, model))
                            response.raise_for_status()
                            data = response.json()
                            p = data["answers"]["answer"]["noul"]
                            assert math.isfinite(p) and 0 <= p <= 1
                            if not is_jev:
                                assert data["model"] == model
                                assert data["usage"]["output_tokens"] == 2
                            break
                        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                                exc.response.status_code in {429, 502, 503, 504, 529}
                            )
                            if not retryable or attempt == 8:
                                raise
                            print(f"Retry index={i}, attempt={attempt + 1}: {exc}", flush=True)
                            await asyncio.sleep(min(30, 2 ** (attempt + 1)))
                    record = {
                        "index": i,
                        "label": row["answer"],
                        "p_yes": p,
                        "passage_hash": hashlib.sha256(row["passage"].encode()).hexdigest(),
                        "latency_ms": round(1000 * (time.monotonic() - begin), 2),
                        "attempts": attempt + 1,
                        "model": data.get("model", model),
                        "usage": data.get("usage"),
                        "server_timing": response.headers.get("server-timing"),
                        "request_id": response.headers.get("x-typesafe-request-id"),
                    }
                    stream.write(json.dumps(record) + "\n")
                    done[i] = record
                    if len(done) % 100 == 0 or len(done) == len(rows):
                        print(
                            f"Completed {len(done)}/{len(rows)} in "
                            f"{time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                    await asyncio.sleep(args.delay)

            async with asyncio.TaskGroup() as group:
                for _ in range(args.concurrency):
                    group.create_task(worker())
    print(f"Saved {predictions}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--provider", choices=["openjev", "jev"], default="openjev")
    parser.add_argument("--output", type=Path, default=Path("evals/runs/boolq"))
    parser.add_argument("--concurrency", type=int, default=2, choices=range(1, 65))
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="Pilot only; 0 means all 3270 rows")
    asyncio.run(collect(parser.parse_args()))
