"""Direct SGLang probe: does token_ids_logprob work together with image_data?

This validates the single assumption the whole multimodal design rests on, without
involving OpenJev at all.
"""

import base64
import json
import urllib.request

from transformers import AutoTokenizer

MODEL = "/mnt/hf/models/qwen3-vl-30b-awq"
img_b64 = base64.b64encode(open("/tmp/label.png", "rb").read()).decode()
img_url = "data:image/png;base64," + img_b64

tok = AutoTokenizer.from_pretrained(MODEL)
IMAGE_PAD = 151655

msgs = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this shipping label. Answer with one letter."},
            {"type": "image_url", "image_url": {"url": img_url}},
        ],
    }
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok.encode(text, add_special_tokens=False)
print("placeholder count in prompt:", ids.count(IMAGE_PAD), "| prompt tokens:", len(ids))
assert ids.count(IMAGE_PAD) == 1, "exactly one placeholder is required"

label_ids = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
ids = ids + tok.encode("\nAnswer:\n", add_special_tokens=False)

payload = {
    "rid": "vl-probe-1",
    "input_ids": ids,
    "image_data": [img_url],
    "sampling_params": {
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": True,
    },
    "stream": False,
    "return_logprob": True,
    "token_ids_logprob": label_ids,
    "logprob_start_len": -1,
    "top_logprobs_num": 0,
    "return_text_in_logprobs": False,
}

req = urllib.request.Request(
    "http://127.0.0.1:30000/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    data = json.load(urllib.request.urlopen(req, timeout=240))
except Exception as exc:  # noqa: BLE001 - report the server's own message
    body = getattr(exc, "read", lambda: b"")()
    print("REQUEST FAILED:", type(exc).__name__, str(exc)[:200])
    print("body:", body[:1000])
    raise SystemExit(1)

meta = data["meta_info"]
print("prompt_tokens:", meta.get("prompt_tokens"), "completion_tokens:", meta.get("completion_tokens"))
entries = meta.get("output_token_ids_logprobs")
if entries is None:
    print("FAIL: no output_token_ids_logprobs. meta_info keys:", sorted(meta))
    raise SystemExit(1)
by_id = {entry[1]: entry[0] for entry in entries[0]}
probs = {c: by_id.get(i) for c, i in zip("ABCD", label_ids)}
print("selected-token logprobs:", {k: (round(v, 4) if v is not None else None) for k, v in probs.items()})
print("span:", len(meta["input_token_logprobs"]) if meta.get("input_token_logprobs") else "n/a")
print("RESULT: token_ids_logprob + image_data WORKS")
