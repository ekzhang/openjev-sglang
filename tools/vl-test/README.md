# Image-state validation scripts

Scripts used to prove that image state works end to end on a single RTX 4090, plus
their measured results. They are development aids, not part of the package or the
test suite; the unit tests for this feature live in `tests/test_prompts.py` and
`tests/test_api.py`.

Requires a running deployment with a vision checkpoint and `OPENJEV_MAX_IMAGES > 0`
(see the `ada-4090-vl` profile). Start one with `run-openjev-vl.sh`.

## Scripts

| Script | Purpose |
| --- | --- |
| `probe-backend.py` | Hits SGLang `/generate` directly, with `input_ids` **and** `image_data`, and requests `token_ids_logprob`. This is the one check the whole design rests on: if selected-token logprobs did not work with an image attached, the one-token readout could not be used for images at all. |
| `probe-api.py` | Full `/v1/systemone` request with a generated label image and six questions; asserts the model actually reads the document. |
| `clock.sh` | Reads `clock.png` and picks between four candidate times. |
| `clock-request.json` | The same clock request as a fixed payload, so no base64 is needed at call time. |
| `make-label.py` | Generates the synthetic shipping label used by `probe-api.py`. |
| `run-openjev-vl.sh` | Launches OpenJev with the vision model on port 8010. |

## Measured results

`probe-backend.py` — the decisive check:

```
placeholder count in prompt: 1 | prompt tokens: 22
prompt_tokens: 528 completion_tokens: 1
selected-token logprobs: {'A': -4.2739, 'B': -5.7739, 'C': -7.0239, 'D': -6.1489}
RESULT: token_ids_logprob + image_data WORKS
```

`probe-api.py` — a 900x560 synthetic shipping label, six questions in one request:

```
usage: {'input_tokens': 4139, 'output_tokens': 7}
doc_type     : shipping_label  P=0.9989
total_amount : 899             P=1.0
has_paid     : P=0.99994
has_signature: P=0.00461  (blank line correctly reported as unsigned)
```

Control: the same `total_amount` question with **no image** attached answers `none`
(P=0.7607) rather than `899`. The model cannot know the amount from the prompt, so
reading it with the image, and reporting no total without it, is direct evidence the
image reached the vision tower instead of being silently dropped.

`clock.sh` — reading an analog clock, four-way choice:

```
C 2:25  0.9863   ← chosen
D 5:40  0.0097
A 10:50 0.0022
B 12:30 0.0019
```

Stable across runs (0.956-0.986 for C). Independently cross-checked by pixel
analysis: fitting a circle to the dial gives centre (284, 206) and radius 179 px;
the two hand-like structures emit at ~60 degrees (toward "2") and ~148-150 degrees
(toward "5", i.e. 25 minutes). A minute hand at 150 degrees matches only the 2:25
option; the others would need 60, 120 or 180 degrees.

## Notes

- `clock-request.json` embeds a real image, so it is ~36 KB. The request-body limit
  is 2 MiB by default, so inline base64 is comfortable for screenshots and scans.
- A 900x560 image expands from 22 text tokens to 528 prompt tokens, so images
  dominate `usage.input_tokens`. Text-only requests are unaffected: the same
  deployment served the README example in 197 input tokens.
