"""End-to-end: image state through the OpenJev API, with multi-task readout."""

import base64
import json
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
img_b64 = base64.b64encode(open("/tmp/label.png", "rb").read()).decode()

payload = {
    "model": "jev-latest",
    "state": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is a scan of a shipping label."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
            ],
        }
    ],
    "questions": {
        "is_label": {
            "type": "noul",
            "instructions": "Does the image show a shipping label or invoice document?",
            "criteria": {"true": "It is a label or invoice", "false": "It is something else"},
        },
        "has_paid": {
            "type": "noul",
            "instructions": "Does the image contain a PAID stamp or mark?",
            "criteria": {"true": "A PAID mark is visible", "false": "No PAID mark"},
        },
        "has_signature": {
            "type": "noul",
            "instructions": "Is the signature line filled in with an actual signature?",
            "criteria": {"true": "Signed", "false": "Blank or absent"},
        },
        "doc_type": {
            "type": "choice",
            "instructions": "What kind of document is shown?",
            "criteria": {
                "shipping_label": "A shipping or courier label",
                "invoice": "An invoice or receipt",
                "id_card": "An identity document",
                "photo": "An ordinary photograph",
            },
        },
        "total_amount": {
            "type": "choice",
            "instructions": "What invoice total appears on the document?",
            "criteria": {
                "899": "899.00 CNY",
                "199": "199.00 CNY",
                "1000": "1000.00 CNY",
                "none": "No total is shown",
            },
        },
        "legibility": {
            "type": "score",
            "instructions": "How legible is the document text?",
            "criteria": ["Unreadable", "Partly legible", "Mostly legible", "Fully legible"],
        },
    },
}

req = urllib.request.Request(
    base + "/v1/systemone",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
        headers = dict(resp.headers)
except Exception as exc:  # noqa: BLE001
    body = getattr(exc, "read", lambda: b"")()
    print("REQUEST FAILED:", type(exc).__name__, str(exc)[:200])
    print("body:", body[:1000])
    raise SystemExit(1)

print("usage:", data["usage"])
print("server-timing:", headers.get("server-timing"))
print("prefix tokens :", headers.get("x-openjev-prefix-tokens"))
print()
a = data["answers"]
print("is_label     : P=%.5f -> %s" % (a["is_label"]["noul"], a["is_label"]["noul"] > 0.5))
print("has_paid     : P=%.5f -> %s" % (a["has_paid"]["noul"], a["has_paid"]["noul"] > 0.5))
print("has_signature: P=%.5f -> %s" % (a["has_signature"]["noul"], a["has_signature"]["noul"] > 0.5))
print("doc_type     :", a["doc_type"]["choice"],
      {k: round(v, 4) for k, v in sorted(a["doc_type"]["probabilities"].items(), key=lambda x: -x[1])})
print("total_amount :", a["total_amount"]["choice"],
      {k: round(v, 4) for k, v in sorted(a["total_amount"]["probabilities"].items(), key=lambda x: -x[1])})
print("legibility   : score=%.3f" % a["legibility"]["score"], a["legibility"]["legend"])
print()
checks = {
    "recognises a shipping label": a["doc_type"]["choice"] == "shipping_label",
    "reads the 899.00 total": a["total_amount"]["choice"] == "899",
    "sees the PAID stamp": a["has_paid"]["noul"] > 0.5,
    "treats signature line as blank": a["has_signature"]["noul"] < 0.5,
}
for name, ok in checks.items():
    print(("  PASS  " if ok else "  FAIL  ") + name)
print("\nRESULT:", "ALL VISION CHECKS PASSED" if all(checks.values()) else "SOME CHECKS FAILED")
