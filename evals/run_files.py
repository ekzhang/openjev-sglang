"""Run manifests and predictions, shared by collectors and offline reports."""

import json
from datetime import datetime, timezone


def read_jsonl(path, *, missing_ok=False):
    if missing_ok and not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_manifest(path, settings, *, mutable=()):
    """Keep the original manifest on resume; reject changes to the protocol."""
    if path.exists():
        previous = json.loads(path.read_text())
        if any(previous.get(k) != v for k, v in settings.items() if k not in mutable):
            raise ValueError("Different run parameters; choose a new output directory")
        return previous
    manifest = {**settings, "started_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_predictions(path, indices, *, complete=False):
    """Reject duplicate/out-of-sample rows, and missing rows in final reports."""
    rows = read_jsonl(path, missing_ok=not complete)
    indices = list(indices)
    expected = set(indices)
    if len(expected) != len(indices):
        raise ValueError("Run manifest contains duplicate sample indices")
    actual = [r["index"] for r in rows]
    if len(actual) != len(set(actual)) or not set(actual).issubset(expected):
        raise ValueError(f"Duplicate or unexpected prediction indices in {path}")
    if complete and set(actual) != expected:
        raise ValueError(f"Incomplete predictions in {path}")
    return rows


def load_run(path):
    manifest = json.loads((path / "manifest.json").read_text())
    indices = manifest["indices"] if "indices" in manifest else range(manifest["count"])
    rows = load_predictions(path / "predictions.jsonl", indices, complete=True)
    return manifest, sorted(rows, key=lambda row: row["index"])
